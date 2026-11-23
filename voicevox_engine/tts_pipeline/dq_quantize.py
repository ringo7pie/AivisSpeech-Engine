"""
声のモデル（.aivmx = ONNX）を 8bit に変換して使うための処理（DungeonQuest の改造。2026-11-23）。

公式の AivisSpeech Engine 1.2.0 に対する改造で、元のモデル（Models フォルダ）には触らない。
- 変換: 静的量子化（QDQ・per-channel・Conv / MatMul / Gemm を int8）。音の波形を作るデコーダー（ノード名が /dec/ で始まる物）は 32bit に残す
  （デコーダーまで 8bit にすると無音の所の雑音が約 15 dB 増えた。残すと元とほぼ同じ）。較正は乱数の入力 8 個。
- 置き場: <データの置き場>/QuantizedModels/<モデルの UUID>-int8.aivmx と、横の <同じ名前>.json（元のファイルの大きさ・更新時刻・SHA-256・変換の版）。
  元のファイルの大きさか更新時刻が替わっていたら、変換済みは使わない（--quantize_models で作り直す）。
- 実測（Ryzen 9 3900X・CPU・onnxruntime 1.32・まお）: エンジンのメモリ 315 → 182 MB/モデル、読み込み 0.45 → 0.66 秒、合成 0.91 → 0.96 秒/文、変換 約 45 秒/モデル。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..logging import logger
from ..utility.path_utility import get_save_dir

QUANTIZE_VERSION = 1  # 変換の手順を替えたら上げる（古い変換済みを使わない）。
EXCLUDE_PATTERN = r"^/dec/"  # 32bit に残すノード（デコーダー）。
CALIBRATION_COUNT = 8
CALIBRATION_LENGTH = 60


def quantized_dir() -> Path:
    return get_save_dir() / "QuantizedModels"


def quantized_paths(aivm_model_uuid: str) -> tuple[Path, Path]:
    model = quantized_dir() / f"{aivm_model_uuid}-int8.aivmx"
    return model, model.with_suffix(".json")


def _source_stamp(source: Path) -> dict[str, Any]:
    stat = source.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def find_quantized(aivm_model_uuid: str, source: Path) -> Path | None:
    """変換済みがあり、元のファイルと版が合っていればそのパス。無ければ None。"""
    model, marker = quantized_paths(aivm_model_uuid)
    if not model.exists() or not marker.exists():
        return None
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    stamp = _source_stamp(source)
    if info.get("version") != QUANTIZE_VERSION or info.get("size") != stamp["size"] or info.get("mtime_ns") != stamp["mtime_ns"]:
        return None
    return model


def _feeds_for(session: Any, seed: int) -> dict[str, np.ndarray]:
    """Style-Bert-VITS2 の入力を乱数で作る（batch 1・音素 CALIBRATION_LENGTH 個。揺らぎは 0）。"""
    rng = np.random.default_rng(seed)
    scalars = {"length_scale": 1.0, "sdp_ratio": 0.0, "noise_scale": 0.0, "noise_scale_w": 0.0}
    feeds: dict[str, np.ndarray] = {}
    for value in session.get_inputs():
        name = value.name
        shape = []
        for dim in value.shape:
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            elif "batch" in str(dim):
                shape.append(1)
            else:
                shape.append(CALIBRATION_LENGTH)
        if name in scalars:
            feeds[name] = np.array(scalars[name], dtype=np.float32)
        elif name == "x_tst":
            feeds[name] = rng.integers(1, 80, size=shape, dtype=np.int64)
        elif name == "x_tst_lengths":
            feeds[name] = np.array([CALIBRATION_LENGTH], dtype=np.int64)
        elif name == "sid":
            feeds[name] = np.zeros(shape, dtype=np.int64)
        elif name == "tones":
            feeds[name] = rng.integers(0, 2, size=shape, dtype=np.int64)
        elif name == "language":
            feeds[name] = np.ones(shape, dtype=np.int64)
        else:
            feeds[name] = (rng.standard_normal(size=shape) * 0.1).astype(np.float32)
    return feeds


class _RandomReader:
    def __init__(self, model_path: Path) -> None:
        import onnxruntime

        session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._items = iter([_feeds_for(session, 100 + index) for index in range(CALIBRATION_COUNT)])

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._items, None)


def quantize_model(aivm_model_uuid: str, source: Path) -> Path:
    """source を変換して置き場に書き、変換済みのパスを返す（失敗は例外）。"""
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    model_path, marker = quantized_paths(aivm_model_uuid)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prepared = model_path.with_suffix(".pre.onnx")
    partial = model_path.with_suffix(".partial")
    started = time.time()
    try:
        quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)
        excluded = [node.name for node in onnx.load(str(prepared), load_external_data=False).graph.node if re.search(EXCLUDE_PATTERN, node.name)]
        quantize_static(
            str(prepared),
            str(partial),
            _RandomReader(prepared),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            nodes_to_exclude=excluded,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
        # AIVM の説明（マニフェスト・ハイパーパラメータ・スタイルベクトル）を元のまま写す（量子化はこれを落とす）。
        original = onnx.load(str(source), load_external_data=False)
        quantized = onnx.load(str(partial))
        del quantized.metadata_props[:]
        for prop in original.metadata_props:
            entry = quantized.metadata_props.add()
            entry.key = prop.key
            entry.value = prop.value
        onnx.save(quantized, str(partial))
        partial.replace(model_path)
    finally:
        prepared.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    info = {"version": QUANTIZE_VERSION, "source": source.name, "sha256": digest.hexdigest(), **_source_stamp(source),
            "exclude": EXCLUDE_PATTERN, "seconds": round(time.time() - started, 1)}
    marker.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return model_path


def quantize_installed_models(aivm_infos: dict[str, Any]) -> int:
    """入っている声のモデルのうち、変換済みが無い（古い）物を全部変換する。失敗した数を返す。ログは DQ_QUANTIZE の行。"""
    failures = 0
    items = list(aivm_infos.items())
    for index, (aivm_model_uuid, aivm_info) in enumerate(items, start=1):
        source = Path(aivm_info.file_path)
        name = aivm_info.manifest.name
        if find_quantized(aivm_model_uuid, source) is not None:
            logger.info(f"DQ_QUANTIZE {index}/{len(items)} status=present name={name} uuid={aivm_model_uuid}")
            continue
        started = time.time()
        try:
            quantize_model(aivm_model_uuid, source)
            logger.info(f"DQ_QUANTIZE {index}/{len(items)} status=created name={name} uuid={aivm_model_uuid} seconds={time.time() - started:.1f}")
        except Exception as ex:  # 1 つ失敗しても残りを続ける（その声は 32bit のまま使う）。
            failures += 1
            logger.error(f"DQ_QUANTIZE {index}/{len(items)} status=failed name={name} uuid={aivm_model_uuid}", exc_info=ex)
    return failures

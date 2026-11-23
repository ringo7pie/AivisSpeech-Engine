# AivisSpeech Engine（DungeonQuest の改造版 1.2.0+dq.1）

このブランチ `dq/ort132-int8` は、[AivisSpeech Engine](https://github.com/Aivis-Project/AivisSpeech-Engine) 1.2.0 に、ゲーム「Demon of the Crown」で使うための小さな改造を加えた物です。ライセンスは元と同じ LGPL-3.0 です。元の README は `README.md` にあります。

## 変更点

- Windows の onnxruntime を `onnxruntime-directml` 1.24 から `onnxruntime`（CPU）1.32 に替えました。1.31 以前の onnxruntime は、声のモデル（約 250 MB）の推論セッションの作成に約 5 秒かかり、1.32 では約 0.5 秒です。DirectML（`--use_gpu`）は使えません。
- `--quantize_models`: 入っている声のモデルを 8bit（静的量子化・QDQ・チャンネルごと。音の波形を作るデコーダーは 32bit のまま）に変換して `<データの置き場>/QuantizedModels` に置き、サーバーを起動せずに終わります。元のモデル（`Models`）には触りません。
- `--use_quantized_models`: 変換済みのモデルがあれば、そちらを読み込みます。声のモデル 1 つあたりのメモリが約 290 MB から約 175 MB になります。
- 変更したファイル: `pyproject.toml`・`uv.lock`・`run.py`・`run.spec`・`voicevox_engine/__init__.py`・`voicevox_engine/tts_pipeline/style_bert_vits2_tts_engine.py`・`voicevox_engine/tts_pipeline/dq_quantize.py`（新規）。

## ビルド

```
uv sync --group dev --group build
uv run pyinstaller --noconfirm run.spec
```

出力は `dist/run`。配布の 7z には、`run.exe` の隣に目印の `dq_engine.json` を入れます（ゲームの起動の台本が、改造版かどうかを見分けるため）。

音声合成モデルの利用は、各モデルのライセンス（ACML など）に従ってください。
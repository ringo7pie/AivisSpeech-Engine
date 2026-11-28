"""DungeonQuest の改造（1.2.0+dq.3）: ONNX Runtime の CPU 推論のスレッドの数と、待つ間のスピンを決める。

既定の ONNX Runtime は、推論のスレッドを物理コアの数だけ作り、仕事を待つ間もスピンして CPU を回し続ける。
同じ PC でゲームを動かすと、声の合成の間にゲームの描画・処理のスレッドが CPU を取られてフレームが落ちる。
`configure_onnx_threads(threads)` を、推論セッションを作る前（StyleBertVITS2TTSEngine の初期化より前）に呼ぶと、
以後に作られる `onnxruntime.SessionOptions()` すべて（Style-Bert-VITS2 の音声合成モデルと BERT モデル）に、
intra_op_num_threads = threads・inter_op_num_threads = 1・スピンなしを入れる。
"""

from typing import Any

import onnxruntime

_original_session_options: Any = None
_configured_threads: int = 0


def configure_onnx_threads(threads: int) -> None:
    """以後の SessionOptions のスレッドの数を threads にし、スピンを止める（threads が 0 以下なら何もしない）。"""

    global _original_session_options, _configured_threads
    if threads <= 0:
        return
    if _original_session_options is None:
        _original_session_options = onnxruntime.SessionOptions
    original = _original_session_options
    _configured_threads = threads

    def limited_session_options() -> Any:
        options = original()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        return options

    onnxruntime.SessionOptions = limited_session_options  # type: ignore[misc]


def configured_threads() -> int:
    """configure_onnx_threads で入れたスレッドの数（入れていなければ 0）。"""

    return _configured_threads

"""AivisSpeech Engine"""

import os

__version__ = "1.2.0+dq.1"  # DungeonQuest の改造版（onnxruntime 1.32・8bit の声のモデル）


# 環境変数 TRANSFORMERS_NO_ADVISORY_WARNINGS を設定して transformers の warning_advice() を無効化
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

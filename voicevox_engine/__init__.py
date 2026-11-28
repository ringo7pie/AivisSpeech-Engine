"""AivisSpeech Engine"""

import os

__version__ = "1.2.0+dq.3.dml"  # DungeonQuest の改造版の GPU（DirectML）版（onnxruntime-directml 1.24.4）


# 環境変数 TRANSFORMERS_NO_ADVISORY_WARNINGS を設定して transformers の warning_advice() を無効化
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

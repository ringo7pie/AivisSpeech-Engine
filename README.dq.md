# AivisSpeech Engine（DungeonQuest の改造版 1.2.0+dq.3.dml・GPU 版）

このブランチ `dq/dml` は、`dq/ort132-int8`（1.2.0+dq.3）の onnxruntime を `onnxruntime-directml` 1.24.4 に替えた GPU（DirectML）版です。`--use_gpu` で GPU（Direct3D のアダプターの 0 番）に載せます。DirectML で 8bit のモデル（`--use_quantized_models`）を合成するとエンジンが落ちるので、GPU 版では使いません。CPU で使うと、声のモデルの読み込みは 1 つ約 5 秒かかります（1.32 の CPU 版は約 0.5 秒）。以下は `dq/ort132-int8` の説明です。

このブランチ `dq/ort132-int8` は、[AivisSpeech Engine](https://github.com/Aivis-Project/AivisSpeech-Engine) 1.2.0 に、ゲーム「Demon of the Crown」で使うための小さな改造を加えた物です。ライセンスは元と同じ LGPL-3.0 です。元の README は `README.md` にあります。

## 変更点

- Windows の onnxruntime を `onnxruntime-directml` 1.24 から `onnxruntime`（CPU）1.32 に替えました。1.31 以前の onnxruntime は、声のモデル（約 250 MB）の推論セッションの作成に約 5 秒かかり、1.32 では約 0.5 秒です。DirectML（`--use_gpu`）は使えません。
- `--quantize_models`: 入っている声のモデルを 8bit（静的量子化・QDQ・チャンネルごと。音の波形を作るデコーダーは 32bit のまま）に変換して `<データの置き場>/QuantizedModels` に置き、サーバーを起動せずに終わります。元のモデル（`Models`）には触りません。
- `--use_quantized_models`: 変換済みのモデルがあれば、そちらを読み込みます。声のモデル 1 つあたりのメモリが約 290 MB から約 175 MB になります。
- `--disable_aivishub`（1.2.0+dq.2。環境変数 `AIVISSPEECH_DISABLE_AIVISHUB=1` でも同じ）: AivisHub（`https://api.aivis-project.com/v1`）へつなぎません。付けると、起動のたびに行う 4 つの通信（既定モデルの一覧の取得と自動のインストール・上書き更新、強制削除の規則の取得と適用、各モデルの更新の確認、起動イベントの送信）を行わず、ソースに書かれた予備の値（既定モデル 2 つ・強制削除の規則 3 つ）も使いません。入っている声のモデルは消えず、書き換わりません。URL からのモデルのインストール（`POST /aivm_models/install` の `url`）とモデルの更新（`POST /aivm_models/{uuid}/update`）は 403 で断ります（ファイルを渡すインストールは使えます）。起動のログに `DQ_AIVISHUB disabled=True` と出ます。付けないときの動きは元と同じです。
- `--onnx_threads N`（1.2.0+dq.3。環境変数 `AIVISSPEECH_ONNX_THREADS` でも同じ）: CPU 推論のスレッドを N 本にし、仕事を待つ間に CPU を回し続けるスピンを止めます（推論セッションを作る前に、以後の `onnxruntime.SessionOptions()` に `intra_op_num_threads = N`・`inter_op_num_threads = 1`・`session.intra_op.allow_spinning = 0` を入れます）。元の動き（物理コアの数のスレッド・スピンあり）では、同じ PC で動くゲームが合成の間に CPU を取られてフレームが落ちました。起動のログに `DQ_ONNX_THREADS threads=N spinning=False` と出ます。付けないときの動きは元と同じです。
- 変更したファイル: `pyproject.toml`・`uv.lock`・`run.py`・`run.spec`・`voicevox_engine/__init__.py`・`voicevox_engine/tts_pipeline/style_bert_vits2_tts_engine.py`・`voicevox_engine/tts_pipeline/dq_quantize.py`（新規）・`voicevox_engine/dq_threads.py`（新規）・`voicevox_engine/utility/aivishub_client.py`・`voicevox_engine/aivm_manager.py`。
- GPU（DirectML）版はブランチ `dq/dml` にあります（onnxruntime-directml 1.24.4。`--use_gpu` で GPU の 0 番のアダプターに載せます。8bit のモデルは使いません）。

## ビルド

```
uv sync --group dev --group build
uv run pyinstaller --noconfirm run.spec
```

出力は `dist/run`。配布の 7z には、`run.exe` の隣に目印の `dq_engine.json` を入れます（ゲームの起動の台本が、改造版かどうかを見分けるため）。

音声合成モデルの利用は、各モデルのライセンス（ACML など）に従ってください。
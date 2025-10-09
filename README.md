# 釧路イベント：秋の虫とシシオドシ（シミュ＋実機）

本リポジトリは「秋の虫とシシオドシ」をテーマに、シミュレーションと実機LED/音を統合制御するための最小構成です。CLIからシミュ・実機・色ツールを操作できます。

## セットアップ

```bash
pip install -r requirements.txt
```

Arduino 側（fastled.ino）を書き込み、`NUM_PHYSICAL_LEDS=1200`, `MAGIC_BYTE=0x7E` を維持。`baud` は `config.yaml` と一致させてください（推奨 921600）。

## 主なコマンド

```bash
# 実機シリアル確認
python -m src.cli serial_ping

# レイアウトJSON出力
python -m src.cli layout_export

# 音源/色を一覧
python -m src.cli audio_check
python -m src.cli config_insects

# 音圧を均一化（EBU R128 loudnorm、元ファイルはバックアップに退避）
python -m src.cli audio_normalize --dir assets/data/sound/trimmed --backup assets/data/sound/backup_originals \\
  --lufs -16 --tp -1.0 --lra 11 --bitrate 192k

# アンビエント（画面表示＋実機ミラー送信対応）
python -m src.cli ambient --mirror            # 画面と同じフレームを実機送信（MAGIC 0x7E）

# アンビエント録音（WAVファイルへ）
python -m src.cli ambient_record --out recordings/ambient.wav \
  --seconds 120 --species 3 --change-interval 60 --wave-seconds 20 --density 1.0

# 全消灯
python -m src.cli black

# 色ツール（MAGIC 0x7E フレーム）
python -m src.cli colors pick    # カラーホイールで直接送信
python -m src.cli colors insect  # 種名を表示しつつ color を実機送信
```

## 設定（config.yaml）

- `serial_port`: Arduino R4 のシリアルパス
- `baud`: 通信速度（スケッチと一致）
- `leds_per_meter`: LED 本数/メートル（60 推奨）
- `segments_m`: U字の区間長（left/bottom/right）
- `audio.gain_master`: 全体ゲイン（dB）
- `sim.chirp_interval_min_s`/`max_s`: 個体の鳴き間隔範囲（秒）
- `sim.max_concurrent_total`: 同時鳴き上限（全体）
- `sim.max_concurrent_per_species`: 同時鳴き上限（種別）

検出（マイク/カメラ/AI）
- `detect.mode`: `timer` | `mic` | `cam` | `auto`（`timer`は内部スケジューラ、`mic`/`cam`は外部検出で置換）
- `detect.mic.*`: 入力サンプリング、解析帯域、`threshold_db`/`release_db`、`min_interval_ms`、`device`
  - 既定はノイズ対策でやや厳しめ（`band_hz: [2000, 9000]`, `threshold_db: -6`, `release_db: -12`, `min_interval_ms: 1800`）
- `detect.ai.*`: YAMNet TFLite を使ったAI検出設定（`model_path`, `label_map_csv`, `target_labels`, `threshold`, `release`, ほか）
- `detect.cam.*`: ROI、角度しきい、`min_interval_ms`
- `fallback_trigger`: 検出が初期化できない場合に `timer` を許可（true/false）

## 振る舞い（概要）

- IDLE: 虫（100匹×12LED）が“たまに”鳴き、単色の輝度変化で明滅
- SILENCE: カコンで即ミュート、effect_version に応じた静寂演出
- WAVE/Calm: 橙波（v1）または青い静寂（v2）が進行/保持
- RESUME: 鳴きは少数→多数へ徐々に復帰。同時鳴き上限は段階的に増加

（検出器関連ツールは削除済み）

録音について
- `ambient_record` はアンビエントのスケジューラをヘッドレスで実行し、虫の音をミックスして WAV に書き出します。
- パラメータは `ambient` と同様（`--species`, `--change-interval`, `--wave-seconds`, `--density`）。
- 出力先は `--out`、長さは `--seconds` で指定できます（既定 60 秒）。

## ミラー送信について

`--mirror` 付きアンビエントは、画面に描画した LED カラーをそのまま `MAGIC_BYTE(0x7E)` フレームで実機へ送信します。スケッチは 1ピクセル=3物理LED の展開で描画します。

## ライセンス

プロジェクト固有の要件に従います。
AI検出（YAMNet TFLite）
- 事前に `yamnet.tflite` と `yamnet_class_map.csv` を取得して `assets/models/` に配置してください。
- `config.yaml` の `detect.mode: ai` を設定し、`detect.ai.model_path`/`label_map_csv`/`target_labels` を確認。
- 運用は `python -m src.cli detect_watch`（AIでの検出イベントに応じて波を送信）。
- 依存: `tflite-runtime`, `sounddevice`。

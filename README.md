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
python -m src.cli audio_normalize --dir assets/data/sound/trimmed --backup assets/data/sound/backup_originals \
  --lufs -16 --tp -1.0 --lra 11 --bitrate 192k

# シミュレーション（画面表示のみ / 実機ミラー送信あり）
python -m src.cli show --version 1         # 橙の波
python -m src.cli show --version 2         # 青い静寂
python -m src.cli show --version 2 --mirror  # 画面と同じフレームを実機送信（MAGIC 0x7E）

# 実機ランナー（自動カコン・音・LED）
python -m src.cli run --version 1
python -m src.cli run --version 2

# 手動トリガ／全消灯
python -m src.cli trigger_once
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
- `wave.speed_mps`: 波の伝播速度（version 1）
- `wave.tail_m`: 波の余韻長（m）
- `wave.pause_ms`: カコン検出後、波開始までの無音時間（ms）
- `audio.gain_master`: 全体ゲイン（dB）
- `sim.kakon_mean_s`/`kakon_std_s`: カコンの平均間隔/分散（秒）
- `sim.kakon_wave_speed_factor`: カコン時の波速度倍率（version 1）
- `sim.chirp_interval_min_s`/`max_s`: 個体の鳴き間隔範囲（秒）
- `sim.max_concurrent_total`: 同時鳴き上限（全体）
- `sim.max_concurrent_per_species`: 同時鳴き上限（種別）
- `sim.active_species_count`: カコンごとに有効化する種数（ランダム選定）
- `sim.effect_version`: 1=橙の波, 2=青い静寂
- `sim.resume_seconds`: 静寂後の復帰にかける時間（秒）
- `sim.calm_blue_rgb`: 青い静寂の色（RGB）
- `sim.calm_hold_s`: 青い静寂の保持時間（秒）

## 振る舞い（概要）

- IDLE: 虫（100匹×12LED）が“たまに”鳴き、単色の輝度変化で明滅
- SILENCE: カコンで即ミュート、effect_version に応じた静寂演出
- WAVE/Calm: 橙波（v1）または青い静寂（v2）が進行/保持
- RESUME: 鳴きは少数→多数へ徐々に復帰。同時鳴き上限は段階的に増加

## ミラー送信について

`--mirror` 付きシミュレーションは、画面に描画した LED カラーをそのまま `MAGIC_BYTE(0x7E)` フレームで実機へ送信します。スケッチは 1ピクセル=3物理LED の展開で描画します。

## ライセンス

プロジェクト固有の要件に従います。

// fastled.ino (デバッガー対応版)

#include <FastLED.h>

// =========================================================================
// === CONFIGURATION - 環境に合わせてこれらの値を変更してください ===
// =========================================================================

// --- LED Strip Hardware Settings ---
// あなたがArduinoに接続している物理的なLEDの総数
#define NUM_PHYSICAL_LEDS    1200      // ★ Pythonのsettings.yamlの 'num_leds' と一致させる

#define DATA_PIN             6        // Arduinoのデータピン番号
#define LED_TYPE             WS2811   // LEDストリップのチップセット
#define COLOR_ORDER          BRG      // ★お使いのLEDに合わせてBRG, GRBなどを設定

// --- Communication Settings ---
// Pythonのsettings.yamlにある値と完全に一致させる必要があります
#define BAUD_RATE            921600   // 通信速度
#define MAGIC_BYTE           0x7E     // '˜' - データフレームの開始を識別するバイト

// =========================================================================
// === DO NOT EDIT BELOW THIS LINE - 以下のコードは編集不要です ===
// =========================================================================

// 制御可能な論理ピクセル数を計算 (1ピクセル = 3 LED)
// (NUM_PHYSICAL_LEDS + 2) / 3 は、割り切れない場合でも正しいピクセル数を計算する常套手段です。
const int NUM_PIXELS = (NUM_PHYSICAL_LEDS + 2) / 3;

// FastLEDライブラリ用のLED配列 (物理LED数で確保)
CRGB leds[NUM_PHYSICAL_LEDS];

// PCからのデータを受信するためのバッファ (論理ピクセル数で確保)
byte buffer[NUM_PIXELS * 3];

void setup() {
  Serial.begin(BAUD_RATE);

  FastLED.addLeds<LED_TYPE, DATA_PIN, COLOR_ORDER>(leds, NUM_PHYSICAL_LEDS)
         .setCorrection(TypicalSMD5050);

  FastLED.setBrightness(100); // 明るさを少し上げる

  FastLED.clear();
  FastLED.show();
  delay(2000); // Arduinoの起動を待つ
}

void loop() {
  if (Serial.available() > 0) {
    if (Serial.read() == MAGIC_BYTE) {
      
      // 期待するデータ長 (ピクセル数 * 3バイト/ピクセル)
      int expectedBytes = NUM_PIXELS * 3;
      int bytesRead = Serial.readBytes(buffer, expectedBytes);
      
      if (bytesRead == expectedBytes) {
        
        for (int i = 0; i < NUM_PIXELS; i++) {
          // バッファからi番目のピクセルの色(R, G, B)を読み込む
          CRGB pixelColor = CRGB(buffer[i * 3], buffer[i * 3 + 1], buffer[i * 3 + 2]);
          
          int led_index = i;
          
          // 配列の範囲外に書き込まないように、安全チェックを行う
          if (led_index < NUM_PHYSICAL_LEDS)     leds[led_index] = pixelColor;
          if (led_index + 1 < NUM_PHYSICAL_LEDS) leds[led_index + 1] = pixelColor;
          if (led_index + 2 < NUM_PHYSICAL_LEDS) leds[led_index + 2] = pixelColor;
        }
        
        FastLED.show();
      }
    }
  }
}
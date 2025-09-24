# debug_led_mapper.py

import yaml
import numpy as np
import time
import os
from src.serial_handler import SerialWriterThread

def led_mapping_debugger():
    """
    A dedicated tool to visually debug the mapping between logical pixels 
    and physical LEDs. It lights up one pixel at a time.
    """
    # 1. Load essential settings from settings.yaml
    try:
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.yaml")
        with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)

        SERIAL_PORT = settings['serial_port']
        BAUD_RATE = settings['baud_rate']
        
        # ★最重要★: num_ledsは物理LEDの総数を指定してください (例: 300)
        # このスクリプトは、この値から制御ピクセル数を計算します。
        NUM_PHYSICAL_LEDS = settings['num_leds']
        # 3つのLEDで1ピクセルなので、制御するピクセル数を計算
        NUM_PIXELS = (NUM_PHYSICAL_LEDS + 2) // 3

        MAGIC_BYTE = 0x7E
        
        print("--- LED Pixel Mapper Debugger ---")
        print(f"Loaded settings for port '{SERIAL_PORT}' at {BAUD_RATE} baud.")
        print(f"Physical LEDs: {NUM_PHYSICAL_LEDS} -> Logical Pixels: {NUM_PIXELS}")
        print("-----------------------------------")

    except Exception as e:
        print(f"FATAL: Error loading settings from 'settings.yaml'. Error: {e}")
        return

    # 2. Start the serial communication thread
    serial_thread = SerialWriterThread(SERIAL_PORT, BAUD_RATE, MAGIC_BYTE, NUM_PIXELS)
    serial_thread.start()

    # もしArduinoとの接続に失敗したら、スレッドは自動的に終了する
    time.sleep(2.5) # Arduinoの起動とシリアルスレッドの接続確立を待つ
    if not serial_thread.running:
        print("Exiting due to connection failure.")
        return

    # 3. Main debug loop
    current_pixel_index = 0
    try:
        while True:
            # Create an array of colors, all black
            colors = np.zeros((NUM_PIXELS, 3), dtype=int)
            
            # Set the current pixel to white
            colors[current_pixel_index] = [255, 255, 255]
            
            # Send the color data to the Arduino
            serial_thread.send(colors)
            
            # Print the current state to the console
            print(f"Testing pixel {current_pixel_index}/{NUM_PIXELS - 1}... Press Enter to move to the next pixel, or Ctrl+C to exit.")
            
            # Wait for user to press Enter
            input()

            # Move to the next pixel, and loop back to the start if at the end
            current_pixel_index = (current_pixel_index + 1) % NUM_PIXELS

    except KeyboardInterrupt:
        print("\nExiting debugger.")
    finally:
        # Turn off all LEDs before exiting
        serial_thread.send(np.zeros((NUM_PIXELS, 3), dtype=int))
        time.sleep(0.1)
        serial_thread.close()
        serial_thread.join()
        print("Cleanup complete.")

if __name__ == '__main__':
    led_mapping_debugger()
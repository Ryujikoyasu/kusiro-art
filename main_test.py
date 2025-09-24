# hardware_test_gradient.py

import serial
import time
import numpy as np
import yaml
import os

def hsv_to_rgb(h, s, v):
    """HSV to RGB color conversion"""
    if s == 0.0: return np.array([v, v, v]) * 255
    i = int(h * 6.)
    f = (h * 6.) - i
    p, q, t = v * (1. - s), v * (1. - s * f), v * (1. - s * (1. - f))
    i %= 6
    if i == 0: rgb = np.array([v, t, p])
    elif i == 1: rgb = np.array([q, v, p])
    elif i == 2: rgb = np.array([p, v, t])
    elif i == 3: rgb = np.array([p, q, v])
    elif i == 4: rgb = np.array([t, p, v])
    elif i == 5: rgb = np.array([v, p, q])
    return rgb * 255

def main():
    """
    Sends a moving rainbow gradient to the Arduino to test the full LED strip.
    """
    try:
        # --- 1. Load configuration from settings.yaml ---
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.yaml")
        with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)

        SERIAL_PORT = settings['serial_port']
        BAUD_RATE = settings['baud_rate']
        
        # COLOR_ORDER は fastled.ino から直接取得 (例: "BRG")
        COLOR_ORDER = "BRG" 
        
        # Test mode settings determine the number of pixels
        is_test_mode = settings.get('enable_test_mode', False)
        num_physical_leds = settings.get('test_strip_led_count') if is_test_mode else settings['num_leds']
        num_pixels = num_physical_leds // 3
        
        magic_byte = 0x7E

        print("--- Hardware Gradient Test ---")
        print(f"Port: {SERIAL_PORT}, Baud: {BAUD_RATE}")
        print(f"Physical LEDs: {num_physical_leds} -> Logical Pixels: {num_pixels}")
        print(f"Color Order: {COLOR_ORDER}")
        print("Press Ctrl+C to stop.")

        # --- 2. Connect to Arduino ---
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for Arduino to reset

        # --- 3. Main loop to send gradient data ---
        start_time = time.time()
        while True:
            elapsed_time = time.time() - start_time
            
            # Create a color array for all pixels
            colors = np.zeros((num_pixels, 3), dtype=np.uint8)
            
            for i in range(num_pixels):
                # Calculate hue based on position and time
                hue = (i / num_pixels + elapsed_time * 0.1) % 1.0
                # Convert HSV to RGB
                rgb = hsv_to_rgb(hue, 1.0, 1.0)
                
                # Apply the COLOR_ORDER
                if COLOR_ORDER == "BRG":
                    colors[i] = [rgb[2], rgb[0], rgb[1]] # B, R, G
                elif COLOR_ORDER == "GRB":
                    colors[i] = [rgb[1], rgb[0], rgb[2]] # G, R, B
                else: # Default to RGB
                    colors[i] = rgb

            # Prepare and send the packet
            packet = bytearray([magic_byte]) + colors.tobytes()
            ser.write(packet)
            
            time.sleep(1 / 30) # Limit to ~30 FPS

    except KeyboardInterrupt:
        print("\nStopping test.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            # Turn off all LEDs before closing
            off_colors = np.zeros((num_pixels, 3), dtype=np.uint8)
            packet = bytearray([magic_byte]) + off_colors.tobytes()
            try:
                ser.write(packet)
            except:
                pass
            ser.close()
            print("Serial port closed.")

if __name__ == '__main__':
    main()
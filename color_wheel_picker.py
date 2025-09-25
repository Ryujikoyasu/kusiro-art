# color_wheel_picker.py (高機能版：輝度スライダー付き)

import pygame
import numpy as np
import os
import colorsys
import time
from src.serial_handler import SerialWriterThread
from src.util.config import load_config

# --- 設定の読み込み（config.yaml を使用） ---
try:
    cfg = load_config()
    SERIAL_PORT = cfg.get('serial_port')
    BAUD_RATE = int(cfg.get('baud', 115200))
    leds_per_m = int(cfg.get('leds_per_meter', 60))
    seg = cfg.get('segments_m', {"left": 9.5, "bottom": 1.0, "right": 9.5})
    NUM_ACTIVE_LEDS = int(round((float(seg['left']) + float(seg['bottom']) + float(seg['right'])) * leds_per_m))
    NUM_ACTIVE_PIXELS = (NUM_ACTIVE_LEDS + 2) // 3
    MAGIC_BYTE = 0x7E
    VIEW_WIDTH, VIEW_HEIGHT = 900, 700
except Exception as e:
    print(f"FATAL: Error loading settings from 'config.yaml'. Error: {e}")
    raise

def create_color_wheel(diameter):
    """(変更なし)"""
    radius, surface = diameter // 2, pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    for y in range(diameter):
        for x in range(diameter):
            dx, dy = x - radius, y - radius
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= radius:
                hue = (np.arctan2(-dy, dx) / (2 * np.pi)) + 0.5
                sat = dist / radius
                rgb = colorsys.hsv_to_rgb(hue, sat, 1.0)
                surface.set_at((x, y), tuple(int(c * 255) for c in rgb))
    return surface

def interactive_color_picker():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
    pygame.display.set_caption("Color Picker with Brightness Slider (MAGIC 0x7E)")
    font = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    serial_thread = SerialWriterThread(SERIAL_PORT, BAUD_RATE, MAGIC_BYTE, NUM_ACTIVE_PIXELS)
    serial_thread.start()

    # --- UI要素のジオメトリ ---
    wheel_diameter = int(min(VIEW_WIDTH, VIEW_HEIGHT) * 0.7)
    wheel_surface = create_color_wheel(wheel_diameter)
    wheel_rect = wheel_surface.get_rect(center=(VIEW_WIDTH // 2 - 50, VIEW_HEIGHT // 2 + 50))

    slider_height = wheel_diameter
    slider_width = 40
    slider_rect = pygame.Rect(0, 0, slider_width, slider_height)
    slider_rect.midleft = (wheel_rect.right + 40, wheel_rect.centery)

    # --- 状態変数 (HSVをマスターとして管理) ---
    current_hsv = [0.0, 0.0, 0.0]  # [Hue, Saturation, Value], Mutable list
    selected_color_rgb = (0, 0, 0) # 派生するRGB値
    
    input_texts = ["0", "0", "0"]
    input_rects = [pygame.Rect(0, 0, 80, 40) for _ in range(3)]
    active_input_index = -1
    total_width = (input_rects[0].width * 3) + 80
    start_x = (VIEW_WIDTH - total_width) // 2
    for i in range(3):
        input_rects[i].center = (start_x + (input_rects[i].width + 40) * i + input_rects[i].width // 2, 120)

    is_dragging_slider = False
    running = True

    while running:
        # --- イベント処理 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                active_input_index = -1
                if wheel_rect.collidepoint(event.pos):
                    # ホイールクリックでHとSを更新
                    local_pos = (event.pos[0] - wheel_rect.left, event.pos[1] - wheel_rect.top)
                    dx, dy = local_pos[0] - wheel_diameter // 2, local_pos[1] - wheel_diameter // 2
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= wheel_diameter // 2:
                        current_hsv[0] = (np.arctan2(-dy, dx) / (2 * np.pi)) + 0.5
                        current_hsv[1] = dist / (wheel_diameter // 2)

                elif slider_rect.collidepoint(event.pos):
                    # スライダークリックでVを更新
                    is_dragging_slider = True
                    current_hsv[2] = 1.0 - (event.pos[1] - slider_rect.top) / slider_height
                    current_hsv[2] = np.clip(current_hsv[2], 0.0, 1.0)
                
                for i in range(3):
                    if input_rects[i].collidepoint(event.pos):
                        active_input_index = i
                        break
            
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                is_dragging_slider = False
            
            if event.type == pygame.MOUSEMOTION and is_dragging_slider:
                current_hsv[2] = 1.0 - (event.pos[1] - slider_rect.top) / slider_height
                current_hsv[2] = np.clip(current_hsv[2], 0.0, 1.0)
            
            if event.type == pygame.KEYDOWN and active_input_index != -1:
                current_text = input_texts[active_input_index]
                if event.key == pygame.K_BACKSPACE: input_texts[active_input_index] = current_text[:-1]
                elif event.unicode.isdigit(): input_texts[active_input_index] = (current_text + event.unicode)[:3]
                try:
                    new_rgb = [int(t or '0') for t in input_texts]
                    new_rgb = [np.clip(c, 0, 255) for c in new_rgb]
                    # RGBからHSVに変換してマスターの状態を更新
                    current_hsv = list(colorsys.rgb_to_hsv(*[c/255.0 for c in new_rgb]))
                except ValueError: pass

        # --- 状態から表示を計算 ---
        # 1. マスターのHSVから現在のRGB色を計算
        selected_color_rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*current_hsv))
        
        # 2. RGB入力ボックスのテキストを更新
        if active_input_index == -1 and not is_dragging_slider: # ユーザーが入力中でないときだけ
            input_texts = [str(c) for c in selected_color_rgb]

        # 3. データ送信
        color_data = np.tile(np.array(selected_color_rgb), (NUM_ACTIVE_PIXELS, 1))
        serial_thread.send(color_data)

        # --- 画面描画 ---
        screen.fill((40, 40, 40))
        screen.blit(wheel_surface, wheel_rect)

        # スライダー描画
        for y in range(slider_height):
            v = 1.0 - y / slider_height
            slider_color_rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(current_hsv[0], current_hsv[1], v))
            pygame.draw.line(screen, slider_color_rgb, (slider_rect.left, slider_rect.top + y), (slider_rect.right, slider_rect.top + y))
        pygame.draw.rect(screen, (200, 200, 200), slider_rect, 2, 4)
        handle_y = slider_rect.top + (1.0 - current_hsv[2]) * slider_height
        pygame.draw.line(screen, (255, 255, 255), (slider_rect.left - 5, handle_y), (slider_rect.right + 5, handle_y), 3)

        # ホイールマーカー描画
        if current_hsv[1] > 0.01: # 彩度があるときだけ
            angle = (current_hsv[0] - 0.5) * 2 * np.pi
            radius = current_hsv[1] * wheel_diameter / 2
            marker_x = wheel_rect.centerx + radius * np.cos(angle)
            marker_y = wheel_rect.centery - radius * np.sin(angle)
            pygame.draw.circle(screen, (0,0,0), (marker_x, marker_y), 8)
            pygame.draw.circle(screen, (255,255,255), (marker_x, marker_y), 8, 2)
            
        # 色見本とRGB入力ボックスの描画 (既存ロジック)
        swatch_rect = pygame.Rect(0, 0, 150, 60); swatch_rect.center = (VIEW_WIDTH // 2, 50)
        pygame.draw.rect(screen, selected_color_rgb, swatch_rect, border_radius=8)
        labels = ["R:", "G:", "B:"]
        for i in range(3):
            label_surf = font_small.render(labels[i], True, (200, 200, 200))
            screen.blit(label_surf, (input_rects[i].left - 30, input_rects[i].centery - 10))
            border_color = (255, 255, 0) if active_input_index == i else (120, 120, 120)
            pygame.draw.rect(screen, (60, 60, 60), input_rects[i], border_radius=5)
            pygame.draw.rect(screen, border_color, input_rects[i], 2, 5)
            text_surf = font.render(input_texts[i], True, (240, 240, 240))
            text_rect = text_surf.get_rect(center=input_rects[i].center)
            screen.blit(text_surf, text_rect)

        pygame.display.flip()
        clock.tick(60)

    # --- 終了処理 ---
    print("\nStopping color picker.")
    serial_thread.send(np.zeros((NUM_ACTIVE_PIXELS, 3), dtype=np.uint8))
    time.sleep(0.1)
    serial_thread.close()
    serial_thread.join()
    pygame.quit()
    print("Picker closed.")

if __name__ == '__main__':
    interactive_color_picker()

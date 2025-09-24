# bird_color_test_alternating.py

import pygame
import numpy as np
import os
import yaml
from src.serial_handler import SerialWriterThread

# config_structure.pyから直接get_base_config関数をインポート
from config.config_structure import get_base_config

# --- 設定ファイルの読み込み ---
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.yaml")
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)
    
    SERIAL_PORT = settings['serial_port']
    BAUD_RATE = settings['baud_rate']
    NUM_LEDS = settings['num_leds']
    ENABLE_TEST_MODE = settings.get('enable_test_mode', False)
    NUM_TEST_STRIP_LEDS = settings.get('test_strip_led_count', 100)
    NUM_ACTIVE_LEDS = NUM_TEST_STRIP_LEDS if ENABLE_TEST_MODE else NUM_LEDS
    NUM_ACTIVE_PIXELS = NUM_ACTIVE_LEDS // 3
    MAGIC_BYTE = 0x7E
    VIEW_WIDTH = settings.get('view_width', 800)
    VIEW_HEIGHT = settings.get('view_height', 800)
    
except Exception as e:
    print(f"FATAL: Error loading settings from 'settings.yaml'. Error: {e}")
    exit()

def bird_color_tester_alternating():
    # --- Pygameの初期化 ---
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
    pygame.display.set_caption("Bird Color Alternating Test")
    font_large = pygame.font.Font(None, 60)
    font_medium = pygame.font.Font(None, 40)
    font_small = pygame.font.Font(None, 32)
    clock = pygame.time.Clock()

    # --- シリアル通信の準備 ---
    serial_thread = SerialWriterThread(SERIAL_PORT, BAUD_RATE, MAGIC_BYTE, NUM_ACTIVE_PIXELS)
    serial_thread.start()

    # --- テストする鳥のリストを生成 ---
    bird_params = get_base_config()
    tests_to_run = []
    for bird_id, params in bird_params.items():
        tests_to_run.append({
            "id": bird_id,
            "base_color": np.array(params['base_color']),
            "accent_color": np.array(params['accent_color'])
        })
    
    print("--- Starting Visual Bird Color Alternating Test ---")
    print(f"Testing Base/Accent alternating for {len(tests_to_run)} birds.")
    print("Press LEFT/RIGHT arrow keys to cycle. Press Q to quit.")

    current_bird_index = 0
    running = True

    while running:
        # --- イベント処理 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    current_bird_index = (current_bird_index + 1) % len(tests_to_run)
                if event.key == pygame.K_LEFT:
                    current_bird_index = (current_bird_index - 1 + len(tests_to_run)) % len(tests_to_run)
        
        # --- 現在の鳥の情報を取得 ---
        current_test = tests_to_run[current_bird_index]
        bird_id = current_test['id']
        color_base = current_test['base_color']
        color_accent = current_test['accent_color']

        # --- ▼▼▼ ここが核心部分 ▼▼▼ ---
        # Arduinoに送るLEDデータを「空間的に交互」に作成
        color_data = np.zeros((NUM_ACTIVE_PIXELS, 3), dtype=int)
        color_data[0::2] = color_base   # 偶数インデックスのピクセルをベースカラーに
        color_data[1::2] = color_accent # 奇数インデックスのピクセルをアクセントカラーに
        serial_thread.send(color_data)
        # --- ▲▲▲ ここが核心部分 ▲▲▲ ---
        
        # --- 画面描画 ---
        screen.fill((25, 28, 35))
        
        # 画面上部に鳥の名前を表示
        bird_name_surf = font_large.render(bird_id.capitalize(), True, (255, 255, 255))
        bird_name_rect = bird_name_surf.get_rect(center=(VIEW_WIDTH // 2, VIEW_HEIGHT * 0.15))
        screen.blit(bird_name_surf, bird_name_rect)

        # 画面中央に、物理LEDを模した交互のストリップを描画
        strip_y = VIEW_HEIGHT * 0.35
        strip_height = 60
        num_visual_leds = 60 # 画面に描画するLEDの数
        led_width = VIEW_WIDTH / num_visual_leds
        for i in range(num_visual_leds):
            color = color_base if i % 2 == 0 else color_accent
            rect = (i * led_width, strip_y, led_width - 1, strip_height) # -1で隙間を作る
            pygame.draw.rect(screen, tuple(color.astype(int)), rect)

        # 画面下部に、ベースとアクセントの凡例を表示
        # Base Color
        base_swatch_rect = pygame.Rect(VIEW_WIDTH * 0.1, VIEW_HEIGHT * 0.65, VIEW_WIDTH * 0.3, 80)
        pygame.draw.rect(screen, tuple(color_base.astype(int)), base_swatch_rect, border_radius=10)
        base_label_surf = font_medium.render("Base Color", True, (200, 200, 200))
        base_label_rect = base_label_surf.get_rect(center=base_swatch_rect.center, y=base_swatch_rect.y - 30)
        screen.blit(base_label_surf, base_label_rect)
        base_rgb_surf = font_small.render(f"RGB: {tuple(color_base.astype(int))}", True, (200, 200, 200))
        base_rgb_rect = base_rgb_surf.get_rect(center=base_swatch_rect.center, y=base_swatch_rect.y + 100)
        screen.blit(base_rgb_surf, base_rgb_rect)
        
        # Accent Color
        accent_swatch_rect = pygame.Rect(VIEW_WIDTH * 0.6, VIEW_HEIGHT * 0.65, VIEW_WIDTH * 0.3, 80)
        pygame.draw.rect(screen, tuple(color_accent.astype(int)), accent_swatch_rect, border_radius=10)
        accent_label_surf = font_medium.render("Accent Color", True, (200, 200, 200))
        accent_label_rect = accent_label_surf.get_rect(center=accent_swatch_rect.center, y=accent_swatch_rect.y - 30)
        screen.blit(accent_label_surf, accent_label_rect)
        accent_rgb_surf = font_small.render(f"RGB: {tuple(color_accent.astype(int))}", True, (200, 200, 200))
        accent_rgb_rect = accent_rgb_surf.get_rect(center=accent_swatch_rect.center, y=accent_swatch_rect.y + 100)
        screen.blit(accent_rgb_surf, accent_rgb_rect)

        pygame.display.flip()
        clock.tick(10)

    # --- 終了処理 ---
    print("\nStopping alternating color test.")
    serial_thread.send(np.zeros((NUM_ACTIVE_PIXELS, 3)))
    time.sleep(0.1)
    serial_thread.close()
    serial_thread.join()
    pygame.quit()
    print("Test finished.")

if __name__ == '__main__':
    bird_color_tester_alternating()
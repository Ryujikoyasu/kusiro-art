
import pygame
import numpy as np
from src.coordinates import CoordinateSystem

class Renderer:
    """
    Handles all rendering tasks for the simulation, including debug and artistic views.
    """
    def __init__(self, settings, pixel_model_positions, coord_system: CoordinateSystem, lidar_pose: dict = None):
        """
        Initializes the Renderer with necessary settings and pre-calculated positions.
        
        Args:
            settings (dict): The main settings dictionary.
            pixel_model_positions (np.array): (N, 2) array of pixel positions in model space.
            coord_system (CoordinateSystem): The coordinate system converter.
            lidar_pose (dict, optional): Dictionary containing LiDAR's pose {'x', 'y', 'theta_deg'}. Defaults to None.
        """
        # Settings
        self.view_width = settings.get('view_width', 800)
        self.view_height = settings.get('view_height', 800)
        self.min_brightness_falloff = settings.get('min_brightness_falloff', 0.3)
        self.debug_min_bird_size_px = 6.0

        # シミュレーター用の色設定を読み込む
        self.simulator_colors = settings.get('simulator_visuals', {})

        self.lidar_pose = lidar_pose # LiDARの姿勢情報を保存

        # --- LiDARアイコンの事前生成 (効率化のため) ---
        self.lidar_icon_surface = None
        if self.lidar_pose:
            self._create_lidar_icon()

        # --- 全体的な輝度設定 ---
        self.global_brightness = settings.get('global_brightness', 0.2)

        # Font for debug text
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 16)

        # Coordinate system
        self.coord_system = coord_system
        
        # Pixel data
        self.pixel_model_positions = pixel_model_positions
        self.pixel_view_positions = np.array([self.coord_system.model_to_view(p) for p in self.pixel_model_positions])
        self.num_pixels = len(pixel_model_positions)

        # Surfaces for drawing
        self.debug_surface = pygame.Surface((self.view_width, self.view_height))
        self.art_surface = pygame.Surface((self.view_width, self.view_height))
        
        # Pre-render static backgrounds
        self._create_static_backgrounds()

        # Calculated colors from the last frame, can be fetched for real-time output
        self.final_pixel_colors = np.zeros((self.num_pixels, 3), dtype=int)

    def _create_lidar_icon(self):
        """LiDARを表す三角形のアイコンを事前に描画しておく"""
        icon_size = 15 # ピクセル単位
        self.lidar_icon_surface = pygame.Surface((icon_size * 2, icon_size * 2), pygame.SRCALPHA)
        
        # 三角形の頂点
        points = [
            (icon_size, 0),                           # 頂点 (前)
            (icon_size * 1.8, icon_size * 2),         # 右下
            (icon_size * 0.2, icon_size * 2)          # 左下
        ]
        
        pygame.draw.polygon(self.lidar_icon_surface, (255, 0, 0, 200), points) # 赤い半透明の三角形
        pygame.draw.polygon(self.lidar_icon_surface, (255, 255, 255), points, 1) # 白い枠線

    def _create_static_backgrounds(self):
        """Creates the non-changing background elements for both views."""
        # Define the bounding rectangle for the ellipse in the view space
        view_rect = pygame.Rect(
            self.coord_system.view_center[0] - self.coord_system.actual_view_radius_x,
            self.coord_system.view_center[1] - self.coord_system.actual_view_radius_y,
            self.coord_system.actual_view_radius_x * 2,
            self.coord_system.actual_view_radius_y * 2
        )

        # Debug view background
        self.static_debug_bg = pygame.Surface((self.view_width, self.view_height))
        self.static_debug_bg.fill((25, 28, 35))
        pygame.draw.ellipse(self.static_debug_bg, (20, 40, 80), view_rect)
        for pos_px in self.pixel_view_positions:
            pygame.draw.circle(self.static_debug_bg, (50, 50, 50), pos_px, 2)

        # Artistic view background
        self.static_art_bg = pygame.Surface((self.view_width, self.view_height))
        self.static_art_bg.fill((5, 8, 15))
        pygame.draw.ellipse(self.static_art_bg, (20, 40, 80), view_rect)

    def calculate_pixel_colors(self, world):
        """
        Calculates the final color for each pixel based on the state of the world.
        This updates the internal `self.final_pixel_colors` attribute.
        """
        brightness_map = np.zeros(self.num_pixels, dtype=float)
        winner_map = np.full(self.num_pixels, -1, dtype=int)
        
        pixel_centers = [np.argmin(np.linalg.norm(self.pixel_model_positions - bird.position, axis=1)) for bird in world.birds]

        for i, bird in enumerate(world.birds):
            center_idx = pixel_centers[i]
            
            # 1. パターンと基本サイズを取得
            color_pattern, num_pixels_pattern = bird.get_current_light_pattern()

            # 2. 輝度を決定 (通常時はグローバル設定値、CHIRPING時は動的計算)
            brightness = self.global_brightness
            if bird.state == "CHIRPING":
                brightness = 0.0 # デフォルトは0
                active_pattern = bird.chirp_patterns.get(bird.active_pattern_key, [])
                if active_pattern:
                    # パターンから現在の輝度を補間して計算
                    # (バグ修正1: ループ変数を i -> pat_idx に変更)
                    # (バグ修正2: ループ範囲を len-1 -> len にし、最後のキーフレームまでチェック)
                    for pat_idx in range(len(active_pattern)):
                        # 最後のキーフレームに達したら、その輝度を使いループを抜ける
                        if pat_idx == len(active_pattern) - 1:
                            brightness = active_pattern[pat_idx][1]
                            break

                        start_time, start_bright = active_pattern[pat_idx]
                        end_time, end_bright = active_pattern[pat_idx+1]
                        
                        if start_time <= bird.chirp_playback_time < end_time:
                            time_delta = end_time - start_time
                            progress = (bird.chirp_playback_time - start_time) / time_delta if time_delta > 0 else 0
                            brightness = start_bright + (end_bright - start_bright) * progress
                            break
                
                # 輝度に基づいて描画サイズを動的に変更
                num_pixels_pattern = int(num_pixels_pattern * (1 + brightness * bird.params['size'] * 0.5))

            # 3. 光の広がりを計算
            spread = num_pixels_pattern // 2

            for j in range(-spread, spread + 1):
                pixel_idx = center_idx + j
                if 0 <= pixel_idx < self.num_pixels:
                    linear_falloff = (spread - abs(j)) / spread if spread > 0 else 1.0
                    falloff = self.min_brightness_falloff + (1.0 - self.min_brightness_falloff) * linear_falloff
                    final_brightness = brightness * falloff
                    if final_brightness > brightness_map[pixel_idx]:
                        brightness_map[pixel_idx] = final_brightness
                        winner_map[pixel_idx] = i

        # Reset colors and then fill them in
        self.final_pixel_colors.fill(0)
        for pixel_idx in range(self.num_pixels):
            bird_idx = winner_map[pixel_idx]
            if bird_idx != -1 and bird_idx < len(world.birds):
                bird = world.birds[bird_idx]
                pixel_offset = pixel_idx - pixel_centers[bird_idx]
                pattern, _ = bird.get_current_light_pattern()
                total_pixels = sum(p[1] for p in pattern)
                
                if total_pixels > 0:
                    color_type, start_pixel = 'b', -total_pixels // 2
                    for p_type, p_count in pattern:
                        if start_pixel <= pixel_offset < start_pixel + p_count:
                            color_type = p_type
                            break
                        start_pixel += p_count
                    
                    color = bird.accent_color if color_type == 'a' else bird.base_color
                    self.final_pixel_colors[pixel_idx] = np.clip(color * brightness_map[pixel_idx], 0, 255)

        # --- ▼ここから追加 ---
        # 描画処理で再利用するために、計算結果をインスタンス変数に保存
        self.brightness_map = brightness_map
        self.winner_map = winner_map
        # --- ▲ここまで追加 ---

    def get_final_colors(self):
        """Returns the latest calculated pixel colors."""
        return self.final_pixel_colors

    def _draw_lidar_pose(self, surface):
        """デバッグ画面にLiDARの姿勢を描画する"""
        if not self.lidar_pose:
            return

        # --- 1. LiDARのローカル座標で形状を定義 (前方がX軸プラス方向) ---
        #    - A: 後方左 (-0.1, -0.1)
        #    - B: 後方右 (-0.1, 0.1)
        #    - C: 前方先端 (0.15, 0)
        shape_local = np.array([
            [-0.1, -0.1],
            [-0.1,  0.1],
            [ 0.15, 0.0]
        ])

        # --- 2. 姿勢データを使って、形状をワールド座標に変換 ---
        pos = np.array(self.lidar_pose['position_xy'])
        theta_rad = np.deg2rad(self.lidar_pose['rotation_z_deg'])
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        rotation_matrix = np.array([[c, -s], [s, c]])

        shape_world = np.array([pos + rotation_matrix @ p for p in shape_local])

        # --- 3. ワールド座標をビュー（ピクセル）座標に変換 ---
        shape_view = np.array([self.coord_system.model_to_view(p) for p in shape_world])

        # --- 4. 描画 ---
        pygame.draw.polygon(surface, (255, 255, 0), shape_view) # 黄色い三角形で描画
        pygame.draw.polygon(surface, (0, 0, 0), shape_view, 2) # 黒い縁取り
        
        # デバッグ情報を追加：LiDARの位置を表示
        debug_text = f"LiDAR: ({pos[0]:.2f}, {pos[1]:.2f}), θ={self.lidar_pose['rotation_z_deg']:.1f}°"
        debug_surface = self.font.render(debug_text, True, (255, 255, 0))
        surface.blit(debug_surface, (10, 10))

    def render(self, screen, world):
        """
        Calculates all colors and draws the full scene to the provided screen.
        """
        # 1. Calculate the light/color values for this frame (This updates self.final_pixel_colors, self.brightness_map, etc.)
        self.calculate_pixel_colors(world)

        # 2. Draw the Debug View (Using Simulator Colors)
        self.debug_surface.blit(self.static_debug_bg, (0, 0))

        # --- LiDAR姿勢の描画 (三角形として) ---
        self._draw_lidar_pose(self.debug_surface)

        for bird in world.birds:
            # --- ▼ここから修正 ▼ ---
            # シミュレーター用の色を取得。なければ物理色をフォールバックとして使用。
            sim_colors = self.simulator_colors.get(bird.id, {})
            base_color = sim_colors.get('base_color', bird.base_color)
            accent_color = sim_colors.get('accent_color', bird.accent_color)
            
            pos_px = self.coord_system.model_to_view(bird.position)
            size_px = max(bird.params['size'] * 2.5, self.debug_min_bird_size_px)
            pygame.draw.circle(self.debug_surface, base_color, pos_px, size_px)
            pygame.draw.circle(self.debug_surface, accent_color, pos_px, size_px * 0.4)
            # --- ▲ここまで修正 ▲ ---

        for human in world.humans:
            pos_px = self.coord_system.model_to_view(human.position)
            pygame.draw.circle(self.debug_surface, (255, 255, 255), pos_px, 10)

            # --- 人間の詳細情報を描画 ---
            info_text = (
                f"Pos: ({human.position[0]:.2f}, {human.position[1]:.2f}) | "
                f"Size: {human.size:.2f} | "
                f"Vel: {np.linalg.norm(human.velocity):.2f} | "
                f"SizeΔ: {human.size_change:.2f}"
            )
            text_surface = self.font.render(info_text, True, (255, 255, 255))
            # テキストを円の少し下に表示
            text_rect = text_surface.get_rect(center=(pos_px[0], pos_px[1] + 20))
            self.debug_surface.blit(text_surface, text_rect)

        # 3. Draw the Artistic View (Translating physical brightness to simulator colors)
        self.art_surface.blit(self.static_art_bg, (0, 0))
        # --- ▼ここから全面修正 ▼ ---
        for i, pos_px in enumerate(self.pixel_view_positions):
            bird_idx = self.winner_map[i]
            brightness = self.brightness_map[i]

            # ピクセルが光っている場合のみ描画
            if bird_idx != -1 and bird_idx < len(world.birds) and brightness > 0.01:
                bird = world.birds[bird_idx]
                
                # シミュレーター用の色を取得
                sim_colors = self.simulator_colors.get(bird.id, {})
                sim_base_color = np.array(sim_colors.get('base_color', bird.base_color))
                sim_accent_color = np.array(sim_colors.get('accent_color', bird.accent_color))

                # 物理LED側で計算された「どの色が使われるべきか」のロジックを再利用
                pixel_offset = i - np.argmin(np.linalg.norm(self.pixel_model_positions - bird.position, axis=1))
                pattern, total_pixels = bird.get_current_light_pattern()
                total_pixels = sum(p[1] for p in pattern)

                color_type_to_use = 'b' # デフォルトはbase_color
                if total_pixels > 0:
                    start_pixel = -total_pixels // 2
                    for p_type, p_count in pattern:
                        if start_pixel <= pixel_offset < start_pixel + p_count:
                            color_type_to_use = p_type
                            break
                        start_pixel += p_count
                
                # 最終的なシミュレーター用の色を決定
                color_to_use = sim_accent_color if color_type_to_use == 'a' else sim_base_color
                final_sim_color = np.clip(color_to_use * brightness, 0, 255)

                pygame.draw.circle(self.art_surface, final_sim_color, pos_px, 4)
        # --- ▲ここまで全面修正 ▲ ---

        # 4. Blit both views to the main screen
        screen.blit(self.debug_surface, (0, 0))
        screen.blit(self.art_surface, (self.view_width, 0))
        
        # 5. Update the display
        pygame.display.flip()

import pygame
import numpy as np
import random
import os

class Human:
    """Represents the user in the simulation. An "actor" in the world."""
    def __init__(self, position, velocity, size, size_change):
        self.position = np.array(position)
        self.velocity = np.array(velocity) # 鳥のAIが参照する、平滑化された速度
        self.smooth_velocity = np.array(velocity) # 次フレーム計算用の平滑化速度
        self.size = size
        self.size_change = size_change # varianceから改名

class Bird:
    """
    Represents a single bird as an AI agent. An "actor" in the world.
    It is only responsible for its own behavior and intentions.
    """
    def __init__(self, bird_id, params, chirp_probability):
        self.id = bird_id
        self.params = params
        self.chirp_probability = chirp_probability

        # Visual and personality parameters
        self.base_color = np.array(self.params['base_color'])
        self.accent_color = np.array(self.params['accent_color'])
        self.base_pixel_count = self.params['base_pixel_count']
        self.color_pattern = self.params['color_pattern']
        self.chirp_color_pattern = self.params.get('chirp_color_pattern', self.color_pattern)
        self.speed = self.params['movement_speed'] / 60.0
        self.approach_speed = self.params['approach_speed'] / 60.0
        self.curiosity = self.params['curiosity']
        self.caution_distance = self.params['caution_distance']
        self.flee_distance = self.params['flee_distance']
        self.chirp_patterns = self.params.get('chirp_pattern', {})
        self.pixel_personal_space = self.params.get('pixel_personal_space', 3) # デフォルト値を設定

        # State initialization
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.target_position = self.position
        self.state = "IDLE"
        self.action_timer = random.randint(180, 400)
        self.current_brightness = 1.0 # Initialize current_brightness
        
        # Playback tracking
        self.chirp_playback_time = 0.0
        self.active_pattern_key = None 

        # Load sounds
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sounds = {}
        try:
            for key, path in self.params.get('sound_files', {}).items():
                abs_path = os.path.join(project_root, path)
                self.sounds[key] = pygame.mixer.Sound(abs_path)
        except Exception as e:
            print(f"ERROR loading sound for {self.id} at '{abs_path}': {e}")

    def get_current_light_pattern(self):
        """Returns the appropriate light pattern and base pixel count based on the current state."""
        if self.state == "CHIRPING":
            return self.chirp_color_pattern, self.base_pixel_count
        return self.color_pattern, self.base_pixel_count

    def update(self, humans, all_birds, my_index, all_pixel_centers):
        """1D/2D空間を考慮して鳥の状態を更新する"""

        # --- 0. 最もインタラクションすべき人間を見つける ---
        nearest_human = None
        min_dist_to_human = float('inf')
        if humans: # 人間が一人でも存在する場合
            distances = [np.linalg.norm(self.position - h.position) for h in humans]
            min_idx = np.argmin(distances)
            min_dist_to_human = distances[min_idx]
            nearest_human = humans[min_idx]

        # --- 1. LEDテープ上(1D)の縄張り意識 ---
        my_pixel_pos = all_pixel_centers[my_index]
        led_repulsion_vec = np.array([0.0, 0.0])

        for other_index, other in enumerate(all_birds):
            if self is other: continue

            other_pixel_pos = all_pixel_centers[other_index]
            pixel_distance = abs(my_pixel_pos - other_pixel_pos)

            # 縄張り内に他の鳥がいたら、2D空間で反発
            if pixel_distance < self.pixel_personal_space:
                vec_to_other_2d = other.position - self.position
                dist_to_other_2d = np.linalg.norm(vec_to_other_2d)

                if dist_to_other_2d > 1e-6:
                    overlap = self.pixel_personal_space - pixel_distance
                    repulsion_force = (vec_to_other_2d / dist_to_other_2d) * (overlap / self.pixel_personal_space)
                    led_repulsion_vec -= repulsion_force
        
        # 反発力を速度に穏やかに加える
        self.velocity += led_repulsion_vec * self.speed * 0.5

        # --- 2. 人間とのインタラクション(2D)とステートマシン ---
        if nearest_human: # 最も近い人間が存在する場合のみ、インタラクションを考慮
            # 鳴いている最中は、他の状態に遷移させない
            if self.state != 'CHIRPING':
                # 新しいロジック：速度や分散に基づく状態変化
                if np.linalg.norm(nearest_human.velocity) > 0.5: # 速度が速い人間には、より遠くから逃げる
                    self.state = "FLEEING"
                
                # 手を広げた（サイズが急に大きくなった）ら、特別な鳴き声を出す
                if nearest_human.size_change > 0.5: # THRESHOLD_SIZE_CHANGE
                    # TODO: ここで特別な鳴き声の再生や状態変化を実装
                    pass # 例: self.state = "SURPRISED_CHIRP"

                if self.state not in ["FLEEING"]: # CHIRPINGは上のifで除外済み
                    if min_dist_to_human < self.flee_distance: self.state = "FLEEING"
                    elif min_dist_to_human < self.caution_distance: self.state = "CAUTION"
                    # 人間の速度が非常に遅い（ほぼ静止）場合に、好奇心を示す
                    elif np.linalg.norm(nearest_human.velocity) < 0.05 and random.random() < self.curiosity: 
                        self.state = "CURIOUS"
            
            self.action_timer -= 1

            if self.state == "IDLE":
                self.velocity *= 0.8
                if self.action_timer <= 0:
                    self.state = "FORAGING" if random.random() < 0.7 else "EXPLORING"
                    self.action_timer = random.randint(120, 300) if self.state == "FORAGING" else random.randint(180, 400)
                    if self.state == "EXPLORING":
                        distance = random.uniform(1.5, 4.0)
                        angle = random.uniform(0, 2 * np.pi)
                        self.target_position = self.position + np.array([np.cos(angle), np.sin(angle)]) * distance
            elif self.state == "FORAGING":
                if random.random() < 0.1: self.velocity += (np.random.rand(2) - 0.5) * 0.02
                else: self.velocity *= 0.7
                if self.action_timer <= 0: self.state = "IDLE"; self.action_timer = random.randint(180, 400)
            elif self.state == "EXPLORING":
                direction_vec = self.target_position - self.position
                if np.linalg.norm(direction_vec) < 0.2: self.state = "IDLE"; self.action_timer = random.randint(180, 400)
                else: self.velocity += direction_vec / np.linalg.norm(direction_vec) * self.speed * 0.1
            elif self.state == "CURIOUS":
                direction_vec = nearest_human.position - self.position; dist = np.linalg.norm(direction_vec)
                if dist < self.caution_distance * 0.8: self.state = "IDLE"; self.action_timer = random.randint(180, 400)
                else: self.velocity += direction_vec / dist * self.approach_speed * 0.1
                # 人間が動き出したら、警戒状態に戻る
                if np.linalg.norm(nearest_human.velocity) > 0.1: self.state = "CAUTION"
            elif self.state == "FLEEING":
                self.velocity += (self.position - nearest_human.position) / min_dist_to_human * self.speed * 0.3
                if min_dist_to_human > self.flee_distance * 1.5: self.state = "CAUTION"
            elif self.state == "CAUTION":
                self.velocity *= 0.8
                if min_dist_to_human > self.caution_distance * 1.2: self.state = "IDLE"
            elif self.state == "CHIRPING":
                # 物理計算はsimulation.py側で完全にスキップされるので、ここでは何もしない
                self.chirp_playback_time += 1.0 / 60.0
                # 輝度計算はレンダラーに任せるので、ここでは時間経過のみを管理
                if self.action_timer <= 0:
                    self.state = "IDLE"
                    self.active_pattern_key = None

        else: # 人間が誰もいない場合
            if self.state in ["FLEEING", "CAUTION", "CURIOUS"]:
                self.state = "IDLE"

        # ランダムなタイミングで鳴き声を開始
        if self.state in ["IDLE", "FORAGING"] and self.action_timer > 0 and random.random() < self.chirp_probability:
            self.active_pattern_key = 'drumming' if self.id == 'kumagera' else 'default'
            if self.active_pattern_key in self.sounds:
                sound_to_play = self.sounds[self.active_pattern_key]
                self.state = "CHIRPING"
                # Set the timer to the actual length of the sound file.
                self.action_timer = int(sound_to_play.get_length() * 60) # Convert seconds to frames
                self.chirp_playback_time = 0.0
                sound_to_play.play()
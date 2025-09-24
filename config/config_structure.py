# This file is a backup of the base bird parameters.
# It is used by audio_sync_generator.py to prevent read errors from a broken config.
def get_base_config():
    return {
        # =========================================================================
        # === Final Color Palette v4 - Final Adjustments ==========================
        # =========================================================================
        
        "ooluri": { # OK
            "name_jp": "オオルリ", 
            "base_color": [48,46,90], "accent_color": [0, 5, 196], 
            "base_pixel_count": 4, "size": 1.0, "pixel_personal_space": 5,
            "color_pattern": [['b', 1], ['a', 2], ['b', 1]], "chirp_color_pattern": [['b', 1], ['a', 3], ['b', 1]],
            "movement_speed": 0.2, "approach_speed": 0.1, "curiosity": 0.5, 
            "caution_distance": 1.5, "flee_distance": 1.0, 
            "sound_files": {"default": "assets/sounds/oruri.mp3"}
        },
        
        "oohakucho": { # OK
            "name_jp": "オオハクチョウ", 
            "base_color": [255, 229, 53], "accent_color": [255, 200, 0], 
            "base_pixel_count": 6, "size": 4.0, "pixel_personal_space": 5,
            "color_pattern": [['b', 2], ['a', 2], ['b', 2]], "chirp_color_pattern": [['a', 2], ['b', 4], ['a', 2]],
            "movement_speed": 0.08, "approach_speed": 0.02, "curiosity": 0.1, 
            "caution_distance": 2.2, "flee_distance": 1.8, 
            "sound_files": {"default": "assets/sounds/ohakucho.mp3"}
        },
        
        "ojirowasi": { # OK
            "name_jp": "オジロワシ", 
            "base_color": [16, 156, 145], "accent_color": [221, 134, 7], 
            "base_pixel_count": 6, "size": 5.0, "pixel_personal_space": 5,
            "color_pattern": [['b', 2], ['a', 2], ['b', 2]], "chirp_color_pattern": [['a', 2], ['b', 4], ['a', 2]],
            "movement_speed": 0.05, "approach_speed": 0.0, "curiosity": 0.0, 
            "caution_distance": 3.0, "flee_distance": 1.2, 
            "sound_files": {"default": "assets/sounds/ojirowasi.mp3"}
        },
        
        "shimafukuro": { # NEW ACCENT: Deeper, more greenish
            "name_jp": "シマフクロウ", 
            # "base_color": [0, 0, 0], "accent_color": [0, 150, 80], 
            "base_color": [4,4,15], "accent_color": [57,255,62], 
            "base_pixel_count": 5, "size": 4.5, "pixel_personal_space": 5,
            "color_pattern": [['b', 2], ['a', 2], ['b', 2]], "chirp_color_pattern": [['a', 3], ['b', 3], ['a', 3]],
            "movement_speed": 0.1, "approach_speed": 0.0, "curiosity": 0.01, 
            "caution_distance": 3.5, "flee_distance": 2.8, 
            "sound_files": {"default": "assets/sounds/simafukuro.mp3"}
        },
        
        "kumagera": { # NEW ACCENT: User-specified red
            "name_jp": "クマゲラ",
            "base_color": [3, 8, 2], "accent_color": [157, 1, 0],
            "base_pixel_count": 3, "size": 1.5, "pixel_personal_space": 5,
            "color_pattern": [['a', 1], ['b', 2], ['a', 1]], "chirp_color_pattern": [['a', 1], ['b', 3], ['a', 1]],
            "movement_speed": 0.3, "approach_speed": 0.1, "curiosity": 0.2, 
            "caution_distance": 1.8, "flee_distance": 1.2, 
            "sound_files": {"call": "assets/sounds/kumagera.mp3", "drumming": "assets/sounds/kumagera_drum.mp3"}
        },
        
        "tancho": { # OK
            "name_jp": "タンチョウ", 
            "base_color": [255, 255, 255], "accent_color": [255, 0, 0], 
            "base_pixel_count": 5, "size": 3.5, "pixel_personal_space": 5,
            "color_pattern": [['b', 1], ['a', 2], ['b', 1]], "chirp_color_pattern": [['a', 2], ['b', 3], ['a', 2]],
            "movement_speed": 0.1, "approach_speed": 0.05, "curiosity": 0.3, 
            "caution_distance": 2.5, "flee_distance": 2.0, 
            "sound_files": {"default": "assets/sounds/tancho.mp3"}
        },
        
        "nogoma": { # NEW BASE & ACCENT: Earthy brown and vivid orange
            "name_jp": "ノゴマ", 
            "base_color": [21, 9, 0], "accent_color": [164, 37, 0], 
            "base_pixel_count": 3, "size": 0.8, "pixel_personal_space": 5,
            "color_pattern": [['b', 1], ['a', 2], ['b', 1]], "chirp_color_pattern": [['a', 1], ['b', 3], ['a', 1]],
            "movement_speed": 0.4, "approach_speed": 0.2, "curiosity": 0.8, 
            "caution_distance": 2.0, "flee_distance": 1.5, 
            "sound_files": {"default": "assets/sounds/nogoma.mp3"}
        },
        
        "benimashiko": { # OK
            "name_jp": "ベニマシコ", 
            "base_color": [25,9,5], "accent_color": [230, 0, 13], 
            "base_pixel_count": 3, "size": 0.9, "pixel_personal_space": 5,
            "color_pattern": [['a', 1], ['b', 2], ['a', 1]], "chirp_color_pattern": [['a', 1], ['b', 3], ['a', 1]],
            "movement_speed": 0.35, "approach_speed": 0.15, "curiosity": 0.7, 
            "caution_distance": 2.2, "flee_distance": 2.0, 
            "sound_files": {"default": "assets/sounds/benimasiko.mp3"}
        }
    }
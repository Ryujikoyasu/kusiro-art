"""
Insect-only base configuration.
Used by audio_sync_generator.py to generate config/insect_config.py (INSECT_PARAMS).
Contains 4 fields per species: colors(base,accent), chirp_pattern(auto), sound_files.
"""

def get_insect_base_config():
    return {
        "aomatsumushi": {
            "colors": {"base": [50, 200, 180], "accent": [120, 255, 230]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/アオマツムシ.mp3"},
        },
        "kutsuwa": {
            "colors": {"base": [255, 80, 20], "accent": [255, 140, 60]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/クツワムシ.mp3"},
        },
        "matsumushi": {
            "colors": {"base": [180, 255, 80], "accent": [220, 255, 140]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/マツムシ.mp3"},
        },
        "umaoi": {
            "colors": {"base": [50, 255, 100], "accent": [120, 255, 160]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/ウマオイ.mp3"},
        },
        "koorogi": {
            "colors": {"base": [160, 100, 255], "accent": [210, 170, 255]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/コオロギ.mp3"},
        },
        "kirigirisu": {
            "colors": {"base": [255, 200, 40], "accent": [255, 235, 120]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/キリギリス.mp3"},
        },
        "suzumushi": {
            "colors": {"base": [200, 220, 255], "accent": [230, 240, 255]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/スズムシ.mp3"},
        },
    }

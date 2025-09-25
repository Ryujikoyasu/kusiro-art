"""
Insect-only base configuration.
Used by audio_sync_generator.py to generate config/insect_config.py (INSECT_PARAMS).
Contains 4 fields per species: colors(base,accent), chirp_pattern(auto), sound_files.
"""

def get_insect_base_config():
    return {
        "aomatsumushi": {
            "colors": {"color": [12, 20, 77]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/アオマツムシ.mp3"},
        },
        "kutsuwa": {
            "colors": {"color": [66, 22, 3]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/クツワムシ.mp3"},
        },
        "matsumushi": {
            "colors": {"color": [47, 41, 5]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/マツムシ.mp3"},
        },
        "umaoi": {
            "colors": {"color": [47, 9, 74]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/ウマオイ.mp3"},
        },
        "koorogi": {
            "colors": {"color": [66, 20, 40]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/コオロギ.mp3"},
        },
        "kirigirisu": {
            "colors": {"color": [8, 31, 5]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/キリギリス.mp3"},
        },
        "suzumushi": {
            "colors": {"color": [0, 14, 9]},
            "chirp_pattern": {},
            "sound_files": {"default": "assets/data/sound/trimmed/スズムシ.mp3"},
        },
    }

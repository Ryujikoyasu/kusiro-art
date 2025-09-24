import pygame
import os
import glob
import numpy as np
import librosa

# --- Analysis function from audio_sync_generator.py ---
# This is the original algorithm we are testing.
def analyze_chirp(file_path):
    try:
        y, sr = librosa.load(file_path)
        # By adding a 'delta' parameter, we set a minimum threshold for onset strength.
        # This should be a more precise way to ignore noise without silencing actual chirps.
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames', hop_length=512, backtrack=True, energy=y**2, delta=0.1)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        
        if not onset_times.any():
            return [(0.0, 0.0), (librosa.get_duration(y=y, sr=sr), 0.0)]
            
        events = [(0.0, 0.0)]
        last_time = 0.0
        
        for t in onset_times:
            if t > last_time + 0.05:
                events.append((np.float64(last_time), 0.0))
                events.append((np.float64(t), 1.2))
                last_time = t + 0.075
        
        events.append((np.float64(last_time), 0.0))
        final_duration = librosa.get_duration(y=y, sr=sr)
        if final_duration > last_time:
            events.append((final_duration, 0.0))
            
        return events
    except Exception as e:
        print(f"  [ERROR] Could not process audio file {os.path.basename(file_path)}: {e}")
        return [(0, 0)]

# --- Pygame Test Application ---
def run_test():
    # --- Setup ---
    pygame.init()
    pygame.mixer.init()

    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (50, 50, 50)
    GREEN = (0, 255, 0)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Sound Sync Test Tool")
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()

    # --- Load Audio Files ---
    sound_dir = os.path.join(os.path.dirname(__file__), 'assets', 'sounds')
    sound_files = sorted(glob.glob(os.path.join(sound_dir, '*.mp3')))
    if not sound_files:
        print(f"Error: No .mp3 files found in {sound_dir}")
        return
    
    current_sound_index = 0
    chirp_events = []
    
    # --- Helper to load sound and analysis ---
    def load_sound(index):
        pygame.mixer.music.stop()
        file_path = sound_files[index]
        print(f"Analyzing {os.path.basename(file_path)}...")
        events = analyze_chirp(file_path)
        print("Analysis complete.")
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        return events, os.path.basename(file_path)

    # --- Initial Load ---
    chirp_events, current_filename = load_sound(current_sound_index)
    
    # --- Main Loop ---
    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_RIGHT:
                    current_sound_index = (current_sound_index + 1) % len(sound_files)
                    chirp_events, current_filename = load_sound(current_sound_index)
                if event.key == pygame.K_LEFT:
                    current_sound_index = (current_sound_index - 1 + len(sound_files)) % len(sound_files)
                    chirp_events, current_filename = load_sound(current_sound_index)

        # --- Sync Logic ---
        playback_time_sec = pygame.mixer.music.get_pos() / 1000.0
        
        current_intensity = 0.0
        # Find the current event state based on playback time
        for i in range(len(chirp_events) - 1):
            if chirp_events[i][0] <= playback_time_sec < chirp_events[i+1][0]:
                current_intensity = chirp_events[i][1]
                break
        
        # --- Drawing ---
        screen.fill(BLACK)

        # Draw file name
        text_surf = font.render(current_filename, True, WHITE)
        screen.blit(text_surf, (20, 20))

        # Draw progress bar
        progress = playback_time_sec / (librosa.get_duration(filename=sound_files[current_sound_index]))
        pygame.draw.rect(screen, GRAY, (20, 70, SCREEN_WIDTH - 40, 20))
        pygame.draw.rect(screen, GREEN, (20, 70, (SCREEN_WIDTH - 40) * progress, 20))

        # Draw light indicator
        indicator_color = WHITE if current_intensity > 0 else GRAY
        pygame.draw.circle(screen, indicator_color, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), 100)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    run_test()

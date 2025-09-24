
import pygame
import numpy as np
import os
import yaml
from config.config import BIRD_PARAMS
from src.objects import Bird
from src.input_source import MouseInputSource # MouseInputSourceをインポート
from src.simulation import World
from src.renderer import Renderer
from src.coordinates import CoordinateSystem

# --- Load all settings from settings.yaml ---
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.yaml")
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)
    
    LED_FILE_NAME = settings['led_layout_file']
    LED_FILE_PATH = os.path.join(PROJECT_ROOT, "assets", "data", LED_FILE_NAME)
    NUM_LEDS = settings['num_leds']
    BIRDS_TO_SIMULATE = settings.get('birds_to_simulate', [])
    AI_TUNING = settings.get('ai_tuning', {})
    CHIRP_PROBABILITY_PER_FRAME = AI_TUNING.get('chirp_probability_per_frame', 0.001)
    
    NUM_PIXELS = NUM_LEDS // 3
    MODEL_WIDTH = settings['model_width']
    MODEL_HEIGHT = settings['model_height']
    VIEW_WIDTH = settings.get('view_width', 800)
    SCREEN_WIDTH, SCREEN_HEIGHT = VIEW_WIDTH * 2, settings.get('view_height', 800)
    
    print("Loaded simulation settings from 'settings.yaml'")

except Exception as e:
    print(f"FATAL: Error loading settings from 'settings.yaml'. Please check the file. Error: {e}")
    exit()

def main():
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Left: Debug View | Right: Artistic View (Synced to Physical Pixels)")
    clock = pygame.time.Clock()

    # --- Coordinate System ---    
    coord_system = CoordinateSystem(view_size=(VIEW_WIDTH, SCREEN_HEIGHT), model_size=(MODEL_WIDTH, MODEL_HEIGHT))

    # --- Data and Object Initialization ---
    try:
        all_led_positions = np.loadtxt(LED_FILE_PATH, delimiter=',', skiprows=1)[:NUM_LEDS]
        pixel_model_positions = np.array([np.mean(all_led_positions[i*3:(i+1)*3], axis=0) for i in range(NUM_PIXELS)])
        
    except Exception as e:
        print(f"FATAL: Could not load LED data from '{LED_FILE_PATH}'. Error: {e}")
        return
    
    # Input Source
    input_source = MouseInputSource(coord_system.view_to_model)

    bird_objects = [Bird(bird_id, BIRD_PARAMS[bird_id], CHIRP_PROBABILITY_PER_FRAME) for bird_id in BIRDS_TO_SIMULATE if bird_id in BIRD_PARAMS]
    world = World(model_size=(MODEL_WIDTH, MODEL_HEIGHT), birds=bird_objects)
    
    # The renderer now handles all drawing surfaces and logic
    renderer = Renderer(settings, pixel_model_positions, coord_system)

    # --- Main Simulation Loop ---
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update simulation state
        detected_objects = input_source.get_detected_objects()
        world.update_humans(detected_objects)
        world.update(pixel_model_positions)

        # Render the current state to the screen
        renderer.render(screen, world)

        clock.tick(60)
        
    pygame.quit()

if __name__ == '__main__':
    main()

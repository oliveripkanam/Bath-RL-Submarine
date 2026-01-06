from src.cave_environment.environment import CaveEnvironment
from src.cave_environment.spritesheet import SpriteSheet
from src.entities.submarine import Submarine
import pygame
import pymunk
from pygame.locals import *
from src.sonar.sensors import Sonar

# Main setup
pygame.init()
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 800
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

spritesheet = SpriteSheet("src/cave_environment/tileset.png")
font = pygame.font.Font(None, 25)
big_font = pygame.font.Font(None, 80)
mid_font = pygame.font.Font(None, 40)

map_files = [
    "src/cave_environment/map1_basic.csv",
    "src/cave_environment/map2_jagged.csv",
    "src/cave_environment/map3_jagged_long_narrow.csv",
    "src/cave_environment/map4_zigzag.csv",
    "src/cave_environment/map5_one_battery.csv",
    "src/cave_environment/map6_three_battery.csv",
    "src/cave_environment/map7_obstacle_simple.csv",
    "src/cave_environment/map8_obstacle_hard.csv"
]
current_map_index = 0

def find_safe_start(env, width, height):
    wall_rects = [t.rect for t in env.environment_tiles]
    start_x = 0
    found_start = False
    
    for x in range(50, width - 50, 16):
        valid_ys = []
        for y in range(50, height - 50, 16):
            test_rect = pygame.Rect(x, y, 30, 30)
            if test_rect.collidelist(wall_rects) == -1:
                valid_ys.append(y)
        
        if len(valid_ys) > 3:
            start_x = x
            found_start = True
            break
            
    if not found_start: return 100, 100
        
    spawn_x = start_x + 64
    valid_ys_at_spawn = []
    for y in range(50, height - 50, 16):
        test_rect = pygame.Rect(spawn_x, y, 30, 30)
        if test_rect.collidelist(wall_rects) == -1:
            valid_ys_at_spawn.append(y)
            
    if valid_ys_at_spawn:
        avg_y = sum(valid_ys_at_spawn) // len(valid_ys_at_spawn)
        return spawn_x, avg_y
    return start_x + 20, sum(valid_ys) // len(valid_ys)

def load_level(map_index):
    new_space = pymunk.Space()
    actual_index = map_index
    
    try:
        if map_index >= len(map_files):
            print(f"Map index {map_index} not found, defaulting to 0")
            actual_index = 0
            
        env = CaveEnvironment(map_files[actual_index], spritesheet)
        
        # Physics walls
        for tile in env.environment_tiles:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (tile.rect.centerx, tile.rect.centery)
            shape = pymunk.Poly.create_box(body, (16, 16))
            shape.elasticity = 0.0
            shape.friction = 0.0
            shape.filter = pymunk.ShapeFilter(group=1)
            new_space.add(body, shape)

        for obstacle in env.obstacles:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (obstacle.rect.centerx, obstacle.rect.centery)
            shape = pymunk.Poly.create_box(body, (obstacle.rect.width, obstacle.rect.height))
            shape.filter = pymunk.ShapeFilter(group=1)
            new_space.add(body, shape)

        return new_space, env, actual_index
    except Exception as e:
        print(f"Error loading map: {e}")
        return None, None, 0

# Initial load
space, cave_env, current_map_index = load_level(0)
if not space: pygame.quit(); exit()

MAP_WIDTH = cave_env.environment_width
MAP_HEIGHT = cave_env.environment_height
canvas = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))

start_x, start_y = find_safe_start(cave_env, MAP_WIDTH, MAP_HEIGHT)
submarine = Submarine(start_x, start_y)
if current_map_index in [4, 5]: submarine.battery = 300
else: submarine.battery = 600

sonar_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
sonar_body.position = (submarine.rect.centerx, submarine.rect.centery)
space.add(sonar_body)
my_sonar = Sonar(space, sonar_body, num_rays=16, max_range=200, agent_size=30)

running = True
game_active = True
pitch_black_mode = False
clock = pygame.time.Clock()

while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == QUIT: running = False

        if event.type == KEYDOWN:
            if event.key in [K_LSHIFT, K_RSHIFT]:
                 pitch_black_mode = not pitch_black_mode

            if not game_active:
                if event.key == K_SPACE:
                    # Restart
                    space, cave_env, _ = load_level(current_map_index)
                    MAP_WIDTH = cave_env.environment_width
                    MAP_HEIGHT = cave_env.environment_height
                    canvas = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))
                    sx, sy = find_safe_start(cave_env, MAP_WIDTH, MAP_HEIGHT)
                    submarine = Submarine(sx, sy)
                    if current_map_index in [4, 5]: submarine.battery = 300
                    else: submarine.battery = 600
                    
                    sonar_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                    sonar_body.position = (submarine.rect.centerx, submarine.rect.centery)
                    space.add(sonar_body)
                    my_sonar.space = space
                    my_sonar.body = sonar_body
                    game_active = True

            # Movement
            if game_active and submarine.battery > 0:
                if event.key == K_UP: submarine.move_up()
                elif event.key == K_DOWN: submarine.move_down()
                elif event.key == K_LEFT: submarine.move_left()
                elif event.key == K_RIGHT: submarine.move_right()
            
            # Map switching
            new_space = None
            if event.key == K_1: new_space, new_env, new_idx = load_level(0)
            elif event.key == K_2: new_space, new_env, new_idx = load_level(1)
            elif event.key == K_3: new_space, new_env, new_idx = load_level(2)
            elif event.key == K_4: new_space, new_env, new_idx = load_level(3)
            elif event.key == K_5: new_space, new_env, new_idx = load_level(4)
            elif event.key == K_6: new_space, new_env, new_idx = load_level(5)
            elif event.key == K_7: new_space, new_env, new_idx = load_level(6)
            elif event.key == K_8: new_space, new_env, new_idx = load_level(7)
            
            if new_space:
                space = new_space
                cave_env = new_env
                current_map_index = new_idx
                
                MAP_WIDTH = cave_env.environment_width
                MAP_HEIGHT = cave_env.environment_height
                canvas = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))
                
                my_sonar.space = space
                sonar_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                sx, sy = find_safe_start(cave_env, MAP_WIDTH, MAP_HEIGHT)
                submarine.true_x, submarine.true_y = sx, sy
                submarine.rect.topleft = (int(sx), int(sy))
                sonar_body.position = (submarine.rect.centerx, submarine.rect.centery)
                space.add(sonar_body)
                my_sonar.body = sonar_body
                
                if current_map_index in [4, 5]: submarine.battery = 300
                else: submarine.battery = 600
                game_active = True

    if game_active:
        prev_x, prev_y = submarine.true_x, submarine.true_y
        submarine.update()
        sonar_body.position = (submarine.rect.centerx, submarine.rect.centery)

        # Collisions
        hits = pygame.sprite.spritecollide(submarine, cave_env.batteries, True)
        for hit in hits: submarine.battery += 300

        obs_hits = pygame.sprite.spritecollide(submarine, cave_env.obstacles, False)
        for hit in obs_hits: submarine.battery -= 50

        sensor_data = my_sonar.get_observation()
        
        hit_wall = False
        for reading in sensor_data:
            if reading == 0:
                hit_wall = True
                break
        if hit_wall:
            submarine.battery -= 10
            submarine.true_x, submarine.true_y = prev_x, prev_y
            submarine.rect.x, submarine.rect.y = int(prev_x), int(prev_y)
            submarine.vel_x *= -0.5
            submarine.vel_y *= -0.5

        if submarine.battery <= 0:
            submarine.battery = 0
            game_active = False
            
        if submarine.rect.right >= cave_env.environment_width - 10:
            game_active = False

    # Drawing
    if pitch_black_mode:
        canvas.fill((0, 0, 0)) # Pitch black
    else:
        canvas.fill((0, 128, 255))
        cave_env.draw(canvas)
        pygame.draw.line(canvas, (255, 0, 0), (start_x, 0), (start_x, MAP_HEIGHT), 2)
        pygame.draw.line(canvas, (0, 255, 0), (MAP_WIDTH - 5, 0), (MAP_WIDTH - 5, MAP_HEIGHT), 5)
        
    submarine.draw(canvas)
    my_sonar.draw(canvas, font)
    
    scale = min(WINDOW_WIDTH / MAP_WIDTH, WINDOW_HEIGHT / MAP_HEIGHT)
    new_size = (int(MAP_WIDTH * scale), int(MAP_HEIGHT * scale))
    scaled_surface = pygame.transform.smoothscale(canvas, new_size)
    
    dest_x = (WINDOW_WIDTH - new_size[0]) // 2
    dest_y = (WINDOW_HEIGHT - new_size[1]) // 2
    
    screen.fill((0, 0, 0))
    screen.blit(scaled_surface, (dest_x, dest_y))

    battery_text = font.render(f'Battery: {submarine.battery} | Map: {map_files[current_map_index]}', True, (255, 255, 255))
    controls_text = font.render('Arrows: Move | 1-8: Maps | SHIFT: Pitch Black Mode', True, (255, 255, 0))
    
    screen.blit(battery_text, (10, 10))
    screen.blit(controls_text, (10, 30))
    
    if not game_active:
        if submarine.battery <= 0:
            text_surf = big_font.render("GAME OVER", True, (255, 0, 0))
        else:
            text_surf = big_font.render("REACHED GOAL!", True, (0, 255, 0))
        text_rect = text_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 40))
        screen.blit(text_surf, text_rect)
        
        restart_surf = mid_font.render("Press SPACE to Restart", True, (255, 255, 255))
        restart_rect = restart_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
        screen.blit(restart_surf, restart_rect)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
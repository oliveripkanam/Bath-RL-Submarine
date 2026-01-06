import pygame
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pymunk
import numpy as np
import glob
import random
import time
from src.cave_environment.environment import CaveEnvironment
from src.cave_environment.spritesheet import SpriteSheet
from src.entities.submarine import Submarine
from src.sonar.sensors import Sonar
from src.ai.agent import ACERAgent

# Configuration
WATCH_MODE = False
LOAD_MODEL = False
NUM_EPISODES = 50000
MAX_STEPS = 4000
BATCH_SIZE = 128
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99995
TARGET_UPDATE = 1000
SAVE_INTERVAL = 50

# Format to HH:MM:SS
def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

# Smart epsilon decay
def get_epsilon(episode):
    if episode < 5000:
        return 1.0
    elif episode < 15000:
        return max(0.1, 1.0 * (EPSILON_DECAY ** (episode - 5000)))
    else:
        return max(EPSILON_END, 0.1 * (EPSILON_DECAY ** (episode - 15000)))

# Map configuration
MAP_FILES = [
    "src/cave_environment/map1_basic.csv",
    "src/cave_environment/map2_jagged.csv",
    "src/cave_environment/map3_jagged_long_narrow.csv",
    "src/cave_environment/map4_zigzag.csv",
    "src/cave_environment/map5_one_battery.csv",
    "src/cave_environment/map6_three_battery.csv",
    "src/cave_environment/map7_obstacle_simple.csv",
    "src/cave_environment/map8_obstacle_hard.csv",
]

# Setting display, clock, and fonts
pygame.init()
screen = pygame.display.set_mode((1200, 800))
pygame.display.set_caption("Double DQN Training")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 25)
hit_font = pygame.font.Font(None, 40)

def load_level(map_index, spritesheet):
    try:
        filename = MAP_FILES[map_index]
        env = CaveEnvironment(filename, spritesheet)
        new_space = pymunk.Space()
        new_space.gravity = (0, 0)
        
        for tile in env.environment_tiles:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (tile.rect.centerx, tile.rect.centery)
            shape = pymunk.Poly.create_box(body, (tile.rect.width, tile.rect.height))
            shape.filter = pymunk.ShapeFilter(group=1)
            new_space.add(body, shape)

        for obstacle in env.obstacles:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (obstacle.rect.centerx, obstacle.rect.centery)
            shape = pymunk.Poly.create_box(body, (obstacle.rect.width, obstacle.rect.height))
            shape.filter = pymunk.ShapeFilter(group=1)
            new_space.add(body, shape)
            
        return new_space, env
    except Exception as e:
        print(f"Error loading map {map_index}: {e}")
        return None, None

def get_full_state(sonar_data, submarine, map_idx, env_width, batteries):
    # Normalize inputs to 0-1 range where possible
    normalized_battery = submarine.battery / 500.0
    norm_vx = (submarine.vel_x + 10.0) / 20.0
    norm_vy = (submarine.vel_y + 10.0) / 20.0
    
    map_encoding = [0.0] * 20
    if map_idx < 20:
        map_encoding[map_idx] = 1.0
    
    map_encoding[-1] = submarine.true_x / env_width

    closest_dist = float('inf')
    bat_dx = 0.0
    bat_dy = 0.0
    
    if batteries and len(batteries) > 0:
        sub_x, sub_y = submarine.rect.centerx, submarine.rect.centery
        for bat in batteries:
            dx = bat.rect.centerx - sub_x
            dy = bat.rect.centery - sub_y
            dist = dx*dx + dy*dy
            if dist < closest_dist:
                closest_dist = dist
                bat_dx = dx
                bat_dy = dy
        
        map_encoding[17] = max(-1.0, min(1.0, bat_dx / 1000.0))
        map_encoding[18] = max(-1.0, min(1.0, bat_dy / 1000.0))
    
    return np.concatenate([
        sonar_data,
        [normalized_battery],
        [norm_vx, norm_vy],
        map_encoding
    ])

def train():
    global WATCH_MODE
    
    # Start timer
    training_start_time = time.time()
    
    if not LOAD_MODEL:
        files = glob.glob("models/ddqn_*.pth")
        for f in files:
            try: os.remove(f)
            except: pass
        print("Cleared previous DDQN models.")

    spritesheet = SpriteSheet("src/cave_environment/tileset.png")
    print("Pre-loading maps...")
    preloaded_maps = []
    for i in range(len(MAP_FILES)):
        s, e = load_level(i, spritesheet)
        preloaded_maps.append((s, e))
    print("Maps loaded.")
    
    agent = ACERAgent(input_shape=39, num_actions=5)
    
    total_steps = 0
    print(f"Starting ACER Training on Device: {agent.device}")
    print("Press TAB to toggle Fast/Watch Mode. Press ESC to quit.")

    success_history = []
    loss_history = []
    
    map_stats = {i: {'goals': 0, 'attempts': 0, 'total_reward': 0} for i in range(len(MAP_FILES))}
    battery_stats = {i: {'picked_up': 0, 'picked_success': 0, 'picked_fail': 0, 'ignored_fail': 0, 'ignored_success': 0} for i in range(len(MAP_FILES))}
    obstacle_stats = {i: {'avoided_won': 0, 'avoided_died': 0, 'hit_died': 0, 'hit_won': 0} for i in range(len(MAP_FILES))}
    clean_stats = {i: {'clean_win': 0, 'dirty_win': 0, 'clean_fail': 0, 'dirty_fail': 0} for i in range(len(MAP_FILES))}

    start_episode = 0

    if LOAD_MODEL:
        # Try to find the latest checkpoint or the final model
        model_path = "models/acer_submarine_final.pth"
        if not os.path.exists(model_path):
            list_of_files = glob.glob('models/ddqn_submarine_ep*.pth')
            if list_of_files:
                model_path = max(list_of_files, key=os.path.getctime)
                try: start_episode = int(model_path.split("ep")[-1].split(".")[0])
                except: pass
        try:
            agent.load(model_path)
            print(f"Loaded: {model_path}")
        except:
            print("Starting fresh.")
            epsilon = 1.0 
    
    for episode in range(start_episode, NUM_EPISODES):
        epsilon = get_epsilon(episode)

        rand_val = random.random()
        if rand_val < 0.25: map_idx = 7
        elif rand_val < 0.65: map_idx = 4
        elif rand_val < 0.80: map_idx = 2
        elif rand_val < 0.90: map_idx = 5
        elif rand_val < 0.95: map_idx = 3
        else: map_idx = random.choice([0, 1, 6])
            
        map_stats[map_idx]['attempts'] += 1
        space, cave_env = preloaded_maps[map_idx]
        
        cave_env.batteries.empty()
        cave_env.obstacles.empty()
        
        with open(MAP_FILES[map_idx], 'r') as f:
            for y, line in enumerate(f.readlines()):
                row = line.strip().split(',')
                for x, tile in enumerate(line.strip().split(',')):
                    if tile == '20': 
                        from src.entities.items import Battery
                        cave_env.batteries.add(Battery(x * 16, y * 16))
                    elif tile == '21': 
                        from src.entities.items import Obstacle
                        cave_env.obstacles.add(Obstacle(x * 16, y * 16))

        for body in list(space.bodies):
            if body.body_type != pymunk.Body.STATIC:
                space.remove(body)
                for s in body.shapes: space.remove(s)
        if not space: continue

        start_x, start_y = 100, 300
        found_start = False
        wall_rects = [t.rect for t in cave_env.environment_tiles]
        
        for x in range(50, cave_env.environment_width - 50, 20):
             valid_ys = []
             for y in range(50, cave_env.environment_height - 50, 10):
                 if pygame.Rect(x - 30, y - 30, 60, 60).collidelist(wall_rects) == -1:
                     valid_ys.append(y)
             if len(valid_ys) > 0:
                 start_x, start_y = x, sum(valid_ys) // len(valid_ys)
                 found_start = True
                 break
        
        if not found_start:
            # Try searching backwards from the left wall
            for x in range(50, 100, -20):
                 valid_ys = []
                 for y in range(50, cave_env.environment_height - 50, 10):
                     if pygame.Rect(x - 30, y - 30, 60, 60).collidelist(wall_rects) == -1:
                         valid_ys.append(y)
                 if len(valid_ys) > 0:
                     start_x, start_y = x, sum(valid_ys) // len(valid_ys)
                     found_start = True
                     break
            
        submarine = Submarine(start_x, start_y)
        if map_idx in [4, 5]: submarine.battery = 300
        else: submarine.battery = 600
            
        current_run_picked_battery = False
        hit_obstacle_this_run = False
        hit_wall_this_run = False
        
        sonar_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        sonar_body.position = (submarine.rect.centerx, submarine.rect.centery)
        sonar = Sonar(space, sonar_body)
        
        state = get_full_state(sonar.get_observation(), submarine, map_idx, cave_env.environment_width, cave_env.batteries)
        total_reward = 0
        done = False
        stagnation_start_x = submarine.true_x
        stagnation_timer = 0
        action = 4
        mu_prob = 1.0 / 5.0 # Default uniform prob

        for step in range(MAX_STEPS):
            # Allow toggling watch mode or quitting
            if WATCH_MODE:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB: WATCH_MODE = not WATCH_MODE
            elif step % 10 == 0:
                 for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB: WATCH_MODE = not WATCH_MODE

            if step % 4 == 0:
                action, mu_prob = agent.select_action(state, epsilon)
            
            prev_x, prev_y = submarine.true_x, submarine.true_y
            reward = 0.0

            if action == 0: submarine.move_up()
            elif action == 1: submarine.move_down()
            elif action == 2: submarine.move_left()
            elif action == 3: submarine.move_right()
            elif action == 4: reward += 0.05
            
            submarine.update()
            sonar_body.position = (submarine.rect.centerx, submarine.rect.centery)
            
            dist_x = submarine.true_x - prev_x
            reward += max(-1.0, min(2.0, dist_x * 0.5))

            hits = pygame.sprite.spritecollide(submarine, cave_env.batteries, True)
            for hit in hits:
                submarine.battery += 300
                reward += 10.0
                current_run_picked_battery = True

            obs_hits = pygame.sprite.spritecollide(submarine, cave_env.obstacles, False)
            for hit in obs_hits:
                submarine.battery -= 50
                reward -= 5.0
                hit_obstacle_this_run = True
            current_observation = sonar.get_observation()
            next_state = get_full_state(current_observation, submarine, map_idx, cave_env.environment_width, cave_env.batteries)

            # Store experience in replay memory for later training
            agent.memory.push(state, action, reward, next_state, done, mu_prob)
            total_reward += reward

            display_hit_msg = False
            display_hit_msg = False
            hit_wall = False
            for reading in current_observation:
                if reading == 0: hit_wall = True; break
            
            if hit_wall:
                hit_wall_this_run = True
                reward -= 50.0
                submarine.battery -= 10
                display_hit_msg = True
                submarine.true_x, submarine.true_y = prev_x, prev_y
                submarine.rect.x, submarine.rect.y = int(prev_x), int(prev_y)
                submarine.vel_x *= -0.5
                submarine.vel_y *= -0.5
                
            if submarine.rect.right >= cave_env.environment_width - 10:
                reward += 100.0
                if len(cave_env.batteries) == 0: reward += 50.0
                reward += submarine.battery * 1.0 
                done = True
                success_history.append(1)
                map_stats[map_idx]['goals'] += 1

            if submarine.battery <= 0:
                reward -= 10.0
                done = True
                success_history.append(0)

            stagnation_timer += 1
            if stagnation_timer >= 300:
                # If agent hasn't moved 100 pixels in 300 steps, kill it to prevent loops
                if abs(submarine.true_x - stagnation_start_x) < 100:
                    reward -= 5.0; done = True; success_history.append(0)
                stagnation_timer = 0
                stagnation_start_x = submarine.true_x

            if step % 4 == 0:
                loss = agent.train_step(BATCH_SIZE)
                if loss is not None: loss_history.append(loss)
            
            total_steps += 1
            if total_steps % TARGET_UPDATE == 0:
                agent.update_target_network()

            state = next_state

            if WATCH_MODE:
                canvas = pygame.Surface((cave_env.environment_width, cave_env.environment_height))
                canvas.fill((0, 128, 255))
                cave_env.draw(canvas)
                submarine.draw(canvas)
                sonar.draw(canvas, font)
                scale = min(1200 / cave_env.environment_width, 800 / cave_env.environment_height)
                new_size = (int(cave_env.environment_width * scale), int(cave_env.environment_height * scale))
                scaled = pygame.transform.smoothscale(canvas, new_size)
                screen.fill((0, 0, 0))
                screen.blit(scaled, ((1200 - new_size[0]) // 2, (800 - new_size[1]) // 2))
                
                info = f"Ep: {episode} | Reward: {total_reward:.1f} | Eps: {epsilon:.2f}"
                screen.blit(font.render(info, True, (255, 255, 255)), (10, 10))
                if display_hit_msg:
                    screen.blit(hit_font.render("HIT WALL!", True, (255,0,0)), (1200-200, 50))
                pygame.display.flip()
                clock.tick(60)
            else:
                if step % 100 == 0:
                    screen.fill((0, 0, 0))
                    # Runtime display
                    elapsed = time.time() - training_start_time
                    time_str = format_time(elapsed)
                    msg = font.render(f"FAST MODE (Ep {episode}) | Time: {time_str}", True, (0, 255, 0))
                    screen.blit(msg, (1200//2 - 200, 400))
                    pygame.display.flip()

            if done: break
        
        map_stats[map_idx]['total_reward'] += total_reward
        
        is_success = success_history[-1] == 1
        if current_run_picked_battery:
            battery_stats[map_idx]['picked_up'] += 1
            if is_success: battery_stats[map_idx]['picked_success'] += 1
            else: battery_stats[map_idx]['picked_fail'] += 1
        else:
            if is_success: battery_stats[map_idx]['ignored_success'] += 1
            else: battery_stats[map_idx]['ignored_fail'] += 1
        
        if hit_obstacle_this_run:
            if is_success: obstacle_stats[map_idx]['hit_won'] += 1
            else: obstacle_stats[map_idx]['hit_died'] += 1
        else:
            if is_success: obstacle_stats[map_idx]['avoided_won'] += 1
            else: obstacle_stats[map_idx]['avoided_died'] += 1

        if not hit_wall_this_run and not hit_obstacle_this_run:
            if is_success: clean_stats[map_idx]['clean_win'] += 1
            else: clean_stats[map_idx]['clean_fail'] += 1
        else:
            # "Dirty" means the run had at least one collision, even if it reached the goal
            if is_success: clean_stats[map_idx]['dirty_win'] += 1
            else: clean_stats[map_idx]['dirty_fail'] += 1

        if episode % SAVE_INTERVAL == 0:
            agent.save(f"models/ddqn_submarine_ep{episode}.pth")

    agent.save("models/acer_submarine_final.pth")
    
    # End timer
    training_end_time = time.time()
    total_time = training_end_time - training_start_time
    
    # Save loss
    np.save("acer_training_loss.npy", np.array(loss_history))
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE - FINAL STATISTICS")
    print(f"Total Training Time: {format_time(total_time)}")
    
    # Record time to file
    with open("acer_training_time.txt", "w") as f:
        f.write(f"Total Training Time: {format_time(total_time)}")
        
    print("="*50)
    print(f"{'Map File':<40} | {'Goals':<5} | {'Attempts':<8} | {'Success Rate':<12} | {'Avg Reward':<10}")
    print("-" * 85)
    for i, filename in enumerate(MAP_FILES):
        stats = map_stats[i]
        attempts = stats['attempts']
        goals = stats['goals']
        avg_reward = stats['total_reward'] / attempts if attempts > 0 else 0
        success_rate = (goals / attempts * 100) if attempts > 0 else 0
        print(f"{filename.split('/')[-1]:<40} | {goals:<5} | {attempts:<8} | {success_rate:<11.1f}% | {avg_reward:<10.1f}")
    
    print("\n" + "="*50)
    print("BATTERY STATS")
    print("="*50)
    print(f"{'Map File':<25} | {'Picked(Up)':<11} | {'Picked(Win)':<11} | {'Picked(Die)':<11} | {'Ignored(Die)':<12} | {'Ignored(Win)':<12}")
    print("-" * 90)
    for i, filename in enumerate(MAP_FILES):
        if map_stats[i]['attempts'] > 0 and (i in [4, 5]): # Maps 5 and 6 have batteries
            bs = battery_stats[i]
            display_name = filename.split('/')[-1]
            print(f"{display_name:<25} | {bs['picked_up']:<11} | {bs['picked_success']:<11} | {bs['picked_fail']:<11} | {bs['ignored_fail']:<12} | {bs['ignored_success']:<12}")

    print("\n" + "="*50)
    print("OBSTACLE STATS")
    print("="*50)
    print(f"{'Map File':<25} | {'Avoid(Win)':<11} | {'Avoid(Die)':<11} | {'Hit(Die)':<11} | {'Hit(Win)':<11}")
    print("-" * 90)
    for i, filename in enumerate(MAP_FILES):
        if map_stats[i]['attempts'] > 0 and (i in [6, 7]): # Maps 7 and 8 have obstacles
            os_stats = obstacle_stats[i]
            display_name = filename.split('/')[-1]
            print(f"{display_name:<25} | {os_stats['avoided_won']:<11} | {os_stats['avoided_died']:<11} | {os_stats['hit_died']:<11} | {os_stats['hit_won']:<11}")
    
    print("\n" + "="*50)
    print("CLEAN RUN STATS (No Wall/Obstacle Hits)")
    print("="*50)
    print(f"{'Map File':<25} | {'Clean Win':<11} | {'Dirty Win':<11} | {'Clean Fail':<11} | {'Dirty Fail':<11}")
    print("-" * 90)
    for i, filename in enumerate(MAP_FILES):
        if map_stats[i]['attempts'] > 0:
            cs = clean_stats[i]
            display_name = filename.split('/')[-1]
            print(f"{display_name:<25} | {cs['clean_win']:<11} | {cs['dirty_win']:<11} | {cs['clean_fail']:<11} | {cs['dirty_fail']:<11}")

    avg_first_100 = sum(loss_history[:100]) / len(loss_history[:100]) if len(loss_history) >= 100 else 0
    avg_last_100 = sum(loss_history[-100:]) / len(loss_history[-100:]) if len(loss_history) >= 100 else 0
    print("\n" + "="*50)
    print("NETWORK HEALTH")
    print(f"Avg Loss (First 100): {avg_first_100:.3f}")
    print(f"Avg Loss (Last 100):  {avg_last_100:.3f}")
    print("="*50)

    pygame.quit()

if __name__ == "__main__":
    train()

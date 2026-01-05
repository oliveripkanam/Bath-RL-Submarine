import pygame
import pymunk
import numpy as np
import os
import glob
import random
import csv
from src.cave_environment.environment import CaveEnvironment
from src.cave_environment.spritesheet import SpriteSheet
from src.entities.submarine import Submarine
from src.entities.items import Battery, Obstacle
from src.sonar.sensors import Sonar
from src.ai.agent import PPOAgent

import time

# Configuration
WATCH_MODE = False
LOAD_MODEL = False
NUM_EPISODES = 50000
MAX_STEPS = 3000
UPDATE_INTERVAL = 8192
BATCH_SIZE = 128
SAVE_INTERVAL = 50

# Load maps
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

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((1200, 800))
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
    normalized_battery = submarine.battery / 500.0
    norm_vx = (submarine.vel_x + 10.0) / 20.0
    norm_vy = (submarine.vel_y + 10.0) / 20.0
    
    map_encoding = [0.0] * 20
    if map_idx < 20:
        map_encoding[map_idx] = 1.0
    
    map_encoding[-1] = 0.0 

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

def calculate_dynamic_weights(map_stats, num_maps):
    weights = []
    for i in range(num_maps):
        if i not in map_stats or map_stats[i]['attempts'] < 5:
            weights.append(10.0)
        else:
            stats = map_stats[i]
            sr = stats['goals'] / stats['attempts'] if stats['attempts'] > 0 else 0.0
            weight = max(0.1, (1.0 - sr)**2) * 10.0
            weights.append(weight)
    return weights

def train():
    global WATCH_MODE
    
    training_start_time = time.time()
    
    if not LOAD_MODEL:
        files = glob.glob("models/*.pth")
        for f in files: 
            try: os.remove(f)
            except: pass

    spritesheet = SpriteSheet("src/cave_environment/tileset.png")
    print("Pre-loading maps...")
    preloaded_maps = []
    for i in range(len(MAP_FILES)):
        s, e = load_level(i, spritesheet)
        preloaded_maps.append((s, e))
    print("Maps loaded.")
    
    agent = PPOAgent(input_shape=39, num_actions=5, batch_size=BATCH_SIZE)
    
    total_steps = 0
    loss_history = []
    
    map_histories = {i: [] for i in range(len(MAP_FILES))}
    map_stats = {i: {'goals': 0, 'attempts': 0, 'total_reward': 0} for i in range(len(MAP_FILES))}
    curriculum_state = {i: 2500 for i in range(len(MAP_FILES))} 
    
    # Stats tracking
    battery_stats = {i: {'picked_up': 0, 'picked_success': 0, 'picked_fail': 0, 'ignored_fail': 0, 'ignored_success': 0} for i in range(len(MAP_FILES))}
    obstacle_stats = {i: {'avoided_won': 0, 'avoided_died': 0, 'hit_died': 0, 'hit_won': 0} for i in range(len(MAP_FILES))}
    clean_run_stats = {i: {'clean_win': 0, 'dirty_win': 0, 'clean_fail': 0, 'dirty_fail': 0} for i in range(len(MAP_FILES))}

    start_episode = 0

    if LOAD_MODEL:
        model_path = "models/ppo_submarine_final.pth"
        if not os.path.exists(model_path):
            list_of_files = glob.glob('models/ppo_submarine_ep*.pth')
            if list_of_files: model_path = max(list_of_files, key=os.path.getctime)
            try:
                start_episode = int(model_path.split("ep")[-1].split(".")[0])
            except: pass
        try:
            agent.load(model_path)
            print(f"Loaded: {model_path}")
        except: pass
    
    if LOAD_MODEL and os.path.exists("training_state.npy"):
        try:
            state_data = np.load("training_state.npy", allow_pickle=True).item()
            saved_histories = state_data.get('map_histories', {})
            for k, v in saved_histories.items(): map_histories[k] = v
            curriculum_state = state_data.get('curriculum_state', curriculum_state)
            print("Training state loaded.")
        except: pass

    for episode in range(start_episode, NUM_EPISODES):
        
        weights = calculate_dynamic_weights(map_stats, len(MAP_FILES))
        map_idx = random.choices(range(len(MAP_FILES)), weights=weights, k=1)[0]
        
        map_stats[map_idx]['attempts'] += 1
        space, cave_env = preloaded_maps[map_idx]
        
        cave_env.batteries.empty()
        cave_env.obstacles.empty()
        with open(MAP_FILES[map_idx]) as csvfile:
            reader = csv.reader(csvfile)
            for y, row in enumerate(reader):
                for x, tile in enumerate(row):
                    if tile == '20': cave_env.batteries.add(Battery(x * 16, y * 16))
                    elif tile == '21': cave_env.obstacles.add(Obstacle(x * 16, y * 16))
        
        for body in space.bodies:
            if body.body_type in [pymunk.Body.KINEMATIC, pymunk.Body.DYNAMIC]:
                space.remove(body)
                for shape in body.shapes: space.remove(shape)
        
        if not space: continue

        start_x, start_y = 100, 300
        target_x_min = 50
        
        if map_idx in curriculum_state:
            curr_x = curriculum_state[map_idx]
            variance = random.randint(-50, 50)
            target_x_min = max(50, min(curr_x + variance, 2800))
        
        # Improved spawn logic
        wall_rects = [t.rect for t in cave_env.environment_tiles]
        wall_rects.extend([o.rect for o in cave_env.obstacles])
        
        search_start_x = target_x_min
        start_x, start_y = 100, 300
        found_start = False
        
        search_zones = list(range(search_start_x, search_start_x + 200, 20)) + \
                       list(range(search_start_x, max(50, search_start_x - 200), -20)) + \
                       list(range(100, 500, 20))
                       
        for x in search_zones:
             if x >= cave_env.environment_width - 50: continue
             valid_ys = []
             for y in range(50, cave_env.environment_height - 50, 20):
                 check_rect = pygame.Rect(x - 40, y - 40, 80, 80)
                 if check_rect.collidelist(wall_rects) == -1:
                     valid_ys.append(y)
             if len(valid_ys) > 0:
                 start_x = x
                 start_y = random.choice(valid_ys)
                 found_start = True
                 break
        
        if not found_start:
             print(f"WARNING: Could not find ANY spawn for Map {map_idx}. forcing 100,300")
             start_x, start_y = 100, 350
             
        submarine = Submarine(start_x, start_y)
        
        if "one_battery" in MAP_FILES[map_idx] or "three_battery" in MAP_FILES[map_idx]:
             submarine.battery = 300
        else:
             submarine.battery = 600
            
        sonar_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        sonar_body.position = (submarine.rect.centerx, submarine.rect.centery)
        sonar = Sonar(space, sonar_body)
        
        state = get_full_state(sonar.get_observation(), submarine, map_idx, cave_env.environment_width, cave_env.batteries)
        total_reward = 0
        done = False
        stagnation_timer = 0
        stagnation_start_x = submarine.true_x
        
        # Run-specific trackers
        current_run_picked_battery = False
        hit_obstacle_this_run = False
        hit_wall_this_run = False  # Track wall hits for dirty/clean stats
        
        action = 4 
        log_prob = 0.0
        value = 0.0

        for step in range(MAX_STEPS):
            if WATCH_MODE or step % 10 == 0:
                 for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_TAB: WATCH_MODE = not WATCH_MODE
                        if event.key == pygame.K_ESCAPE: pygame.quit(); return

            if step % 4 == 0:
                action, log_prob, value = agent.select_action(state)
            
            prev_x = submarine.true_x
            prev_y = submarine.true_y
            
            if action == 0: submarine.move_up()
            elif action == 1: submarine.move_down()
            elif action == 2: submarine.move_left()
            elif action == 3: submarine.move_right()
            elif action == 4: pass 
            
            submarine.update()
            sonar_body.position = (submarine.rect.centerx, submarine.rect.centery)
            
            reward = 0.0
            
            # Progress
            dist_x = submarine.true_x - prev_x
            reward += np.clip(dist_x * 0.5, -1.0, 2.0) 
            if action == 4: reward += 0.05
            
            # Batteries
            hits = pygame.sprite.spritecollide(submarine, cave_env.batteries, True)
            for hit in hits: 
                submarine.battery += 300
                reward += 10.0
                current_run_picked_battery = True
            
            # Obstacles
            obs_hits = pygame.sprite.spritecollide(submarine, cave_env.obstacles, False)
            for hit in obs_hits: 
                submarine.battery -= 50
                reward -= 5.0
                hit_obstacle_this_run = True

            next_observation = sonar.get_observation()
            next_state = get_full_state(next_observation, submarine, map_idx, cave_env.environment_width, cave_env.batteries)

            # Crash logic
            hit_wall = False
            for reading in next_observation:
                if reading == 0: hit_wall = True; break
            
            if hit_wall:
                hit_wall_this_run = True # Track wall hit
                reward -= 50.0 
                submarine.battery -= 10
                submarine.true_x = prev_x
                submarine.true_y = prev_y
                submarine.rect.x = int(prev_x)
                submarine.rect.y = int(prev_y)
                submarine.vel_x *= -0.5; submarine.vel_y *= -0.5

            # Goal logic
            if submarine.rect.right >= cave_env.environment_width - 10:
                reward += 100.0
                if len(cave_env.batteries) == 0: reward += 50.0
                reward += submarine.battery * 0.01
                done = True
                map_histories[map_idx].append(1)
                map_stats[map_idx]['goals'] += 1

            if submarine.battery <= 0:
                reward -= 10.0; done = True; map_histories[map_idx].append(0)
            
            # Stagnation
            stagnation_timer += 1
            if stagnation_timer >= 400:
                if abs(submarine.true_x - stagnation_start_x) < 100:
                    reward -= 5.0; done = True; map_histories[map_idx].append(0)
                else: stagnation_timer = 0; stagnation_start_x = submarine.true_x

            # PPO storage & update
            if step % 4 == 0:
                agent.store_transition(state, action, log_prob, reward, value, done)
                total_steps += 1
                if total_steps % UPDATE_INTERVAL == 0:
                    loss = agent.train_step()
                    if loss is not None: 
                        loss_history.append(loss)
                        print(f"PPO Update @ {total_steps}: Loss {loss:.4f}")

            state = next_state
            total_reward += reward

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
                info_text = f"Map: {map_idx} | Ep: {episode} | Step: {step}"
                batt_text = f"Battery: {submarine.battery:.1f}% | Reward: {total_reward:.1f}"
                mode_text = "Press TAB for FAST MODE"
                screen.blit(font.render(info_text, True, (255, 255, 255)), (10, 10))
                screen.blit(font.render(batt_text, True, (255, 255, 0)), (10, 40))
                screen.blit(font.render(mode_text, True, (0, 255, 0)), (10, 70))
                if hit_wall: screen.blit(hit_font.render("HIT!", True, (255,0,0)), (500, 50))
                pygame.display.flip()
                clock.tick(60)
            else:
                if step % 500 == 0:
                    screen.fill((0, 0, 0))
                    msg = font.render(f"FAST MODE (Ep {episode}). TAB to Watch.", True, (0, 255, 0))
                    screen.blit(msg, (500, 400))
                    pygame.display.flip()

            if done: break
        
        # End of episode stats updates
        map_stats[map_idx]['total_reward'] += total_reward
        is_success = map_histories[map_idx][-1] == 1
        
        # Battery stats
        if current_run_picked_battery:
            battery_stats[map_idx]['picked_up'] += 1
            if is_success: battery_stats[map_idx]['picked_success'] += 1
            else: battery_stats[map_idx]['picked_fail'] += 1
        else:
            if is_success: battery_stats[map_idx]['ignored_success'] += 1
            else: battery_stats[map_idx]['ignored_fail'] += 1
            
        # Obstacle stats
        if hit_obstacle_this_run:
            if is_success: obstacle_stats[map_idx]['hit_won'] += 1
            else: obstacle_stats[map_idx]['hit_died'] += 1
        else:
            if is_success: obstacle_stats[map_idx]['avoided_won'] += 1
            else: obstacle_stats[map_idx]['avoided_died'] += 1

        # Clean run stats
        is_clean = (not hit_wall_this_run) and (not hit_obstacle_this_run)
        if is_success:
            if is_clean: clean_run_stats[map_idx]['clean_win'] += 1
            else: clean_run_stats[map_idx]['dirty_win'] += 1
        else:
            if is_clean: clean_run_stats[map_idx]['clean_fail'] += 1
            else: clean_run_stats[map_idx]['dirty_fail'] += 1

        history = map_histories[map_idx]
        if len(history) >= 50:
            recent = history[-50:]
            sr = sum(recent) / len(recent)
            current_x = curriculum_state.get(map_idx, 2500)
            
            if sr > 0.80:
                curriculum_state[map_idx] = max(100, current_x - 50)
                map_histories[map_idx] = [] # Reset to prove mastery
            elif sr < 0.20:
                curriculum_state[map_idx] = min(2500, current_x + 50)
                map_histories[map_idx] = []

        if episode % SAVE_INTERVAL == 0:
            agent.save(f"models/ppo_submarine_ep{episode}.pth")
            state_data = {
                'map_histories': map_histories,
                'curriculum_state': curriculum_state
            }
            np.save("training_state.npy", state_data)

    agent.save("models/ppo_submarine_final.pth")
    
    training_end_time = time.time()
    total_seconds = training_end_time - training_start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    # Final detailed printouts
    print("\n" + "="*50)
    print("TRAINING COMPLETE - FINAL STATISTICS")
    print("="*50)
    # Print the time
    print(f"Total Training Time: {hours}h {minutes}m {seconds}s") 
    print(f"Total Episodes:      {NUM_EPISODES}")
    print("-" * 50)
    print(f"{'Map File':<40} | {'Goals':<5} | {'Atts':<6} | {'SR %':<6} | {'Avg Rwd':<8}")
    print("-" * 80)
    for i, filename in enumerate(MAP_FILES):
        if i in map_stats and map_stats[i]['attempts'] > 0:
            stats = map_stats[i]
            sr = (stats['goals'] / stats['attempts']) * 100
            avg = stats['total_reward'] / stats['attempts']
            name = filename.split('/')[-1]
            print(f"{name:<40} | {stats['goals']:<5} | {stats['attempts']:<6} | {sr:>6.1f} | {avg:>8.1f}")
            
    print("\n" + "="*50)
    print("BATTERY STATS")
    print("="*50)
    print(f"{'Map File':<30} | {'Picked(Up)':<10} | {'Picked(Win)':<11} | {'Picked(Die)':<11} | {'Ignored(Die)':<12} | {'Ignored(Win)':<12}")
    print("-" * 100)
    for i, filename in enumerate(MAP_FILES):
        if i in battery_stats and (battery_stats[i]['picked_up'] > 0 or battery_stats[i]['ignored_fail'] > 0):
            bs = battery_stats[i]
            name = filename.split('/')[-1]
            print(f"{name:<30} | {bs['picked_up']:<10} | {bs['picked_success']:<11} | {bs['picked_fail']:<11} | {bs['ignored_fail']:<12} | {bs['ignored_success']:<12}")

    print("\n" + "="*50)
    print("OBSTACLE STATS")
    print("="*50)
    print(f"{'Map File':<30} | {'Avoid(Win)':<11} | {'Avoid(Die)':<11} | {'Hit(Die)':<10} | {'Hit(Win)':<10}")
    print("-" * 80)
    for i, filename in enumerate(MAP_FILES):
        if i in obstacle_stats and (obstacle_stats[i]['avoided_won'] > 0 or obstacle_stats[i]['hit_died'] > 0 or obstacle_stats[i]['hit_won'] > 0):
            obs = obstacle_stats[i]
            name = filename.split('/')[-1]
            print(f"{name:<30} | {obs['avoided_won']:<11} | {obs['avoided_died']:<11} | {obs['hit_died']:<10} | {obs['hit_won']:<10}")
            
    print("\n" + "="*50)
    print("CLEAN RUN STATS (No Wall/Obstacle Hits)")
    print("="*50)
    print(f"{'Map File':<30} | {'Clean Win':<10} | {'Dirty Win':<10} | {'Clean Fail':<10} | {'Dirty Fail':<10}")
    print("-" * 80)
    for i, filename in enumerate(MAP_FILES):
        if i in clean_run_stats and map_stats[i]['attempts'] > 0:
            cs = clean_run_stats[i]
            name = filename.split('/')[-1]
            print(f"{name:<30} | {cs['clean_win']:<10} | {cs['dirty_win']:<10} | {cs['clean_fail']:<10} | {cs['dirty_fail']:<10}")

    print("="*50)
    
    if loss_history:
        np.save("training_loss.npy", np.array(loss_history))
        print("Loss saved.")

    pygame.quit()

if __name__ == "__main__":
    train()

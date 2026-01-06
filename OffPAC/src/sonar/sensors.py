import pygame
import pymunk
import math
import numpy as np

class Sonar:
    def __init__(self, space, body, num_rays=16, max_range=1000, agent_size=40):
        self.space = space
        self.body = body
        self.num_rays = num_rays
        self.max_range = max_range
        self.agent_size = agent_size
    
    def _get_surface_offset(self, angle_rad):
        angle_rad = (angle_rad + math.pi) % (2 * math.pi) - math.pi
        half_size = self.agent_size / 2
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        if abs(cos_a) < 0.0001: cos_a = 0.0001
        if abs(sin_a) < 0.0001: sin_a = 0.0001
        return half_size / max(abs(cos_a), abs(sin_a))

    def get_observation(self):
        start_angle = self.body.angle
        step_angle = (2 * math.pi) / self.num_rays
        readings = []

        for i in range(self.num_rays):
            local_angle = i * step_angle
            world_angle = start_angle + local_angle
            direction = pymunk.Vec2d(math.cos(world_angle), math.sin(world_angle))
            
            dist_to_edge = self._get_surface_offset(local_angle)
            start_pos = self.body.position + direction * (dist_to_edge - 10)
            end_pos = start_pos + direction * self.max_range
            
            result = self.space.segment_query_first(start_pos, end_pos, 1, pymunk.ShapeFilter())
            
            visual_start_pos = self.body.position + direction * dist_to_edge
            actual_end = result.point if result else end_pos
            
            raw_dist = visual_start_pos.get_distance(actual_end) if result else self.max_range
            dist_center_to_hit = self.body.position.get_distance(actual_end) if result else 9999
            
            distance = raw_dist
            if result and dist_center_to_hit < dist_to_edge:
                 distance = 0

            if distance < 2.0:
                distance = 0

            readings.append(distance / self.max_range)

        return np.array(readings)

    def draw(self, surface, font):
        start_angle = self.body.angle
        step_angle = (2 * math.pi) / self.num_rays
        data = self.get_observation()
        hit_wall = False

        for i in range(self.num_rays):
            distance = data[i] * self.max_range
            if distance == 0:
                hit_wall = True
                
            local_angle = i * step_angle
            world_angle = start_angle + local_angle
            direction = pymunk.Vec2d(math.cos(world_angle), math.sin(world_angle))
            dist_to_edge = self._get_surface_offset(local_angle)
            
            visual_start_pos = self.body.position + direction * dist_to_edge
            actual_end = visual_start_pos + direction * distance
            
            pygame.draw.line(surface, (255, 255, 255), visual_start_pos, actual_end, 1)
            
            if distance < self.max_range:
                text = font.render(f"{int(distance)}", True, (200, 200, 200))
                surface.blit(text, actual_end + (5, 5))

        if hit_wall:
            msg = font.render("HIT A WALL!", True, (255, 50, 50))
            surface.blit(msg, (surface.get_width() - 120, 10))


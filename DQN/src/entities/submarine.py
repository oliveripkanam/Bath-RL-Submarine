import pygame, math

class Submarine:
    def __init__(self, x, y):
        self.image = pygame.image.load('src/entities/submarine.png')
        self.image = pygame.transform.flip(self.image, True, False)
        self.rect = self.image.get_rect(topleft=(x, y))

        self.vel_x = 0
        self.vel_y = 0
        self.acceleration = 3.0
        self.friction = 0.93
        self.gravity = 0.05
        self.max_speed = 10

        self.true_x = float(x)
        self.true_y = float(y)

        self.battery = 600

    def use_battery(self):
        self.battery -= 1

    def move_up(self):
        self.vel_y -= self.acceleration
        self.use_battery()

    def move_down(self):
        self.vel_y += self.acceleration
        self.use_battery()

    def move_left(self):
        self.vel_x -= self.acceleration
        self.use_battery()

    def move_right(self):
        self.vel_x += self.acceleration
        self.use_battery()

    def update(self):
        self.vel_y += self.gravity
        self.vel_x *= self.friction
        self.vel_y *= self.friction

        speed = math.hypot(self.vel_x, self.vel_y)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.vel_x *= scale
            self.vel_y *= scale

        self.true_x += self.vel_x
        self.true_y += self.vel_y

        self.rect.x = int(self.true_x)
        self.rect.y = int(self.true_y)

    def draw(self, surface):
        surface.blit(self.image, self.rect)
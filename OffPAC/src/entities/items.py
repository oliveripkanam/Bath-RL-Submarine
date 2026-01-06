import pygame

class Battery(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        raw_image = pygame.image.load('src/entities/healthpack.png').convert_alpha()
        self.image = pygame.transform.smoothscale(raw_image, (24, 24)) # Resize to fit grid
        self.rect = self.image.get_rect(center=(x + 8, y + 8)) # Center on the 16x16 tile

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        raw_image = pygame.image.load('src/entities/pufferfish.png').convert_alpha()
        self.image = pygame.transform.smoothscale(raw_image, (24, 24))
        self.rect = self.image.get_rect(center=(x + 8, y + 8))


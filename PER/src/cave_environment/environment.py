import pygame, csv
from .spritesheet import SpriteSheet
from src.entities.items import Battery, Obstacle

# Reference: https://www.youtube.com/watch?v=37phHwLtaFg

class Tile(pygame.sprite.Sprite):
    def __init__(self, image, x, y, spritesheet: SpriteSheet):
        pygame.sprite.Sprite.__init__(self)
        self.image = spritesheet.get_sprite(image)
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = x, y
    
    def draw(self, surface):
        surface.blit(self.image, self.rect)
        
class CaveEnvironment:
    def __init__(self, plot_filename, spritesheet):
        self.tile_size = 16
        self.start_x, self.start_y= 0, 0
        self.spritesheet = spritesheet
        
        self.batteries = pygame.sprite.Group()
        self.obstacles = pygame.sprite.Group()
        
        self.environment_tiles = self.load_tiles(plot_filename)
        
        # Calculate dimensions based on the data just loaded
        # Find the tile with the max x and max y coordinates
        if self.environment_tiles:
            max_x = max(tile.rect.right for tile in self.environment_tiles)
            max_y = max(tile.rect.bottom for tile in self.environment_tiles)
            self.environment_width = max_x
            self.environment_height = max_y
        else:
            self.environment_width = 800 # Default fallback
            self.environment_height = 600

        self.environment_surface = pygame.Surface((self.environment_width, self.environment_height))
        self.environment_surface.set_colorkey((0, 0, 0))
        self.load_environment()
        
        
    def read_csv_file(self, filename):
        environment_data = []
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                environment_data.append(list(row))
                
        return environment_data
    
    def load_tiles(self, filename):
        environment_tiles = []
        environment_data = self.read_csv_file(filename)
        x, y  = 0, 0
        for row in environment_data:
            x = 0
            for tile in row:
                if tile == '20': # Battery
                    self.batteries.add(Battery(x * self.tile_size, y * self.tile_size))
                elif tile == '21': # Obstacle
                    self.obstacles.add(Obstacle(x * self.tile_size, y * self.tile_size))
                
                # Create wall tile
                elif tile != '-1':
                    environment_tiles.append(
                        Tile(
                            self.spritesheet.labels[int(tile) // 5][int(tile) % 5],
                            x * self.tile_size,
                            y * self.tile_size,
                            self.spritesheet,
                        )
                    )
                x += 1
            y += 1

        return environment_tiles
        
        
    def load_environment(self):
        for tile in self.environment_tiles:
            tile.draw(self.environment_surface)
            
    def draw(self, surface):
        surface.blit(self.environment_surface, (self.start_x, self.start_y))
        self.batteries.draw(surface)
        self.obstacles.draw(surface)

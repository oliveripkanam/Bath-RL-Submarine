import pygame

class SpriteSheet:
    def __init__(self, filename):
        self.filename = filename
        self.sprite_sheet = pygame.image.load(filename).convert_alpha() 
        self.tile_size = 16    
        labels = [["inside_top_left", "inside_top_right", "outside_top_left", "outside_top_middle", "outside_top_right"],
                  ["inside_bottom_left", "inside_bottom_right", "outside_middle_left", "middle", "outside_middle_right"],
                  ["empty", "empty", "outside_bottom_left", "outside_bottom_middle", "outside_bottom_right"]]
        
        pixel_map = {labels[i][j]: (j * self.tile_size, i * self.tile_size, self.tile_size, self.tile_size) 
                     for i in range(len(labels)) for j in range(len(labels[0])) if labels[i][j] != "empty"}
        self.pixel_map = pixel_map
        self.labels = labels

    def get_sprite(self, label):
        sprite = pygame.Surface((self.pixel_map[label][2], self.pixel_map[label][3]))
        sprite.blit(self.sprite_sheet, (0, 0), self.pixel_map[label])
        sprite.set_colorkey((0, 0, 0))
        return sprite
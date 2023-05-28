import math

import pygame as pygame

from road import Road
from visualisation import from_id


class RoadAnimation:
    road: Road
    screen_width: int
    screen_height: int
    road_width: int
    speed: float
    vehicle_size: float
    screen: pygame.Surface

    def __init__(self,
                 road: Road,
                 screen_width: int = 400,
                 screen_height: int = 600,
                 road_width: int = 10,
                 speed: float = 1.,
                 vehicle_size: float = 2):
        self.road = road
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.road_width = road_width
        self.speed = speed
        self.radius = min(screen_width, screen_height) * 0.4
        self.vehicle_size = vehicle_size
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height], flags=pygame.HIDDEN)

    def draw_car(self, position, color):
        angle = 2 * position/self.road.length * math.pi

        car_width = self.road_width * 1.1
        car_length = 2 * math.pi * self.radius * self.vehicle_size / self.road.length

        center_x = self.screen_width // 2 + self.radius * math.cos(angle)
        center_y = self.screen_height // 2 + self.radius * math.sin(angle)

        width_diff_x, width_diff_y = car_width/2 * math.cos(angle), car_width/2 * math.sin(angle)
        length_diff_x, length_diff_y = car_length/2 * math.sin(angle), - car_length/2 * math.cos(angle)

        border = [[center_x + a * width_diff_x + b * length_diff_x, center_y + a * width_diff_y + b * length_diff_y]
                  for a, b in [(1.25, 1.15), (1.25, -1.15), (-1.25, -1.15), (-1.25, 1.15)]]
        corners = [[center_x + a * width_diff_x + b * length_diff_x, center_y + a * width_diff_y + b * length_diff_y]
                   for a, b in [(1, 1), (1, -1), (-1, -1), (-1, 1)]]
        pygame.draw.polygon(self.screen, (0, 0, 0), border)
        pygame.draw.polygon(self.screen, color, corners)

    def draw_road(self):
        pygame.draw.circle(self.screen, (100, 100, 100), (self.screen_width // 2, self.screen_height // 2),
                           self.radius + self.road_width/2, self.road_width)

    def draw_frame(self, t):
        self.draw_road()
        positions = [self.road.road_data[t, n][0] for n in range(self.road.number_of_vehicles)]
        for i, pos in enumerate(positions):
            self.draw_car(pos, from_id(i))

    def show(self):
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height], flags=pygame.SHOWN)
        pygame.init()
        running = True
        clock = pygame.time.Clock()
        limit = self.road.time_horizon if self.road.crashed_at < 0 else self.road.crashed_at
        fps = int(30 * self.speed)

        for t in range(limit):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not running:
                self.screen = pygame.display.set_mode([self.screen_width, self.screen_height], flags=pygame.HIDDEN)
            self.screen.fill((255, 255, 255))
            self.draw_frame(t)
            pygame.display.flip()
            clock.tick(fps)

    def dispose(self):
        pygame.quit()
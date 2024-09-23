from typing import List, Tuple

import numpy as np

import pygame
import math
import time
import random

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
class Game:
    QUAD_COUNT=1
    FPS=30
    FRAMES_PER_STEP = 6 # TEMP
    MAP_SIZE=256
    RENDER_SCALE=3
    RESOLUTION=256*RENDER_SCALE

    n_actions=9
    frames_since_last_attack=0

    prev_input = 0

    running = True

    done = False
    reward = 0

    all_frames = 0
    current_frame = 0

    def __init__(self, render=False, render_interval=1, limit_fps=False):
        self.RENDER = render
        self.RENDER_INTERVAL = render_interval
        self.START_TIME = time.time()
        self.LAST_FRAME_TIME = time.time()
        self.LIMIT_FPS=limit_fps

        self.reset()

        if (self.RENDER):
            pygame.init()
            self.screen = pygame.display.set_mode((256*self.RENDER_SCALE, 256*self.RENDER_SCALE))

    def reset(self):
        self.humvee = Humvee()
        self.quads = []
        self.humvee.pos = [random.random(), random.random()]
        for quad in range(self.QUAD_COUNT):
            self.quads.append(QuadCannon())
        #for quad in self.quads:
        #    quad.pos = [self.humvee.pos[0]+random.random()*0.0001, self.humvee.pos[1]+random.random()*0.0001]
        self.frames_since_last_attack = 0
        self.done = False

        return self.get_state(), {}

    # Returns
    # - 9 ints, representing which input was last
    # - 4 floats, representing the distance from the edges of the screen
    # - X distance from closest Quad Cannon
    # - Y distance from closest Quad Cannon
    def get_state(self):
        closest_quad = self.__get_closest_quad()

        last_input = np.zeros(9, dtype=np.float32)
        last_input[self.prev_input] = 1.0

        if (closest_quad == -1):
            return np.array([
            *last_input,
            self.humvee.pos[0],
            self.humvee.pos[1],
            1.0-self.humvee.pos[0],
            1.0-self.humvee.pos[1],
            0,
            0,
            ])
        return np.array([
            *last_input,
            self.humvee.pos[0],
            self.humvee.pos[1],
            1.0-self.humvee.pos[0],
            1.0-self.humvee.pos[1],
            closest_quad.pos[0]-self.humvee.pos[0],
            closest_quad.pos[1]-self.humvee.pos[1],
        ])
    
    def __get_closest_quad(self):
        if (len(self.quads) == 0):
            return -1
        return min(self.quads, key=lambda quad: self.__distance(self.humvee, quad))

    def __distance(self, unit_1, unit_2):
        return ((unit_1.pos[0]-unit_2.pos[0])**2+(unit_1.pos[1]-unit_2.pos[1])**2) ** 0.5

    # Plays the game for 200ms
    def step(self, action= -1) -> Tuple[List[float], float, bool]:
        self.last_input = action
        if (action != -1):
            self.__set_humvee_dest(action)
        self.reward = 0

        for _ in range(self.FRAMES_PER_STEP):
            self.__step_frame()

        return self.get_state(), self.reward, self.done, False, {}

    def __set_humvee_dest(self, dir: int):
        direction_vectors = {
            0: (0, 0),      # No movement
            1: (0, 1),      # North
            2: (1, 1),      # Northeast
            3: (1, 0),      # East
            4: (1, -1),     # Southeast
            5: (0, -1),     # South
            6: (-1, -1),    # Southwest
            7: (-1, 0),     # West
            8: (-1, 1)      # Northwest
        }
        
        if dir in direction_vectors:
            dir_vector = direction_vectors[dir]
            
            if dir != 0:
                # Calculate new destination position
                angle = math.atan2(dir_vector[1], dir_vector[0])
                dest_x = self.humvee.pos[0] + 0.2 * math.cos(angle)
                dest_y = self.humvee.pos[1] + 0.2 * math.sin(angle)
                self.humvee.dest_pos = [dest_x, dest_y]
            else:
                # No movement
                self.humvee.dest_pos = self.humvee.pos.copy()

    # Plays 1 frame
    def __step_frame(self):

        if (self.LIMIT_FPS):
            self.__limit_fps()

        self.frames_since_last_attack += 1
        if (self.frames_since_last_attack > self.FPS*200):
            self.done = True
            print('Ran out of frames')

        # Move units
        self.__move_unit(self.humvee, self.humvee.dest_pos)

        distance_from_center = math.sqrt((self.humvee.pos[0]-0.5)**2+(self.humvee.pos[1]-0.5)**2)
        if distance_from_center > 0.5:
            self.reward -= 500*abs(distance_from_center - 0.3) / self.FPS
        for quad in self.quads:
            self.__move_unit(quad, self.humvee.pos)

        # Attack with units
        if (len(self.quads) > 0):
            humvee_attacked = self.__attack(self.humvee, self.__get_closest_quad())
            if humvee_attacked:
                self.reward += 20
        for quad in self.quads:
            quad_attacked = self.__attack(quad, self.humvee)
            if quad_attacked:
                self.reward -= 100 * (1.0-self.__distance(self.humvee, quad))
        self.humvee.reload()
        for quad in self.quads:
            quad.reload()

        # Check for dead units
        if (self.humvee.health <= 0):
            self.done = True

        if len(self.quads) == 0:
            self.done = True
            print('Quads died! :)')

        for quad in self.quads:
            if quad.health <= 0:
                self.reward += 100

        # Remove dead quads
        self.quads = [quad for quad in self.quads if quad.health > 0]

        # Render
        if (self.RENDER):
            self.__handle_pygame_events()
            self.render()
        
    def __limit_fps(self):
        CURRENT_TIME = time.time()
        TIME_DIFF = (CURRENT_TIME - self.LAST_FRAME_TIME)
        MIN_TIME_DIFF = 1000.0 / self.FPS

        if TIME_DIFF < MIN_TIME_DIFF:
            time.sleep((MIN_TIME_DIFF-TIME_DIFF) / 1000.0)
        
        self.LAST_FRAME_TIME = CURRENT_TIME

    def __move_unit(self, unit, pos):
        PREV_POS = unit.pos
        TARGET_POS = pos
        SPEED = unit.SPEED / self.MAP_SIZE / 100.0
        MOVE_AWAY_DISTANCE = 1.5 / 100.0  # Minimum distance to maintain between units

        # Function to get the distance between two units
        def distance(u1, u2):
            return ((u1.pos[0] - u2.pos[0]) ** 2 + (u1.pos[1] - u2.pos[1]) ** 2) ** 0.5

        # Check distance to all other units and adjust target position if necessary
        if unit.NAME != 'humvee':
            for other_unit in self.quads + [self.humvee]:
                if other_unit != unit:
                    dist = distance(unit, other_unit)
                    if dist < MOVE_AWAY_DISTANCE:
                        if dist == 0:  # If the distance is zero, move in a random direction
                            dir_x = random.uniform(-1, 1)
                            dir_y = random.uniform(-1, 1)
                        else:
                            # Calculate direction to move away from the other unit
                            dir_x = unit.pos[0] - other_unit.pos[0]
                            dir_y = unit.pos[1] - other_unit.pos[1]
                            length = math.sqrt(dir_x ** 2 + dir_y ** 2)
                            dir_x /= length
                            dir_y /= length
                        
                        # Move the unit away from the other unit
                        TARGET_POS = [unit.pos[0] + dir_x * MOVE_AWAY_DISTANCE, unit.pos[1] + dir_y * MOVE_AWAY_DISTANCE]

        # Calculate the desired angle
        delta_x = TARGET_POS[0] - PREV_POS[0]
        delta_y = TARGET_POS[1] - PREV_POS[1]
        if delta_x == 0 and delta_y == 0:
            desired_angle = unit.angle
        else:
            desired_angle = math.atan2(delta_y, delta_x)

        # Turn the unit towards the desired angle
        angle_diff = desired_angle - unit.angle
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        turn_amount = unit.TURNING_SPEED / self.FPS
        if abs(angle_diff) <= turn_amount:
            unit.angle = desired_angle
        else:
            unit.angle += turn_amount if angle_diff > 0 else -turn_amount

        # Calculate the new position based on the angle
        if unit.angle != desired_angle:
            # If not facing the desired direction, do not move
            return

        # If position is inside move square then just move there
        if abs(delta_x) < SPEED and abs(delta_y) < SPEED:
            unit.pos = TARGET_POS
        else:
            if abs(delta_x) > abs(delta_y):
                x_move = min(SPEED, abs(delta_x))
                y_move = delta_y * (SPEED / abs(delta_x))

                if delta_x < 0:
                    x_move = -x_move

                unit.pos = [unit.pos[0] + x_move, unit.pos[1] + y_move]
            else:
                y_move = min(SPEED, abs(delta_y))
                x_move = delta_x * (SPEED / abs(delta_y))

                if delta_y < 0:
                    y_move = -y_move

                unit.pos = [unit.pos[0] + x_move, unit.pos[1] + y_move]

        # Teleport the unit back onto the map if it goes off the map
        unit.pos[0] = max(0, min(unit.pos[0], 1))
        unit.pos[1] = max(0, min(unit.pos[1], 1))


    def __attack(self, unit_1, unit_2) -> bool:
        distance = math.sqrt((unit_1.pos[0]-unit_2.pos[0])**2+(unit_1.pos[1]-unit_2.pos[1])**2) * 10 * self.MAP_SIZE
        if distance > unit_1.RANGE:
            return False
        
        self.frames_since_last_attack = 0

        if unit_1.isReloaded():
            unit_1.fire()
            unit_2.health -= unit_1.DAMAGE
            return True
        
        return False

    def render(self):
        self.current_frame += 1
        self.all_frames += 1

        if self.all_frames % self.RENDER_INTERVAL == 0:
            # Fill the screen with black
            self.screen.fill((0, 0, 0))  # Note: use (0, 0, 0) for black

            # Draw Humvee as a blue circle with radius 2 pixels
            humvee_coord = self.__get_screen_coordinate(self.humvee)
            pygame.draw.circle(self.screen, (100, 100, 255), humvee_coord, self.RENDER_SCALE * self.humvee.RANGE * 0.1)  # Light blue circle
            for quad in self.quads:
                quad_coord = self.__get_screen_coordinate(quad)
                pygame.draw.circle(self.screen, (200, 50, 50,), quad_coord, self.RENDER_SCALE * quad.RANGE * 0.1)  # Light red circle

            # Draw QuadCannons as red circles with radius 2 pixels
            for quad in self.quads:
                quad_coord = self.__get_screen_coordinate(quad)
                pygame.draw.circle(self.screen, (255, 0, 0), quad_coord, 1*self.RENDER_SCALE)  # Red circle

            pygame.draw.circle(self.screen, (0, 0, 255), humvee_coord, 1*self.RENDER_SCALE)  # Blue circle

            # Update the display
            pygame.display.flip()

    def __get_screen_coordinate(self, unit):
        return [math.floor(i * self.MAP_SIZE * self.RENDER_SCALE) for i in unit.pos]

    def __handle_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.RENDER = False
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                self.humvee.dest_pos = [pos[0]/self.RESOLUTION, pos[1]/self.RESOLUTION]



class Humvee:
    NAME = 'humvee'
    MAX_HEALTH = 240
    RANGE = 175 * 1.3 # Missile Defenders & SND
    SPEED = 60 * 1.5 # TEMP speed boost
    TURNING_SPEED = 8 * 1.1
    FIRING_SPEED = 1000 # Milliseconds
    DAMAGE = 40 * 5 / 10.0

    health = MAX_HEALTH
    reload_progress = 1.0
    angle = 0.0

    pos = [0.25,0.5]
    dest_pos = pos

    def fire(self):
        self.reload_progress = 0.0

    def reload(self):
        if (self.isReloaded()):
            return
        
        new_progress = self.reload_progress + (1000.0 / self.FIRING_SPEED) / 30.0
        if new_progress > 1.0:
            self.reload_progress = 1.0
        else:
            self.reload_progress = new_progress

    def isReloaded(self):
        return self.reload_progress == 1.0


class QuadCannon:
    NAME = 'quad'
    MAX_HEALTH = 300
    RANGE = 150 * 1.2 # TMP
    SPEED = 40 # 40 generals units/second
    TURNING_SPEED = 8
    FIRING_SPEED = 50
    DAMAGE = 10 / 100.0

    health = MAX_HEALTH
    reload_progress = 1.0
    angle = 0.0
    
    pos = [0.75, 0.5]
    dest_pos = pos

    def fire(self):
        self.reload_progress = 0.0

    def reload(self):
        if (self.isReloaded()):
            return
        
        new_progress = self.reload_progress + (1000.0 / self.FIRING_SPEED) / 30.0
        if new_progress > 1.0:
            self.reload_progress = 1.0
        else:
            self.reload_progress = new_progress

    def isReloaded(self):
        return self.reload_progress == 1.0

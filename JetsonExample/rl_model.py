import numpy as np
from stable_baselines3 import PPO
import copy
from threading import Lock
import sys

# Import scripts from submodule
sys.path.append('VEX-AI-Reinforcement-Learning')
import path_planner
import rl_environment

# Actions
Actions = rl_environment.Actions

# Constants
ROBOT_LENGTH = 15 # inches
ROBOT_WIDTH = 15 # inches
INCHES_PER_FIELD = rl_environment.INCHES_PER_FIELD
ENV_FIELD_SIZE = rl_environment.ENV_FIELD_SIZE
BUFFER_RADIUS = rl_environment.BUFFER_RADIUS
NUM_WALL_STAKES = rl_environment.NUM_WALL_STAKES
NUM_GOALS = rl_environment.NUM_GOALS
NUM_RINGS = rl_environment.NUM_RINGS
TIME_LIMIT = rl_environment.TIME_LIMIT
DEFAULT_PENALTY = rl_environment.DEFAULT_PENALTY


class Observation:
    def __init__(self):
        self.__state = {
            'robot_x': np.zeros((1,), dtype=np.float32),
            'robot_y': np.zeros((1,), dtype=np.float32),
            'robot_orientation': np.zeros((1,), dtype=np.float32),
            'holding_goal': 0,
            'holding_rings': 0,
            'rings': np.full((NUM_RINGS * 2,), -1, dtype=np.float32),
            'goals': np.full((NUM_GOALS * 2,), -1, dtype=np.float32),
            'wall_stakes': np.zeros(NUM_WALL_STAKES, dtype=np.int32),
            'holding_goal_full': 0,
            'time_remaining': np.zeros((1,), dtype=np.float32),
            'visible_rings_count': 0,
            'visible_goals_count': 0,
        }
        self.__lock = Lock()

    def update_from_brain(self, new_data: str):
        # Define the format for observation packets from the brain here
        fields = new_data.split(',')
        try:
            # Parse everything before writing state to avoid partial writes
            robot_x = float(fields[0])
            robot_y = float(fields[1])
            robot_orientation = float(fields[2])
            time_remaining = float(fields[3])

            self.__lock.acquire()

            self.__state['robot_x'][0] = robot_x
            self.__state['robot_y'][0] = robot_y
            self.__state['robot_orientation'][0] = robot_orientation
            self.__state['time_remaining'][0] = time_remaining

            self.__lock.release()
        except ValueError:
            print('WARNING: Invalid observation packet from brain')

    def update_from_camera(self, obj_list: list):
        # This gets called when the camera updates its detections
        self.__lock.acquire()
        # TODO
        self.__lock.release()

    def update_from_action(self, action: int):
        # This is here in case we need to guess some observation values
        self.__lock.acquire()
        # TODO
        self.__lock.release()

    def get_model_obs(self):
        self.__lock.acquire()
        ret = copy.deepcopy(self.__state)
        self.__lock.release()
        return ret


class RLModel():
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
        self.observation = Observation()
        self.path_planner = path_planner.PathPlanner(
            robot_length=ROBOT_LENGTH/INCHES_PER_FIELD,
            robot_width=ROBOT_WIDTH/INCHES_PER_FIELD,
            buffer_radius=BUFFER_RADIUS/INCHES_PER_FIELD,
            max_velocity=80/INCHES_PER_FIELD,
            max_accel=100/INCHES_PER_FIELD)
        self.permanent_obstacles = [
            path_planner.Obstacle(3/6, 2/6, 3.5/INCHES_PER_FIELD, False), # Bottom
            path_planner.Obstacle(3/6, 4/6, 3.5/INCHES_PER_FIELD, False), # Top
            path_planner.Obstacle(2/6, 3/6, 3.5/INCHES_PER_FIELD, False), # Left
            path_planner.Obstacle(4/6, 3/6, 3.5/INCHES_PER_FIELD, False)] # Right

    def get_observation(self):
        return self.observation

    def predict(self):
        # model.predict returns tuple of (array(action_num), None)
        return int(self.model.predict(self.observation.get_model_obs())[0])

    def get_path(self, action):
        pass  # TODO

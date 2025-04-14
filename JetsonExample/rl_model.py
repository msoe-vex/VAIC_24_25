from stable_baselines3 import PPO

# Import scripts from submodule
import sys
sys.path.append('VEX-AI-Reinforcement-Learning')
import path_planner
import rl_environment

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

class RLModel():

    def __init__(self, model_path):
        self.model = PPO.load(model_path)
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

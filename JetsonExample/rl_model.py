import numpy as np
import copy
from threading import Lock
import torch, os, sys, time, math
from V5Position import RobotLocation
from V5Position import Position

# Import scripts from submodule
# sys.path.append(os.path.abspath("VEXAI"))
from VEXAI.pettingZooEnv import High_Stakes_Multi_Agent_Env
from VEXAI.pettingZooEnv import NUM_WALL_STAKES, NUM_GOALS, NUM_RINGS


class Observation:
    def __init__(self, robot_loc: RobotLocation):
        self.__state = np.zeros(
            2 + 1 + 1 + 1 + (NUM_RINGS * 2) + (NUM_GOALS * 2) + NUM_WALL_STAKES + 1 + 1 + 1 + 1,
            dtype=np.float32
        )
        self.__last_reset = time.time()
        self.__begin_time = 60
        self.__loc = robot_loc
        self.__lock = Lock()

    def begin_auton(self, begin_time=60):
        self.__lock.acquire()
        self.__last_reset = time.time()
        self.__begin_time = begin_time
        self.__state[3] = 0  # Holding Goal
        self.__lock.release()

    def update_robot_pos(self):
        new_pos = self.__loc.get_pos_for_rl(RobotLocation.FIELD_RL)

        self.__lock.acquire()

        self.__state[0] = new_pos.x
        self.__state[1] = new_pos.y
        self.__state[2] = new_pos.azimuth

        self.__lock.release()

    def update_from_camera(self, obj_list: list):
        # This gets called when the camera updates its detections
        self.__lock.acquire()

        ring_idx = 0
        goal_idx = 0

        for obj in obj_list:
            # Scale camera points to match what RL model expects
            obj_pos = Position(0, 1, obj['x'], obj['y'], 0, 0, 0, 0)
            obj_pos_scaled = RobotLocation.convert_to(RobotLocation.FIELD_RL, 
                    RobotLocation.convert_from(RobotLocation.FIELD_GPS, obj_pos))
            x = obj_pos_scaled.x
            y = obj_pos_scaled.y

            # Place ring & goal coordinates in our observation
            if not math.isnan(x) and not math.isnan(y):
                if obj['type'] == 'goal' and goal_idx < NUM_GOALS:
                    self.__state[10 + NUM_RINGS * 2 + 2 * goal_idx] = x
                    self.__state[10 + NUM_RINGS * 2 + 2 * goal_idx + 1] = y
                    goal_idx += 1
                elif obj['type'] == 'red_ring' and ring_idx < NUM_RINGS:
                    self.__state[4 + 2 * ring_idx] = x
                    self.__state[4 + 2 * ring_idx + 1] = y
                    ring_idx += 1

        # Fill the rest of rings and goals with -1
        if goal_idx < NUM_GOALS:
            self.__state[10 + NUM_RINGS * 2 + 2 * goal_idx:] = -1
        if ring_idx < NUM_RINGS:
            self.__state[4 + 2 * ring_idx:] = -1

        # Set ring and goal counts
        self.__state[-2] = ring_idx
        self.__state[-1] = goal_idx

        self.__lock.release()

    def update_from_action(self, action: str):
        # This is here in case we need to guess some observation values
        self.__lock.acquire()
        if action == 'PICKUP_GOAL':
            self.__state[3] = 1
        elif action == 'DROP_GOAL':
            self.__state[3] = 0
        self.__lock.release()

    def update_time_remaining(self):
        self.__lock.acquire()
        self.__state[-4] = max(self.__begin_time - (time.time() - self.__last_reset), 0)
        self.__lock.release()

    def get_model_obs(self):
        self.__lock.acquire()
        ret = self.__state.copy()
        self.__lock.release()
        return ret


class RLModel():
    def __init__(self, model_path, robot_loc: RobotLocation):
        self.observation = Observation(robot_loc)
        self.env = High_Stakes_Multi_Agent_Env()
        self.last_action = None

        # Load the TorchScript model
        try:
            self.model = torch.jit.load(model_path)
            self.model.eval()  # Set the model to evaluation mode
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def get_observation(self):
        return self.observation

    def predict(self):
        # Get physical environment state
        obs = self.observation.get_model_obs()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_logits = self.model(obs_tensor)
        # Sort actions by descending logit value (best first)
        sorted_actions = torch.argsort(action_logits, dim=1, descending=True).squeeze(0).tolist()
        # Find the first valid action
        for candidate_action in sorted_actions:
            if self.env.is_valid_action(candidate_action, obs, self.last_action):
                action = candidate_action
                break
        else:
            # Fallback: if no valid action found, pick top choice
            action = torch.argmax(action_logits, dim=1).item()

        # Do path planning and convert to lower-level actions
        action_list = self.env.break_down_action(action, obs)

        # Output: Action number, list of actions for the robot to take next
        return action, action_list

if __name__ == "__main__":
    # Example usage
    model_path = os.path.join(os.path.dirname(__file__), "model.pt")
    rl_model = RLModel(model_path)

    # Simulate getting observations and making predictions
    rl_model.observation.update_from_brain("0.5 0.5 45")
    rl_model.observation.update_from_camera([{"x": 1, "y": 2, "type": "goal"}])
    rl_model.observation.update_from_action("PICKUP_GOAL")
    action, action_list = rl_model.predict()
    print(f"Predicted action: {action}, Action list: {action_list}")
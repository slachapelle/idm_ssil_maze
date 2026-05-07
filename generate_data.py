from mazelab.generators import random_maze
from mazelab.solvers import get_policy
import numpy as np
from mazelab import BaseMaze, Object, BaseEnv, VonNeumannMotion
from mazelab import DeepMindColor as color
import gym
from gym.spaces import Box
from gym.spaces import Discrete
import os
import glob
import humanize
import pickle
import matplotlib.pyplot as plt

class Maze(BaseMaze):
    def __init__(self, maze_shape=(81,51)):
        self.x = random_maze(width=maze_shape[0], height=maze_shape[1], complexity=.75, density=.75)  # seeded, should always be the same maze
        #self.x = random_maze(width=maze_shape[0], height=maze_shape[1], complexity=.00001, density=.9)  # seeded, should always be the same maze
        super().__init__()
        
        
    @property
    def size(self):
        return self.x.shape
    
    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.x == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])
        goal = Object('goal', 3, color.goal, False, [])
        return free, obstacle, agent, goal


class Env(BaseEnv):
    def __init__(self, maze_shape=(8, 5)):
        super().__init__()
        print("env maze_shape", maze_shape)
        self.maze = Maze(maze_shape=maze_shape)
        self.motions = VonNeumannMotion()
        
        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))
        
    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]
        
        if self._is_goal(new_position):
            reward = +1
            done = True
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False
        return self.maze.to_value(), reward, done, {}
        
    def reset(self, start=[1,1], goal=None):
        if goal is None:
            goal = [self.maze.x.shape[0] - 2, self.maze.x.shape[1] - 2]
        self.maze.objects.agent.positions = [start]
        self.maze.objects.goal.positions = [goal]
        return self.maze.to_value()
    
    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable
    
    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out
    
    def get_image(self):
        return self.maze.to_rgb()


def save_array(filename, my_array):
    # .npy
    np.save(f"{filename}.npy", my_array)
    print(f"Saved {filename + '.npy'} ({my_array.dtype}): {humanize.naturalsize(os.path.getsize(f'{filename}.npy'))}")


def create_maze_data(maze_shape, data_dir="./data"):
    n_goals = 10
    n_starts = 1
    data_folder = os.path.join(data_dir, f"data_{maze_shape[0]}x{maze_shape[1]}")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # create environment
    print("Initializing environment...")
    env_id = 'RandomMaze-v0'
    if env_id not in gym.envs.registry.env_specs:
        gym.envs.register(id=env_id, entry_point=Env, max_episode_steps=200, kwargs={"maze_shape": maze_shape})
    env = gym.make(env_id)
    env.reset()
    impassable_array = env.unwrapped.maze.to_impassable()
    motions = env.unwrapped.motions
    motion2idx = {}
    idx2motion = {}
    for i, motion in enumerate(motions):
        motion2idx[tuple(motion)] = i
        idx2motion[i] = tuple(motion)

    # create and save idm
    print("Building IDM...")
    print(impassable_array.shape)
    idm = np.zeros(maze_shape + maze_shape + (2,), dtype=np.int32)
    idm_idx = -np.ones(maze_shape + maze_shape, dtype=np.int32)
    for x1 in range(maze_shape[0]):
        for y1 in range(maze_shape[1]):
            for action in motions:
                x2 = x1 + action[0]
                y2 = y1 + action[1]
                if 0 <= x2 < maze_shape[0] and 0 <= y2 < maze_shape[1]:
                    idm[x1, y1, x2, y2, :] = action
                    idm_idx[x1, y1, x2, y2] = motion2idx[tuple(action)]
                        
    print("Saving IDM...")
    save_array(os.path.join(data_folder, "idm"), idm)
    save_array(os.path.join(data_folder, "idm_idx"), idm_idx)

    print("Generating data...")
    for g in range(n_goals):
        # sample goal
        goal = list(env.maze.objects.free.positions[np.random.randint(0, len(env.maze.objects.free.positions))])
        # compute policy function
        policy, policy_idx = get_policy(impassable_array, motions, goal)

        # save policy & goal
        save_array(os.path.join(data_folder, f"g{g}_policy"), policy)
        save_array(os.path.join(data_folder, f"g{g}_policy_idx"), policy_idx)
        with open(os.path.join(data_folder, f"g{g}_goal.pkl"), "wb") as f:
            pickle.dump(goal, f)

        for s in range(n_starts):
            # sample starting position
            start = list(env.maze.objects.free.positions[np.random.randint(0, len(env.maze.objects.free.positions))])

            # for monitoring
            env = gym.wrappers.Monitor(env, data_folder, force=True)

            # initialize environment
            env.reset(start=start, goal=goal)
            done = False
            state = start

            # run policy
            states = [state]
            actions = []
            while not done:
                action = policy_idx[state[0], state[1]]
                _, reward, done, _ = env.step(action)
                state = env.unwrapped.maze.objects.agent.positions[0]
                states.append(state)
                actions.append(action)
            env.close()

            # Find the generated mp4 file and rename it
            mp4_files = glob.glob(os.path.join(data_folder, "openai*.mp4"))
            if mp4_files:
                new_name = os.path.join(data_folder, f"maze_g{g}_s{s}.mp4")
                os.rename(mp4_files[0], new_name)
            
            # save actions and states
            np.save(os.path.join(data_folder, f"maze_g{g}_s{s}_states.npy"), np.array(states))
            np.save(os.path.join(data_folder, f"maze_g{g}_s{s}_actions.npy"), np.array(actions))
    
    # delete other annoying openai files
    files_to_delete = glob.glob(os.path.join(data_folder, "openai*"))
    for f in files_to_delete:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Could not delete {f}: {e}")


if __name__ == "__main__":
    maze_shapes = [(10, 10), (20, 20), (50, 50)]
    for maze_shape in maze_shapes:
        create_maze_data(maze_shape)


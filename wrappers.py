import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info
    

def apply_wrappers(env):
    env = SkipFrame(env, skip=4) # Num of frames to apply one action to
    env = ResizeObservation(env, shape=84) # Resize frame from 240x256 to 84x84
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True) # May need to change lz4_compress to False if issues arise
    return env

def make_env(env_name, render_mode='human'):
    """
    Creates and configures a Super Mario Bros environment with all necessary wrappers.
    
    Args:
        env_name (str): The name of the environment (e.g., 'SuperMarioBros-1-1-v0')
        render_mode (str): The rendering mode (e.g., 'human' or 'rgb')
    
    Returns:
        The wrapped environment
    """
    env = gym_super_mario_bros.make(env_name, render_mode=render_mode, apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)
    return env

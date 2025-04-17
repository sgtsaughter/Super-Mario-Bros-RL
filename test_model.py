import os
import time  # Added import for sleep function
from agent import Agent
from wrappers import make_env

# Constants
MODEL_PATH = os.path.join("models", "2025-04-14-17_14_10", "model_50000_iter.pt")
NUM_TEST_EPISODES = 100
FRAME_DELAY = 0.05  # Add a delay between frames (in seconds) - adjust as needed

# Initialize environment
env = make_env("SuperMarioBros-1-1-v0")

# Load the trained agent
agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
agent.load_model(MODEL_PATH)
agent.epsilon = 0.1  # Enable 10% exploration for testing - adjust as desired

# Track successful episodes
high_reward_count = 0

# Test the model
for episode in range(NUM_TEST_EPISODES):
    print(f"Testing Episode: {episode + 1}")
    done = False
    state, _ = env.reset()
    total_reward = 0

    while not done:
        action = agent.choose_action(state)  # Use the trained model to choose actions
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = next_state
        
        # Add delay to slow down gameplay
        time.sleep(FRAME_DELAY)

    print(f"Total reward for episode {episode + 1}: {total_reward}")
    
    # Track episodes with high rewards.  
    # For me, around 3000 was the average amount of rewards the agent received by the end of the level
    if total_reward >= 3000:
        high_reward_count += 1

# Print summary statistics
print(f"\n--- Test Results Summary ---")
print(f"Episodes with reward ≥ 3000: {high_reward_count}/{NUM_TEST_EPISODES} ({high_reward_count/NUM_TEST_EPISODES:.1%})")

env.close()
import gymnasium as gym
from q_agent import QLearningAgent
import matplotlib.pyplot as plt
from sarsa_agent import SARSAAgent
import numpy as np
# WRITE YOUR CODE HERE
def train_agent(agent, env, n_episodes, eval_interval=100):
    returns = []
    best_avg_return = -np.inf

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        n_steps = 0
        total_reward = 0
        
        while not done:        
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)            
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated

            n_steps += 1
            total_reward += reward           

        agent.decay_epsilon()
        returns.append(total_reward)  

        if episode >= eval_interval:
            avg_return = np.mean(returns[episode - eval_interval : episode])
            if avg_return > best_avg_return:
                best_avg_return = avg_return

        if episode % eval_interval == 0 and episode > 0:
            print(f'Episode {episode}: best average return = {best_avg_return}')
    return returns
# Define the hyperparameters
n_episodes = 50000      
learning_rate = 0.5
initial_epsilon = 1.0
final_epsilon = 0
epsilon_decay = (initial_epsilon - final_epsilon) / (n_episodes / 2)

# Create the environment and the agent
env = gym.make('Taxi-v3')

# agent = QLearningAgent(
#     env=env,
#     learning_rate=learning_rate,
#     initial_epsilon=initial_epsilon,
#     epsilon_decay=epsilon_decay,
#     final_epsilon=final_epsilon
# )
agent = SARSAAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=initial_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon
)

# Train the agent
returns = train_agent(agent, env, n_episodes)
# def plot_returns(returns, file_name):
#     plt.plot(np.arange(len(returns)), returns)
#     plt.title('Episode returns')
#     plt.xlabel('Episode')
#     plt.ylabel('Return')
#     plt.savefig(file_name)
#     plt.show()

# plot_returns(returns, file_name='q_learning_curve.png')

def show_policy(agent, env):
    agent.epsilon = 0    # No need to keep exploring
    obs, _ = env.reset()
    env.render()
    done = False

    while not done:        
        action = agent.get_action(obs)        
        next_obs, _, terminated, truncated, _ = env.step(action)
        env.render()
        done = terminated or truncated
        obs = next_obs

env = gym.make('Taxi-v3', render_mode='human')
show_policy(agent, env)

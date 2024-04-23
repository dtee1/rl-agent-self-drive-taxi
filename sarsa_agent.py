import gymnasium as gym
import numpy as np
from collections import defaultdict

class SARSAAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95
    ):
        """Initialize a SARSA agent with an empty dictionary of state-action values (q_values).
        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q values
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor 

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Initialize an empty dictionary of state-action values
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs) -> int:
        """An epsilon-greedy action selection. Returns the best action with probability (1 - epsilon),
        otherwise a random action with probability epsilon to ensure exploration.
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_values[obs])
        
    def update(self, obs, action, reward, terminated, next_obs):
        """Update the Q-value of the chosen action.
        Args:
            obs: The current observation
            action: The action chosen by the agent
            reward: The reward received for taking that action
            terminated: Whether the episode has terminated after taking that action
            next_obs: The observation received from the environment after taking that action
        """
        next_action = self.get_action(next_obs)
        future_q_value = (not terminated) * self.q_values[next_obs][next_action]
        td = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        self.q_values[obs][action] += self.learning_rate * td




    def decay_epsilon(self):
        """Decrease the exploration rate epsilon until it reaches its final value"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
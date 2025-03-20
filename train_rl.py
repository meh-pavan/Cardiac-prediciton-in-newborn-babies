from stable_baselines3 import PPO
from modules.rl_module import CardiacAlertEnv

def train_rl_agent():
    # Create the custom environment
    env = CardiacAlertEnv()
    
    # Initialize the RL agent (using PPO with a multilayer perceptron policy)
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the agent for a set number of timesteps
    model.learn(total_timesteps=10000)
    
    # Save the trained RL agent model
    model.save("models/rl_agent")
    print("RL agent trained and saved as models/rl_agent.zip.")

if __name__ == "__main__":
    train_rl_agent()

import argparse

import gymnasium as gym
import torch
from tqdm import tqdm

import config
from utils import preprocess, grayscale
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1', 'MountainCar-v0', 'Pong-v5'], default='CartPole-v1')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--using_screen', type=bool, default=False, help='Are we using the screen as observation vector ?.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole,
    'MountainCar-v0': config.MountainCar,
    'Pong-v5': config.AtariPong,
}

def plot_learning(mean_perf, max_perf):
    eval_epochs = len(mean_perf)
    epochs = (np.arange(eval_epochs) * args.evaluate_freq)+1

    plt.plot(epochs, mean_perf, label="Mean returns")
    plt.plot(epochs, max_perf, label ="Max returns")

    plt.title("Performance over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Return")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    
    # Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    # Keep track of performances.
    mean_performances = np.empty(env_config['n_episodes'] // args.evaluate_freq)
    max_performances = np.empty(env_config['n_episodes'] // args.evaluate_freq)

    for episode in tqdm(range(env_config['n_episodes'])):
        terminated = False
        obs, info = env.reset()

        obs = preprocess(obs, env=args.env).unsqueeze(0)

        if args.using_screen:
            previous_screen = grayscale(env.render())
            current_screen = grayscale(env.render())

            obs = np.block([previous_screen, current_screen]).ravel()
        
        # initialize steps
        steps = 0
        
        while not terminated:
            # get action from dqn
            action = dqn.act(obs, exploit=False)

            # Act in the true environment.
            next_obs, reward, terminated, truncated, info = env.step(action.item())
                        
            # step counter
            steps += 1
            
            # Preprocess incoming observation.
            if not terminated:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)

                if args.using_screen:
                    previous_screen = current_screen
                    current_screen = grayscale(env.render())

                    next_obs = np.block([previous_screen, current_screen]).ravel()

            else:
                next_obs = None
                
            # Store transition in replay memory and ensure type torch.Tensor.
            memory.push(obs, action, next_obs, torch.tensor([reward], device=device))
            
            # Update observation.
            obs = next_obs
            
            # Optimize the DQN.
            if steps % env_config['train_frequency'] == 0:
                optimize(dqn, target_dqn, memory, optimizer)
                
            # Update target network.
            if steps % env_config['target_update_frequency'] == 0:
                target_dqn.load_state_dict(dqn.state_dict())
            
        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return, max_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')

            mean_performances[episode // args.evaluate_freq] = mean_return
            max_performances[episode // args.evaluate_freq] = max_return

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
        
    # Close environment after training is completed.
    env.close()
    plot_learning(mean_performances, max_performances)

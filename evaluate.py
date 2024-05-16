import argparse

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from tqdm import tqdm
import torch

import config
from utils import preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1', 'MountainCar-v0', 'Pong-v5', 'Breakout-v5'], default='CartPole-v1')
parser.add_argument('--path', type=str, help='Path to stored DQN model.')
parser.add_argument('--n_eval_episodes', type=int, default=1, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--render', dest='render', action='store_true', help='Render the environment.')
parser.add_argument('--save_video', dest='save_video', action='store_true', help='Save the episodes as video.')
parser.set_defaults(render=False)
parser.set_defaults(save_video=False)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole,
    'MountainCar-v0': config.MountainCar,
    'Pong-v5': config.AtariPong,
    'Breakout-v5': config.AtariBreakout,
}


def evaluate_policy(dqn, env, env_config, args, n_episodes, render=False, verbose=False):
    """Runs {n_episodes} episodes to evaluate current policy."""
    total_return = 0

    returns = [0] * n_episodes

    for i in range(n_episodes):
        obs, info = env.reset()
        obs = preprocess(obs, env=args.env).unsqueeze(0)

        terminated = False
        episode_return = 0

        while not terminated:
            if render:
                env.render()

            if args.env in ['Pong-v5', 'Breakout-v5']:
                action = dqn.act(obs, exploit=True)
                if action.item() != 0:
                    action_mapped = torch.tensor([[1 + action.item()]], dtype = torch.long)
                else:
                    action_mapped = action
                obs, reward, terminated, truncated, info = env.step(action_mapped.item())
            else:
                action = dqn.act(obs, exploit=True).item()
                obs, reward, terminated, truncated, info = env.step(action)

            # Preprocess incoming observation.
            if not terminated:
                obs = preprocess(obs, env=args.env).unsqueeze(0)
            else:
                obs = None

            episode_return += reward

        total_return += episode_return
        returns[i] = episode_return

        if verbose:
            print(f'Finished episode {i+1} with a total return of {episode_return}')


    return total_return / n_episodes, max(returns)

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env_config = ENV_CONFIGS[args.env]

    if args.env in ['Pong-v5', 'Breakout-v5']:
        env = gym.make('ALE/' + args.env, full_action_space=False, render_mode='rgb_array') # Already has a frameskip of 4
        env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
        env = gym.wrappers.FrameStack(env, num_stack=env_config['observation_stack_size'])
    else:
        env = gym.make(args.env)

    if args.save_video:
        # env = gym.make(args.env, render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(env, './video/', episode_trigger=lambda episode_id: True)

    # Load model from provided path.
    dqn = torch.load(args.path, map_location=torch.device('cuda'))
    dqn.eval()

    mean_return = evaluate_policy(dqn, env, env_config, args, args.n_eval_episodes, render=args.render and not args.save_video, verbose=True)
    print(f'The policy got a mean return of {mean_return} over {args.n_eval_episodes} episodes.')

    env.close()

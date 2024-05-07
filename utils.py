import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v1']:
        return torch.tensor(obs, device=device).float()
    elif env in ['MountainCar-v0']:
        return torch.tensor(obs, device=device).float()
    elif env in ['Pong-v5']:
        # rescale the observations from 0-255 to 0-1
        obs = obs / 255
        # convert to grayscale
        obs = grayscale(obs)
        # convert to tensor
        return torch.tensor(obs, device=device).float()
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')
    

def grayscale(image):
    """Converts an image to gray scale"""
    return np.mean(image, 2)
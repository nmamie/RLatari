import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.n_episodes = env_config['n_episodes']
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.steps_annealed = 0
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # Calculate the current epsilon
        eps_curr = self.eps_end + (self.eps_start - self.eps_end) * (1 - min(1.0, self.steps_annealed / self.anneal_length))
        self.steps_annealed += 1

        # Disable epsilon-greedy for inference
        if exploit or random.random() > eps_curr:
            # Assume self(observation) returns a [batch_size, n_actions] tensor
            # containing the Q-values for the given observation.
            with torch.no_grad():
                action = self.forward(observation).max(1).indices.view(1, 1)
        else:
            # Apply exploration to the entire batch
            action = torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

        return action
    
class ConvDQN(nn.Module):
    
    def __init__(self, env_config):
        super(ConvDQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.n_episodes = env_config['n_episodes']
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.steps_annealed = 0
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        """Runs the forward pass of the ConvNet."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # Calculate the current epsilon
        eps_curr = self.eps_end + (self.eps_start - self.eps_end) * (1 - min(1.0, self.steps_annealed / self.anneal_length))
        self.steps_annealed += 1

        # Disable epsilon-greedy for inference
        if exploit or random.random() > eps_curr:
            # Assume self(observation) returns a [batch_size, n_actions] tensor
            # containing the Q-values for the given observation.
            with torch.no_grad():
                action = self.forward(observation).max(1).indices.view(1, 1)
        else:
            # Apply exploration to the entire batch
            action = torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
            
        # there are 6 actions in Pong, but only action 2 and 3 are valid
        # so we need to convert the action to 2 or 3
        action = torch.tensor([[2 + action.item()]], device=device, dtype=torch.long)

        return action
    

def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # Sample a batch from the replay memory and concatenate so that there are
    # four tensors in total: observations, actions, next observations and rewards.
    # Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    # Note that special care is needed for terminal transitions!
    sample = memory.sample(dqn.batch_size)

    # handle terminal transitions (none)
    obs = torch.cat([s for s in sample[0]], dim=0).to(device)
    actions = torch.cat(sample[1], dim=0).to(device)
    next_obs = torch.cat([s if s is not None else torch.zeros_like(obs[0]).reshape(1, -1) for s in sample[2]], dim=0).to(device)
    rewards = torch.cat(sample[3], dim=0).to(device)

    non_terminal_mask = torch.tensor([s is not None for s in sample[2]], device=device, dtype=torch.bool)
    terminal_mask = torch.tensor([s is None for s in sample[2]], device=device, dtype=torch.bool)


    # Compute the current estimates of the Q-values for each state-action
    # pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    # corresponding to the chosen actions.
    q_values = dqn(obs)
    q_values = torch.gather(q_values, 1, actions)

    # Compute the Q-value targets. Only do this for non-terminal transitions!
    target_q_values = torch.zeros(dqn.batch_size, device=device)
    target_q_values[non_terminal_mask] = rewards[non_terminal_mask] + target_dqn.gamma * target_dqn(next_obs[non_terminal_mask]).max(1).values
    target_q_values[terminal_mask] = rewards[terminal_mask]
    target_q_values = target_q_values.detach()
    # import pdb; pdb.set_trace()

    # Compute the loss using the mse loss function
    loss = F.mse_loss(q_values, target_q_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # # Clip the gradients to avoid exploding gradients
    # for param in dqn.parameters():
    #     param.grad.data.clamp_(-1, 1)

    optimizer.step()

    return loss.item()

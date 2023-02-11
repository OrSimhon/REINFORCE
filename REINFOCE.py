import numpy as np
import torch
import gym
import time
import itertools


class REINFORCE:
    def __init__(self, env, nn_class, **hyper_parameters):

        # look for a gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device.type} ")

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self._init_hyper_parameters(hyper_parameters)  # Initialize Hyper Parameters

        # Policy network initialize
        self.pi = nn_class(in_dim=self.obs_dim, out_dim=self.act_dim)
        self.optimizer = torch.optim.Adam(params=self.pi.parameters(), lr=self.lr)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            'Episode': 0,  # current episode
            'total_steps': 0,
            'rewards': [],  # episodic returns in batch
            'losses': [],  # losses of actor network in current iteration
            'total_time': 0  # time from start of the learning
        }

    def generate_episode(self):
        states, actions, rewards = [], [], []
        state = self.env.reset()

        for step in itertools.count():
            self.save_render_log(step)
            self.logger['total_steps'] += 1
            action = self.pi.sample_action(state)
            next_state, reward, done, info = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if done:
                break
        return np.array(states), np.array(actions), rewards

    def get_returns(self, rewards):
        Gt = 0
        returns = []
        for reward in reversed(rewards):
            Gt = reward + Gt * self.gamma
            returns.insert(0, Gt)
        return np.array(returns)

    def update_policy(self, states, actions, targets):
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

        actions_probs = self.pi(states)
        log_probs = torch.log(actions_probs)
        selected_log_probs = torch.gather(log_probs, dim=1, index=actions)
        pi_loss = -torch.mean(selected_log_probs * targets)

        self.optimizer.zero_grad()
        pi_loss.backward()
        self.optimizer.step()

        return pi_loss.detach().numpy()

    def train(self):
        for episode in range(self.max_episodes):
            states, actions, rewards = self.generate_episode()
            self.logger['rewards'].append(sum(rewards))
            self.logger['Episode'] = episode + 1

            returns = self.get_returns(rewards)
            pi_loss = self.update_policy(states, actions, targets=returns)
            self.logger['losses'].append(pi_loss)

            # print(f"Episode {episode + 1} | Reward: {self.logger['Episode'][-1]:04.2f} | Loss: {pi_loss:04.2f}")

    def save_render_log(self, step):
        if self.logger['Episode'] > 0 and (self.logger['Episode'] % self.save_every == 0):
            if step == 0:
                torch.save(self.pi.state_dict(), './SavedNets/' + str(self.env.unwrapped.spec.id) + '_REINFORCE.pth')
                self._log_summary()
            if self.render:
                self.env.render()
        elif self.render and (self.logger['Episode'] - 1) % self.save_every == 0:
            self.env.close()

    def _init_hyper_parameters(self, hyperparameters):
        self.lr = 0.0001
        self.gamma = 0.99
        self.max_episodes = 5_000
        self.save_every = 50

        # Miscellaneous parameters
        self.render = True
        self.seed = None

        # Change any default values to custom values for specified HP
        for param, val in hyperparameters.items():
            exec('self.' + param + '=' + str(val))

        # Sets the seed if specified
        if self.seed is not None:
            # validity check
            assert type(self.seed) == int

            # Set the seed
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            self.env.seed(self.seed)
            self.env.action_space.seed(self.seed)
            print("Successfully set seed to {}".format(self.seed))

    def _log_summary(self):
        """ Print to stdout what we have logged so far in the most recent batch """

        # Calculate logging values
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        self.logger['total_time'] += delta_t

        total_hours = self.logger['total_time'] // 3600
        total_minutes = self.logger['total_time'] // 60 - total_hours * 60

        episode = self.logger['Episode']

        avg_ep_rews = np.mean(self.logger['rewards'][-100:])
        avg_loss = np.mean(self.logger['losses'][-100:])

        # Print logging statements
        print(flush=True)
        print("-------------------- Episode #{} --------------------".format(episode), flush=True)
        print("Average Episodic Return: {:.3f}".format(avg_ep_rews), flush=True)
        print("Average Loss: {:.5f}".format(avg_loss), flush=True)
        print("Total learning time: Hours: {:.0f} | Minutes: {:.0f}".format(total_hours, total_minutes), flush=True)
        print("------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['rewards'] = []
        self.logger['losses'] = []
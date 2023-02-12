from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
#import d4rl
import gym
import time
import AWAC.core as core
from AWAC.utils.logx import EpochLogger
import torch.nn.functional as F
import os
import scipy
import warnings
from termcolor import colored
import pickle
from AWAC.parameters_awac import Parameters

device = torch.device("cpu")

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def retrieve(self,index):
        data = dict(obs=self.obs_buf[index],
                    obs2=self.obs2_buf[index],
                    act=self.act_buf[index],
                    rew=self.rew_buf[index],
                    done=self.done_buf[index])
        return data

    def sample_batch(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

class AWAC:

    def __init__(self, env_fn,observation_space, action_space,agent,number_of_agents):
        # -------------------- Definition of parameters used for AWAC -------------------------
        self.env_fn = env_fn
        self.agent = agent
        self.observation_space = observation_space
        self.action_space = action_space
        self.number_of_agents = number_of_agents
        self.ep_ret = None
        self.ep_len = None
        self.reset_flag = None

        [self.actor_critic,self.alpha,self.num_train_episodes, self.done , self.ac_kwargs, self.seed, self.steps_per_epoch, self.epochs, self.replay_size,
         self.gamma, self.polyak, self.lr, self.p_lr, self.alpha, self.batch_size, self.start_steps, self.update_every,
         self.update_after, self.num_test_episodes, self.max_ep_len, self.logger_kwargs, self.save_freq, self.algo] = Parameters()

        self.logger = EpochLogger(**self.logger_kwargs)
        self.logger.save_config(locals())

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # -------------Folder to save the Results-----------------
        self.current_folder = os.getcwd() + '\\AWAC\\RESULTS\\Agent=' + str(agent)
        if not os.path.isdir(self.current_folder):
            os.mkdir(self.current_folder)

        self.env, self.test_env = self.env_fn, self.env_fn
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = self.actor_critic(self.env.observation_space, self.env.action_space,
                               special_policy='awac', **self.ac_kwargs)
        self.ac_targ = self.actor_critic(self.env.observation_space, self.env.action_space,
                                    special_policy='awac', **self.ac_kwargs)
        self.ac_targ.load_state_dict(self.ac.state_dict())

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        print("here")
        # Experience buffer
        self.replay_buffer = [ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,size=self.replay_size)
                              for i in range(number_of_agents)]

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(
            core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.p_lr, weight_decay=1e-4)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)
        print("Running Offline RL algorithm: {}".format(self.algo))

    def populate_buffer(self,ii,file):
        ####for 'MountainCarContinuous-v0'
        '''for i in range(num_of_agents):
            self.replay_buffer[i].store([0.2,0.5],[0.4],0.24,[0.2,0.6],0.0)
            self.replay_buffer[i].store([0.25, 0.68], [0.29], 0.67, [0.32, 0.45], 0.0)'''
        '''self.replay_buffer[ii].store([0.2, 0.5], [0.4], 0.24, [0.2, 0.6], 0.0)
        self.replay_buffer[ii].store([0.42, 0.24], [0.29], 0.67, [0.32, 0.45], 0.0)
        self.replay_buffer[ii].store([0.67, 0.92], [0.76], 0.17, [0.75, 0.53], 0.0)
        self.replay_buffer[ii].store([0.15, 0.57], [0.2], 0.36, [0.87, 0.77], 0.0)'''
        dataset = pickle.load(open(file, 'rb'))

        for j in range(dataset[ii].size):
            self.replay_buffer[ii].store(dataset[ii].retrieve(j)['obs'], dataset[ii].retrieve(j)['act'],
                                        dataset[ii].retrieve(j)['rew'],dataset[ii].retrieve(j)['obs2'],
                                        float(dataset[ii].retrieve(j)['done']))

        print("replay_buffer of agent ",ii)
        print(self.replay_buffer[ii].retrieve(0))
        '''dataset = pickle.load(open('replay_buffer.pkl', 'rb'))

        for i in range(num_of_agents):
            #print(dataset[i].size)
            for j in range(dataset[i].size):
                self.replay_buffer[i].store(dataset[i].retrieve(j)['obs'], dataset[i].retrieve(j)['act'], dataset[i].retrieve(j)['rew'],
                                            dataset[i].retrieve(j)['obs2'],float(dataset[i].retrieve(j)['done']))'''

        print("Loaded dataset")

    '''def populate_replay_buffer(self, env_name):
        data_envs = {
            'HalfCheetah-v2': (
                "awac_data/hc_action_noise_15.npy",
                "awac_data/hc_off_policy_15_demos_100.npy"),
            'Ant-v2': (
                "awac_data/ant_action_noise_15.npy",
                "awac_data/ant_off_policy_15_demos_100.npy"),
            'Walker2d-v2': (
                "awac_data/walker_action_noise_15.npy",
                "awac_data/walker_off_policy_15_demos_100.npy"),
        }
        if env_name in data_envs:
            print('Loading saved data')
            for file in data_envs[env_name]:
                if not os.path.exists(file):
                    warnings.warn(colored('Offline data not found. Follow awac_data/instructions.txt to download. Running without offline data.', 'red'))
                    break
                data = np.load(file, allow_pickle=True)
                for demo in data:
                    for transition in list(zip(demo['observations'], demo['actions'], demo['rewards'],
                                               demo['next_observations'], demo['terminals'])):
                        self.replay_buffer.store(*transition)
        else:
            dataset = d4rl.qlearning_dataset(self.env)
            N = dataset['rewards'].shape[0]
            for i in range(N):
                self.replay_buffer.store(dataset['observations'][i], dataset['actions'][i],
                                         dataset['rewards'][i], dataset['next_observations'][i],
                                         float(dataset['terminals'][i]))
            print("Loaded dataset")'''

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']

        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        v_pi = torch.min(q1_pi, q2_pi)

        beta = 2
        q1_old_actions = self.ac.q1(o, data['act'])
        q2_old_actions = self.ac.q2(o, data['act'])
        q_old_actions = torch.min(q1_old_actions, q2_old_actions)

        adv_pi = q_old_actions - v_pi
        weights = F.softmax(adv_pi / beta, dim=0)
        policy_logpp = self.ac.pi.get_logprob(o, data['act'])
        loss_pi = (-policy_logpp * len(weights) * weights.detach()).mean()

        # Useful info for logging
        pi_info = dict(LogPi=policy_logpp.detach().numpy())

        return loss_pi, pi_info

    def update(self, data, update_timestep):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)  # Get unnormalized score

            # self.logger.store(TestEpRet=100*self.test_env.get_normalized_score(ep_ret), TestEpLen=ep_len)  # Get normalized score

    def run(self,agent,t, obs, a, r, o2, d):

        if t == 0:
            self.ep_ret = 0
            self.ep_len = 0
            self.reset_flag = False
            print(o2)
            #action = self.env.action_space.sample()
            action = self.get_action(o2, deterministic=False)
            print("actiooooon")
            print(action)
            self.num_train_episodes = 0
        else:
            if self.reset_flag:
                self.reset_flag = False
                if d and t > 0:
                    self.logger.store(ExplEpRet=self.ep_ret, ExplEpLen=self.ep_len)
                    self.ep_ret = 0
                    self.ep_len = 0
                    self.reset_flag = True
                    self.num_train_episodes += 1

            # Collect experience
            action = self.get_action(o2, deterministic=False)

            self.replay_buffer[agent].store(obs, a, r, o2, d)
            print(self.replay_buffer[agent].size)
            print("okokok ", self.replay_buffer[agent].size)
            #print(self.replay_buffer[agent].retrieve(t))

            # Update handling
            if t > self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    batch = self.replay_buffer[agent].sample_batch(self.batch_size)
                    self.update(data=batch, update_timestep=t)

            # End of epoch handling
            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()

                # Log info about epoch
                self.logger.log_tabular('Agent', self.agent)
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('TotalEnvInteracts', t)
                '''self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('TotalUpdates', t)
                self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                self.logger.log_tabular('LogPi', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)'''
                scipy.io.savemat(self.current_folder + '\Replay_buffer.mat', {'Replay_Buffer': self.replay_buffer[agent]}) #############
                torch.save(self.ac, self.current_folder + '\ actor' + str(epoch) + '.pt') #############
                #self.logger.log_tabular('Time', time.time() - start_time)
                self.logger.dump_tabular()

        return action, self.reset_flag



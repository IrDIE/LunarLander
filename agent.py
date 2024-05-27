import random
import time
import itertools
import torch.nn.functional as F
import numpy as np
import torch
import os
import torch.nn as nn
from loguru import logger

from torch.utils.tensorboard import SummaryWriter

from p_exp_replay import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
LOGG = True  # <------- change here
if LOGG: logger.add(f"{os.path.dirname(os.path.realpath(__file__))}/logs/log_{time.time()}.log")

LOAD_FROM = './saved_weights/per_3/'
HIDDEN_GRU = 48
LR_CRITIC = 2.5e-4 #3e-4
LR_ACTOR = 2.5e-5 #1e-4
BATCH_SIZE = 64 # 128 #32 # FROM REPLAY BUFFER
BUFFER_SIZE = 50_000
MIN_REPLAY_SIZE = 1000
NUM_ENVS = 1
LOGGING_INTERVAL = 50 # 150
TAU = 5e-3 # SOFT NET UPDATE
GAMMA = 0.99

ACTION_SIZE = 2
CLIP_ACTION = [-2, 2]

RANDOM = 0.12
seq_size = 5
TRAIN_RATE = 4

SAVE_WEIGHTS = './saved_weights/per_4/'
LOGG_TB_DIR = "./logs/per_4/"


def logg_hyperparams(tb_logger: SummaryWriter, info: dict):
    hyperparameters = {
        'HIDDEN_GRU': HIDDEN_GRU,
        'RANDOM': RANDOM,
        'LOAD_FROM': LOAD_FROM,
        'LR_CRITIC': LR_CRITIC,
        'LR_ACTOR': LR_ACTOR,
        'BATCH_SIZE': BATCH_SIZE,
        'BUFFER_SIZE': BUFFER_SIZE,
        'MIN_REPLAY_SIZE': MIN_REPLAY_SIZE,
        'TAU': TAU,
        'GAMMA': GAMMA,
        'TRAIN_RATE': TRAIN_RATE,
        'SAVE_WEIGHTS': SAVE_WEIGHTS,
        'LOGG_TB_DIR': LOGG_TB_DIR,

    }
    hyperparameters.update(info)
    hyp_str = '\n'.join(['%s =  %s\n' % (key, value) for (key, value) in hyperparameters.items()])
    tb_logger.add_text(text_string=hyp_str, tag='Hyperparameters')


class TimeDistributed(nn.Module):
    def __init__(self, layer, time_steps, **kwargs):
        super(TimeDistributed, self).__init__()
        self.layers = nn.ModuleList([layer(**kwargs) for _ in range(time_steps)])

    def forward(self, x):
        # batch_size, time_steps, C, H, W = x.size()
        time_steps = x.size()[1]
        output = torch.tensor([]).to(device)
        for i in range(time_steps):
            # output_t = self.layers[i](x[:, i,...])
            output_t = self.layers[i](x[:, i, ...])
            if type(output_t) == tuple:
                output_t = output_t[0].unsqueeze(1)  # do not need last hidden state
            else:
                output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        return output


class Actor(nn.Module):
    def __init__(self, input_size, seq_size, save_path, action_size, min_max_actions, velocity_size=None,
                 name=''):
        super(Actor, self).__init__()

        self.save_path = save_path
        self.name = name
        self.action_size = action_size
        self.min_max_actions = min_max_actions
        self.input_size = input_size
        self.velocity_size = velocity_size
        self.seq_size = seq_size

        self.gru_shaped = False

        self.get_shared_net()
        self.get_state_vel_process(vel=False)
        self.get_actor()

        self.init_weights()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR_ACTOR)
        self.to(device)

    def do_checkpoint(self, load=True, best=False, alias=''):
        signature = 'actor_best.pt' if best else 'actor_last.pt'  # save/load best or last
        signature = alias + signature
        if load:
            ckpt_info = torch.load(self.save_path + signature)
            logger.info(f'ckpt_info from path : {signature} \nloaded with keys = {ckpt_info.keys()}')
            self.load_state_dict(ckpt_info['model_state_dict'], strict=False)
            self.optimizer.load_state_dict(ckpt_info['optimizer_state_dict'])
        else:  # save
            ckpt_info = {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }
            torch.save(ckpt_info, self.save_path + signature)

    def load_checkpoint(self, path):
        ckpt_info = torch.load(path)
        logger.info(f'Actor ckpt_info loaded with keys = {ckpt_info.keys()}')
        self.load_state_dict(ckpt_info['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(ckpt_info['optimizer_state_dict'])

    def forward(self, state, velocity=None):
        # logger.info(f'actor got {state.shape}. {velocity.shape}. {state[0].device}, {velocity.device}')
        out_shared = self.shared_net(state)
        out_shared = self.gru_block(out_shared)[0][:, -1, :]  # get last timestamp output from GRU
        out_shared = self.process_state_block(out_shared)

        if self.velocity_size is not None:
            assert velocity is not None, "velocity must be passed"
            velocity_embed = self.process_vel(velocity)
            out_shared = out_shared + velocity_embed

        actions_pred = torch.tanh(self.actor(out_shared))
        # logger.info(f'actions_pred {actions_pred.shape}') # torch.Size([1, 3])
        # actions_pred = torch.clip(actions_pred, min=self.min_max_actions[0], max=self.min_max_actions[1])
        return actions_pred

    def init_weights(self):

        def _init_weights_uniform(block, value=None, gru=False, conv=False):
            if gru:  # no bias
                torch.nn.init.xavier_uniform_(block.weight_ih_l0)
                torch.nn.init.xavier_uniform_(block.weight_hh_l0)
            else:
                torch.nn.init.uniform_(block.weight.data, -value, value)
                if conv: torch.nn.init.uniform_(block.bias.data, -value, value)

        logger.info('\nstart actor initialisation weights\n')
        for named_p in self.named_modules():
            if isinstance(named_p[1], nn.Conv2d):
                size_w = named_p[1].weight.data.size()
                value = 1. / np.sqrt(size_w[1] * size_w[2] * size_w[3])
                _init_weights_uniform(block=named_p[1], value=value, conv=True)

            elif isinstance(named_p[1], nn.Linear):
                size_w = named_p[1].weight.data.size()
                value = 1. / np.sqrt(size_w[0]) if size_w[1] != self.action_size else 0.003
                _init_weights_uniform(block=named_p[1], value=value)

            elif isinstance(named_p[1], nn.GRU):
                _init_weights_uniform(block=named_p[1], gru=True)

    def get_shared_net(self):
        self.shared_net = nn.Sequential(

            TimeDistributed(nn.Linear, time_steps=self.seq_size, in_features=self.input_size, out_features=64),
            nn.ELU(),
            TimeDistributed(nn.Linear, time_steps=self.seq_size, in_features=64, out_features=64),
            nn.ELU(),
            TimeDistributed(nn.Linear, time_steps=self.seq_size, in_features=64, out_features=32),
            nn.ELU(),
            TimeDistributed(nn.Linear, time_steps=self.seq_size, in_features=32, out_features=32),
            nn.ELU(),
            TimeDistributed(nn.Linear, time_steps=self.seq_size, in_features=32, out_features=32),
            nn.ELU(),

        )
        if not self.gru_shaped:
            self.gru_input_size = self.shared_net.to(device)(
                torch.rand(2, 5, self.input_size, device=device)).size()[-1]

            logger.info(f'GRU input size defined as : {self.gru_input_size}')
        self.gru_block = nn.Sequential(
            nn.GRU(input_size=self.gru_input_size, hidden_size=HIDDEN_GRU, bias=False, batch_first=True)
        )

    def get_state_vel_process(self, vel=False):

        self.process_state_block = nn.Sequential(
            nn.LayerNorm(HIDDEN_GRU),
            nn.Tanh()

        )
        if vel:
            assert self.velocity_size is not None, "velocity_size must be defined"
            self.process_vel = nn.Sequential(
                nn.Linear(self.velocity_size, HIDDEN_GRU, bias=False),
                nn.LayerNorm(HIDDEN_GRU),
                nn.Tanh()
            )

    def get_actor(self):

        self.actor_backbone = nn.Sequential(
            nn.Linear(HIDDEN_GRU, 32, bias=False),
            nn.LayerNorm(32),
            nn.ELU(),

            nn.Linear(32, 32, bias=False),
            nn.LayerNorm(32),
            nn.ELU()
        )
        self.actor_head = nn.Linear(32, self.action_size, bias=False)

        self.actor = nn.Sequential(
            self.actor_backbone, self.actor_head
        )


class Critic(nn.Module):
    def __init__(self, input_size, seq_size, save_path, action_size, velocity_size=None, name=''):
        super(Critic, self).__init__()

        self.save_path = save_path
        self.action_size = action_size
        self.input_size = input_size
        self.velocity_size = velocity_size
        self.seq_size = seq_size
        self.gru_shaped = False
        self.name = name
        self.get_shared_net()
        self.get_state_vel_process(vel=False)

        self.get_critic()
        self.init_weights()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR_CRITIC)
        self.to(device)

    @logger.catch()
    def forward(self, state, velocity=None, action=None):
        out_shared = self.shared_net(state)
        out_shared = self.gru_block(out_shared)[0][:, -1, :]  # get last timestamp output from GRU
        out_shared = self.process_state_block(out_shared)

        if self.velocity_size is not None:
            assert velocity is not None, "velocity must be passed"
            velocity_embed = self.process_vel(velocity)
            out_shared = out_shared + velocity_embed

        critic_embed = self.critic_backbone(action)
        state_action = critic_embed + out_shared
        q_critic = self.critic_head(state_action)
        q_critic = self.critic_q(q_critic)

        return q_critic

    def get_shared_net(self):
        self.shared_net = nn.Sequential(

            TimeDistributed(nn.Linear, time_steps=self.seq_size, in_features=self.input_size, out_features=64),
            nn.ELU(),
            TimeDistributed(nn.Linear, time_steps=self.seq_size, in_features=64, out_features=64),
            nn.ELU(),
            TimeDistributed(nn.Linear, time_steps=self.seq_size, in_features=64, out_features=32),
            nn.ELU(),
            TimeDistributed(nn.Linear, time_steps=self.seq_size, in_features=32, out_features=32),
            nn.ELU(),
            TimeDistributed(nn.Linear, time_steps=self.seq_size, in_features=32, out_features=32),
            nn.ELU(),

        )
        if not self.gru_shaped:
            self.gru_input_size = self.shared_net.to(device)(
                torch.rand(2, 2, self.input_size,  device=device)).size()[-1]

            logger.info(f'GRU input size defined as : {self.gru_input_size}')
        self.gru_block = nn.Sequential(
            nn.GRU(input_size=self.gru_input_size, hidden_size=HIDDEN_GRU, bias=False, batch_first=True)
        )

    def get_state_vel_process(self, vel=False):

        self.process_state_block = nn.Sequential(
            nn.LayerNorm(HIDDEN_GRU),
            nn.Tanh()

        )
        if vel:
            assert self.velocity_size is not None, "velocity_size must be defined"
            self.process_vel = nn.Sequential(
                nn.Linear(self.velocity_size, HIDDEN_GRU, bias=False),
                nn.LayerNorm(HIDDEN_GRU),
                nn.Tanh()
            )

    def init_weights(self):

        def _init_weights_uniform(block, value=None, gru=False, conv=False):
            if gru:  # no bias
                torch.nn.init.xavier_uniform_(block.weight_ih_l0)
                torch.nn.init.xavier_uniform_(block.weight_hh_l0)
            else:
                torch.nn.init.uniform_(block.weight.data, -value, value)
                if conv: torch.nn.init.uniform_(block.bias.data, -value, value)

        logger.info('\nstart critic initialisation weights\n')
        for named_p in self.named_modules():
            if isinstance(named_p[1], nn.Conv2d):
                size_w = named_p[1].weight.data.size()
                value = 1. / np.sqrt(size_w[1] * size_w[2] * size_w[3])
                _init_weights_uniform(block=named_p[1], value=value, conv=True)

            elif isinstance(named_p[1], nn.Linear):
                size_w = named_p[1].weight.data.size()
                value = 1. / np.sqrt(size_w[0]) if size_w[1] != 1 else 0.003
                _init_weights_uniform(block=named_p[1], value=value)


            elif isinstance(named_p[1], nn.GRU):
                _init_weights_uniform(block=named_p[1], gru=True)

    def do_checkpoint(self, load=True, best=False, alias=''):
        signature = 'critic_best.pt' if best else 'critic_last.pt'  # save/load best or last
        signature = alias + signature  # target or not
        if load:
            ckpt_info = torch.load(self.save_path + signature)
            logger.info(f'ckpt_info from path : {signature} \nloaded with keys = {ckpt_info.keys()}')
            self.load_state_dict(ckpt_info['model_state_dict'], strict=False)
            self.optimizer.load_state_dict(ckpt_info['optimizer_state_dict'])
        else:  # save
            ckpt_info = {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }
            torch.save(ckpt_info, self.save_path + signature)

    def load_checkpoint(self, path):
        ckpt_info = torch.load(path)
        logger.info(f'Critic ckpt_info loaded with keys = {ckpt_info.keys()}')
        self.load_state_dict(ckpt_info['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(ckpt_info['optimizer_state_dict'])

    def get_critic(self):
        self.critic_backbone = nn.Sequential(

            nn.Linear(self.action_size, HIDDEN_GRU, bias=False),
            nn.LayerNorm(HIDDEN_GRU),
            nn.Tanh()
        )

        self.critic_head = nn.Sequential(
            nn.Linear(HIDDEN_GRU, 32, bias=False),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, 32, bias=False),
            nn.LayerNorm(32),
            nn.ELU()

        )
        self.critic_q = nn.Linear(32, 1)  # kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.07, theta=.1, dt=1e-3, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self.mu, self.sigma)


class DDPG_Agent(nn.Module):
    def __init__(self, save_path, tau, input_size, gamma, memory_size, batch_size, seq_size, action_size, \
                 min_max_actions, velocity_size, load_ckpt=None):
        super(DDPG_Agent, self).__init__()
        self.save_path = save_path
        self.tau = tau
        self.gamma = gamma
        self.memory_size = memory_size
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.state_size = [input_size]
        self.velocity_size = velocity_size
        self.action_size = action_size

        self.actor = Actor(input_size, seq_size, save_path, action_size, min_max_actions, velocity_size)
        self.target_actor = Actor(input_size, seq_size, save_path, action_size, min_max_actions, velocity_size,
                                  name='target')
        self.critic = Critic(input_size, seq_size, save_path, action_size, velocity_size)
        self.target_critic = Critic(input_size, seq_size, save_path, action_size, velocity_size, name='critic')

        if load_ckpt is not None:
            self.actor.load_checkpoint(path=load_ckpt + 'actor_last.pt')
            self.critic.load_checkpoint(path=load_ckpt + 'critic_last.pt')
            logger.info(f'loaded chkpt from {load_ckpt}')

        self.update_network_parameters(tau=1, strict=False)
        self.noise = OUActionNoise(mu=np.zeros(action_size))
        self.replay_buffer_per = Memory(capacity=self.memory_size)

    def update_network_parameters(self, tau=None, strict=True):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_dict[name].clone()
        time.sleep(1)
        self.target_critic.load_state_dict(critic_state_dict, strict=strict)

        time.sleep(1)
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_dict[name].clone()
        time.sleep(1)
        self.target_actor.load_state_dict(actor_state_dict, strict=strict)

    def train(self):
        batch, idxs, _ = self.replay_buffer_per.sample(self.batch_size)

        images = torch.zeros([self.batch_size] + [self.seq_size] + self.state_size, device=device)
        actions = torch.zeros([self.batch_size, self.action_size], device=device)
        rewards = torch.zeros([self.batch_size, 1], device=device)
        next_images = torch.zeros([self.batch_size] + [self.seq_size] + self.state_size, device=device)

        dones = torch.zeros([self.batch_size, 1], device=device)

        # targets = torch.zeros([self.batch_size, 1], device=device)

        for i, sample in enumerate(batch):
            # logger.info(f'\ntraining agent. sample type = {sample[1]}, {sample[2]}, {sample[4]}')
            images[i] =torch.as_tensor(sample[0], device=device, dtype=torch.float32)
            actions[i] = torch.as_tensor(sample[1], device=device, dtype=torch.float32)
            rewards[i] = torch.as_tensor(sample[2], device=device, dtype=torch.float32)
            next_images[i]= torch.as_tensor(sample[3], device=device, dtype=torch.float32)
            dones[i] = torch.as_tensor(sample[4], device=device, dtype=torch.float32)

        # torch.backends.cudnn.enabled = False
        actions_clean = self.actor.forward(state=images, velocity=None)
        target_actions = self.target_actor.forward(state=next_images, velocity=None)
        target_next_q_s = self.target_critic.forward(state=next_images, velocity=None, action=target_actions)
        targets = rewards + self.gamma * (1 - dones) * target_next_q_s
        targets = targets.view(self.batch_size, 1)

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(state=images, velocity=None, action=actions_clean)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        critic_value = self.critic.forward(state=images, velocity=None, action=actions)  # pred
        td_error = torch.abs(critic_value - targets)  # error
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(targets, critic_value)  # loss
        critic_loss.backward()
        self.critic.optimizer.step()

        for i in range(self.batch_size):
            idx = idxs[i]
            self.replay_buffer_per.update(idx, td_error[i].detach().cpu().numpy()[0])
            # logger.info(f'td_error[{i}] in for_ loop in train = {td_error[i].detach().cpu().numpy()[0]}')

        self.update_network_parameters(strict=False)
        return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()

    def append_replay_buffer_per(self, state, action, reward, new_state, done):
        state_tensor = torch.as_tensor(state, device=device, dtype=torch.float32)
        new_state_tensor = torch.as_tensor(new_state, device=device, dtype=torch.float32)

        action_tensor = torch.as_tensor(action, device=device, dtype=torch.float32)
        done_tensor = torch.as_tensor([done], device=device, dtype=torch.float32)
        reward_tensor = torch.as_tensor([reward], device=device, dtype=torch.float32)
        # logger.info(f'*-*, {state_tensor[0].shape}, {state_tensor[1].shape}')
        # logger.info(f'*-*, {action_tensor.shape}, done_tensor = {done_tensor} - {done_tensor.shape}, reward_tensor={reward_tensor},{reward_tensor.shape}')
        critic_value = self.critic.forward(state=state_tensor, velocity=None, action=action_tensor)

        target_actions = self.target_actor.forward(state=new_state_tensor, velocity=None)
        target_next_q_s = self.target_critic.forward(state=new_state_tensor, velocity=None,
                                                     action=target_actions)
        # logger.info(f'**, {target_next_q_s.shape}, {done_tensor.shape}, {reward_tensor.shape}')

        targets = reward_tensor + self.gamma * (1 - done_tensor) * target_next_q_s
        targets = targets.view(1, 1)

        td = torch.abs(targets - critic_value)
        # logger.info(f'td in append memory = {td}')
        # logger.info(f'td[0] in append memory = {td[0].detach().cpu().numpy()[0]}')
        self.replay_buffer_per.add(td[0].detach().cpu().numpy(), (state, action, reward, new_state, done))
        return td.detach().cpu().numpy()

    def take_action(self, observation):

        actions = self.actor.forward(observation)
        actions = actions + torch.tensor(self.noise(),
                                         dtype=torch.float32).to(device)
        actions = torch.clip(actions, self.actor.min_max_actions[0], self.actor.min_max_actions[1])
        # self.actor.train()
        return actions.cpu().detach().numpy()

# torch.as_tensor(observation[0], device=device, dtype=torch.float32).permute(0, 1, 4, 2, 3)


def inference(env):
    rddpg_agent = DDPG_Agent(save_path=SAVE_WEIGHTS, tau=TAU, input_size=8, gamma=GAMMA, memory_size=BUFFER_SIZE, \
                             batch_size=BATCH_SIZE, seq_size=seq_size, action_size=ACTION_SIZE, \
                             min_max_actions=CLIP_ACTION, velocity_size=None, load_ckpt=LOAD_FROM).to(device)
    rddpg_agent.actor.eval()
    rddpg_agent.critic.eval()
    rddpg_agent.noise = OUActionNoise(mu=np.zeros(rddpg_agent.action_size), sigma=0, theta=0, dt=0)
    for dones in range(10):

        observation, info = env.reset()
        done = False
        history = np.stack([observation[np.newaxis, ...]] * seq_size, axis=1)  # Batch-sequence-h-w-channels
        state = history

        while not done:
            state_tensor = torch.as_tensor(state, device=device, dtype=torch.float32)

            action = rddpg_agent.take_action(state_tensor)
            env.render()
            observe, reward, terminated, truncated, info = env.step(action[0])  # step from noisy action / 1d list
            print(f'reward={reward}')
            time.sleep(0.001)

            # append to buffer
            history = np.append(history[:, 1:], observe[np.newaxis, np.newaxis, ...], axis=1)
            #  vel = vel.reshape(1, -1)
            new_state = history
            done = terminated or truncated
            state = new_state




def training_ddpg_per(env, seq_size):
    logger.info(f'---------- device  =  {device} ')
    tb_summary = SummaryWriter(LOGG_TB_DIR)
    logg_hyperparams(tb_summary, {"bug": "fix not doing reset after done in min_replay before training"})
    rddpg_agent =DDPG_Agent(save_path=SAVE_WEIGHTS, tau=TAU, input_size = 8, gamma=GAMMA, memory_size=BUFFER_SIZE, \
                             batch_size = BATCH_SIZE, seq_size = seq_size, action_size=ACTION_SIZE,\
                             min_max_actions = CLIP_ACTION , velocity_size = None, load_ckpt = LOAD_FROM).to(device)

    global_step = 0

    while global_step < MIN_REPLAY_SIZE:
        observation, info = env.reset()
        done = False
        history = np.stack([observation[np.newaxis, ...]] * seq_size, axis=1)  # Batch-sequence-h-w-channels
        state = history
        logger.info(f'reset.')

        logger.info(f'COLLECTING REPLAY BUFFER at step = {global_step}')


        while not done:
            state_tensor = torch.as_tensor(state, device=device, dtype=torch.float32)
            random = (np.random.random() < RANDOM)

            if random:
                action = np.array(np.random.uniform(-1, 1, (1, ACTION_SIZE)))
            else:
                action = rddpg_agent.take_action(state_tensor)  # with noise

            observe, reward, terminated, truncated, info = env.step(action[0])  # step from noisy action / 1d list

            # append to buffer
            history = np.append(history[:, 1:], observe[np.newaxis, np.newaxis, ...], axis=1)
            #  vel = vel.reshape(1, -1)
            new_state = history
            done = terminated or truncated
            # logger.info(f'do append')
            rddpg_agent.append_replay_buffer_per(state, action, reward, new_state, done)
            state = new_state
        global_step += 1



    # main training:
    logger.info(f'\n****\nstart main training ...\n')
    global_step = 0
    actor_sum_loss_episodes, critic_sum_loss_episodes, sum_reward_episodes, tds_avg = 0, 0, 0, 0
    time_limit = 600
    for episode in itertools.count():
        actor_sum_loss_per_ep, critic_sum_loss_per_ep, sum_reward_per_episode, tds_per_ep = 0, 0, 0, 0
        observation, info = env.reset()
        logger.info(f'do reset. done={done}')
        history = np.stack([observation[np.newaxis, ...]] * seq_size, axis=1)  # Batch-sequence-h-w-channels
        state = history
        done = False
        step_epoch = 0


        while not done and step_epoch < time_limit:
            step_epoch += 1
            global_step += 1
            if global_step % TRAIN_RATE == 0:
                # do train
                critic_loss, actor_loss = rddpg_agent.train()
                print(f'critic_loss={critic_loss}, actor_loss = {actor_loss}')

                actor_sum_loss_per_ep += actor_loss.item()
                critic_sum_loss_per_ep += critic_loss.item()

            # get action from state
            state_tensor = torch.as_tensor(state, device=device, dtype=torch.float32)
            random = (np.random.random() < RANDOM)
            if random:
                action = np.array(np.random.uniform(-1, 1, (1, ACTION_SIZE)))
            else:
                action = rddpg_agent.take_action(state_tensor)  # with noise

            observe, reward, terminated, truncated, info = env.step(action[0])  # step from noisy action / 1d list


            tb_summary.add_scalar('reward_step_immediate', reward, global_step=episode)
            sum_reward_per_episode += reward
            # append to buffer
            history = np.append(history[:, 1:], observe[np.newaxis, np.newaxis, ...], axis=1)
            #  vel = vel.reshape(1, -1)
            new_state = history
            done = terminated or truncated
            td = rddpg_agent.append_replay_buffer_per(state, action, reward, new_state, done)
            tds_per_ep += td
            state = new_state


        # done
        sum_reward_episodes += sum_reward_per_episode
        tb_summary.add_scalar('sum_reward_per_episode', sum_reward_per_episode, global_step=episode)

        actor_sum_loss_episodes += actor_sum_loss_per_ep
        critic_sum_loss_episodes += critic_sum_loss_per_ep

        tds_avg += (tds_per_ep / step_epoch)

        # logg weights best / last

        logger.info(f'\nLogging....')
        avg_per_episode = sum_reward_episodes / (episode + 1)
        actor_loss_per_episode = actor_sum_loss_episodes / (episode + 1)
        critic_loss_per_episode = critic_sum_loss_episodes / (episode + 1)

        # do checkpoint for last
        rddpg_agent.actor.do_checkpoint(load=False, best=False, alias='')
        rddpg_agent.target_actor.do_checkpoint(load=False, best=False, alias='target')
        rddpg_agent.critic.do_checkpoint(load=False, best=False, alias='')
        rddpg_agent.target_critic.do_checkpoint(load=False, best=False, alias='target')

        tb_summary.add_scalar('avg_reward_current', avg_per_episode, global_step=episode)
        tb_summary.add_scalar('actor_loss_per_episode', actor_loss_per_episode, global_step=episode)
        tb_summary.add_scalar('critic_loss_per_episode', critic_loss_per_episode, global_step=episode)
        tb_summary.add_scalar('tds_avg pper step ', tds_avg / (episode + 1), global_step=episode)
















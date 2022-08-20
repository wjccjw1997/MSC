import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from wrappers import wrap_deepmind, make_atari, wrap_pytorch
from agent import Agent
from agent import DQN
from memory import *
import matplotlib.pyplot as plt

def get_env():
    env = make_atari("PongNoFrameskip-v4")
    #env = make_atari("BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    return env

def get_epsilon(frame_idx, eps_start=1, eps_final=0.1, eps_decay=40000):
    eps = eps_final + (eps_start - eps_final) * np.exp(-1 * frame_idx / eps_decay)
    return eps

def train(args, device, verbose=True):
    env = get_env()
    buffer = ReplayBuffer(args.capacity, args.batch_size)
    agent = Agent(args.algorithm, env, args.learning_rate, args.gamma, device)
    ep = 0
    num_loss = 0
    num_reward = 0
    S = env.reset()
    for idx in range(1, args.num_frames + 1):
        eps = get_epsilon(idx)
        var_S = Variable(torch.FloatTensor(S).unsqueeze(0)).to(device)
        A = agent.get_action(var_S, eps)
        S_, R, is_done, _ = env.step(A)
        buffer.push(S, A, R, S_, is_done)
        num_reward += R
        if is_done:
            S = env.reset()
            ep += 1
            if verbose:
                print("Episode: %3d - reward : %.2f" %(ep, num_reward))
            num_loss = 0
            num_reward = 0
        else:
            S = S_
        if len(buffer) > args.warm_up:
            S, A, R, S_, done = buffer.sample()
            S = Variable(torch.FloatTensor(S)).to(device)
            S_ = Variable(torch.FloatTensor(S_)).to(device)
            R = Variable(torch.FloatTensor(R)).to(device)
            done = Variable(torch.FloatTensor(done)).to(device)
            A = Variable(torch.LongTensor(A)).to(device)
            loss = agent.update(S, S_, R, done, A)
            num_loss += loss.item()
        if idx % args.target_update == 0:
            agent.backup()
    agent.save()

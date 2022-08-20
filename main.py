import torch
import argparse
from train import train

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str, default="DQN")
#parser.add_argument("--algorithm", type=str, default="doubleDQN")
parser.add_argument("--num_frames", type=int, default=700000)
parser.add_argument("--capacity", type=int, default=15000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--warm_up", type=int, default=10000)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--target_update", type=int, default=1000)
args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda")
    train(args, device)
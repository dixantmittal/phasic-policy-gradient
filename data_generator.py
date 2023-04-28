import argparse
import os
import random
from collections import defaultdict
from itertools import count

import gym
import torch as t
from tqdm import tqdm

import phasic_policy_gradient.torch_util as tu
from phasic_policy_gradient.tree_util import tree_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="coinrun",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=100_000,
    )
    args = parser.parse_args()

    env = gym.make(f"procgen:procgen-{args.env}-v0")
    model = t.load(f"runs/{args.env}/model.jd")

    os.makedirs(f"data/procgen/{args.env}/", exist_ok=True)
    data = defaultdict(list)
    batch = 0
    for i in count():
        observation = env.reset()
        terminal = False
        info = {"prev_level_complete": 0}
        episode_reward = 0
        episode = defaultdict(list)
        for _ in tqdm(count(), desc=f"Trajectory: {i:02d}"):
            observation = tree_map(tu.np2th, observation)[None]
            action = model.act(
                ob=observation,
                first=t.tensor([True]),
                state_in={"pi": [], "vf": []},
            )[0].item()

            episode["states"].append(observation.cpu())
            episode["actions"].append(action)

            observation, reward, terminal, info = env.step(action)
            episode["rewards"].append(reward)
            episode_reward += reward
            if terminal:
                break

        # Store only if the agent succeded
        if info["prev_level_complete"] == 1:
            episode["states"].append(tree_map(tu.np2th, observation)[None].cpu())
            episode["actions"].append(random.randint(0, env.action_space.n - 1))
            episode["rewards"].append(0)

            episode["values"] = [0] * len(episode["rewards"])
            episode["values"][-1] = episode["rewards"][-1]
            for i in reversed(range(len(episode["rewards"]) - 1)):
                episode["values"][i] = (
                    episode["rewards"][i] + 0.99 * episode["values"][i + 1]
                )

            data["states"].append(episode["states"])
            data["actions"].append(episode["actions"])
            data["rewards"].append(episode["rewards"])
            data["values"].append(episode["values"])

        if len(data["states"]) == args.batch_size:
            t.save(data, f"data/procgen/{args.env}/batch{batch:02d}.data")
            data = defaultdict(list)
            batch += 1

        if (batch + 1) * args.batch_size >= args.n_trajectories:
            break

    print("Done!")

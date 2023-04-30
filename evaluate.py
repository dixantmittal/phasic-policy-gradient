import argparse
from itertools import count

import gym
import numpy as np
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
    args = parser.parse_args()

    model = t.load(f"runs/{args.env}/model.jd")

    alive = list(range(1000))
    episode_rewards = [0] * len(alive)
    all_envs = [gym.make(f"procgen:procgen-{args.env}-v0") for env_id in alive]

    states = [all_envs[env_id].reset() for env_id in alive]
    for _ in tqdm(count()):
        states = t.stack(tree_map(tu.np2th, states))
        actions = model.act(states, first=t.tensor([True]),
                            state_in={"pi": [], "vf": []}, )[0]

        next_states, dead = {}, []
        for env_id, a in zip(alive, actions):
            next_state, reward, terminal, *_ = all_envs[env_id].step(int(a))
            episode_rewards[env_id] += reward

            if terminal:
                dead.append(env_id)
            else:
                # Do not append next state if terminal is reached
                next_states[env_id] = next_state

        for env_id in dead:
            alive.remove(env_id)

        if len(alive) == 0:
            break

        states = [next_states[env_id] for env_id in alive]

    print("Episode rewards:", np.mean(episode_rewards))

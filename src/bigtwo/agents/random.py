"""Uniform random over the legal moves (pass included when legal)."""

import random

from bigtwo.agents.base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs, legal):
        return legal[int(self.rng.random() * len(legal))]

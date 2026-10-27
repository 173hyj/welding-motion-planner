"""Exact finite source-state exclusions and finer legal opening proposals."""
from seed_selector_v79 import rank_seeds


def state_key(seed):
    return (seed['candidate']['robot_id'], seed['positioner_deg'], tuple(seed['q_rad']))


def unsearched_seeds(items, searched, limit):
    old = {state_key(seed) for seed in searched}
    return rank_seeds([seed for seed in items if state_key(seed) not in old], limit)


def opening_amplitudes():
    return (80., 60., 30., 10., 5., 20., 40., 50., 70.)

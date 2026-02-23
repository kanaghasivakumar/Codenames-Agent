"""
Bootstrap user profiles with simulated gameplay outcomes.
Run:
    python -m codenames_agent.bootstrap_profiles --user alice --games 200

This script initializes an agent with the given user id and runs a
number of simulated turns. For each spymaster clue, a simple simulated
operative will 'guess' some subset of intended targets according to a
configurable success probability. The script calls `agent.record_outcome`
to persist profile data for the user.
"""

import argparse
import random
import time
import os
import sys
from typing import List

# Fix import path — script lives inside codenames_agent/, so we add the project root
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from main_with_reasoning import CodenamesAgentWithReasoning


def simulate_game_turn(agent: CodenamesAgentWithReasoning, user_pref: dict):
    # Generate a random board and get a clue
    red, blue, assassin, neutral = agent.generate_board()
    result = agent.play_turn(red, blue, assassin, neutral)
    if result is None:
        return

    clue = result.get('clue')
    targets = result.get('targets', [])
    source = result.get('source', 'vector')

    # Decide success probability based on source and user_pref
    base_rate = user_pref.get('base_rate', 0.5)
    source_bias = user_pref.get('source_bias', {})
    rate = base_rate * source_bias.get(source, 1.0)

    guessed = []
    for t in targets:
        if random.random() < rate:
            guessed.append(t)

    # Occasionally guess opponent/neutral to create varied outcomes
    if random.random() < user_pref.get('mistake_rate', 0.05):
        # simulate hitting a neutral or opponent
        guessed = []

    # Record outcome to agent profile
    try:
        agent.record_outcome(clue, source, targets, guessed)
    except Exception as e:
        print('record_outcome error:', e)


def run_simulation(user_id: str, n_games: int, seed: int = None):
    if seed is not None:
        random.seed(seed)

    # create agent for this user
    agent = CodenamesAgentWithReasoning(user_id=user_id)

    # Example user behaviors; these could be randomized per user
    user_pref = {
        'base_rate': 0.5,
        'source_bias': {
            'vector': 1.0,
            'reasoning': 0.8
        },
        'mistake_rate': 0.06
    }

    print(f"Simulating {n_games} turns for user '{user_id}'...")
    for i in range(n_games):
        simulate_game_turn(agent, user_pref)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_games} simulated")
        time.sleep(0.01)

    print('Simulation complete. Profile saved to:', agent.profile_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', dest='user', default='anon')
    parser.add_argument('--games', dest='games', type=int, default=200)
    parser.add_argument('--seed', dest='seed', type=int, default=None)
    args = parser.parse_args()

    run_simulation(args.user, args.games, seed=args.seed)

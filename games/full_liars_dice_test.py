#Test openspiel vs jax implementation
import random
import jax 
import jax.numpy as jnp
from tqdm import tqdm
from games.liars_dice import LiarsDice
from games.full_liars_dice import FullLiarsDice, full_to_round, round_to_full

def unif_policy(legal, key):
    legal_logpi = jnp.where(legal, 1.0, -jnp.inf)
    #Sample action
    action = jax.random.categorical(key, legal_logpi, axis=-1)
    return action


def to_count(dice, num_dice=1, max_num_dice=5, dice_sides=6):
    count = jnp.zeros((2, dice_sides+1,), dtype=jnp.int8)
    for i in range(2):
        _dice = dice_sides * jnp.ones(max_num_dice, dtype=jnp.int8)
        _dice = _dice.at[:num_dice].set(dice[i,:])
        _dice = jnp.sort(_dice, axis=-1)
        __dice = jnp.zeros((dice_sides+1,), dtype=jnp.int8)
        _dice = __dice.at[_dice].add(1)
        count = count.at[i,:].set(_dice)
    return count


def test_diff(x,y):
    eps = 1e-8
    return jnp.abs(jnp.float32(x) - jnp.float32(y)).sum() < eps

def play_traj(key, num_dice=1, max_num_dice=5, dice_sides=6, test_full_to_round=True):
    '''Sample a trajectory
    '''
    game_round = LiarsDice(num_dice=num_dice, dice_sides=dice_sides)
    game_full = FullLiarsDice(max_num_dice=max_num_dice, dice_sides=dice_sides)

    key, _key = jax.random.split(key)
    state_round = game_round.init(_key)

    _state = game_full.full_init(key, num_dice=num_dice)
    _dice = to_count(state_round._dices, num_dice, max_num_dice, dice_sides)
    _state = _state.replace(_dices = _dice)
    state_full = _state.replace(observation=game_full.observe(_state, _state.current_player))

    while not state_round.terminated:
        #Get obs and legal
        obs_round = state_round.observation
        legal_round = state_round.legal_action_mask

        obs_full = state_full.observation
        legal_full = state_full.legal_action_mask
        if test_full_to_round:
            obs_full, legal_full = full_to_round(obs_full, legal_full)
        else:
            obs_round, legal_round = round_to_full(obs_round, legal_round, num_dice, max_num_dice, dice_sides)
        #Test obs
        assert test_diff(obs_round, obs_full), "obs mismatch"
        #Test legal
        assert test_diff(legal_round, legal_full), "legal mismatch"
        #Get action
        key, _key = jax.random.split(key)
        action = unif_policy(legal_round, _key)
        #Update states
        state_round = game_round.step(state_round, action)
        state_full = game_full.step(state_full, action)
        #Test reward
        assert test_diff(state_round.rewards, state_full.rewards), "reward mismatch"

def _test_full_liars_dice(N, seed=1, num_dice=1, max_num_dice=5, dice_sides=6, test_full_to_round=True):
    print(f'Test full vs one round Liars dice over {N} trajectories')
    print(f'with  {num_dice} dice and {max_num_dice} max number of dice with {dice_sides} sides')
    key = jax.random.PRNGKey(seed)
    for _ in tqdm(range(N)):
        key, _key = jax.random.split(key)
        play_traj(_key, num_dice, max_num_dice, dice_sides, test_full_to_round)
    print()
    print('test passed!')
    print()


def test_full_liars_dice():
    N = 200
    seed = 32
    _test_full_liars_dice(N, seed, num_dice=1, max_num_dice=5, dice_sides=6, test_full_to_round=False)
    _test_full_liars_dice(N, seed, num_dice=2, max_num_dice=5, dice_sides=6, test_full_to_round=True)
    _test_full_liars_dice(N, seed, num_dice=3, max_num_dice=5, dice_sides=6, test_full_to_round=False)
    _test_full_liars_dice(N, seed, num_dice=5, max_num_dice=5, dice_sides=6, test_full_to_round=True)

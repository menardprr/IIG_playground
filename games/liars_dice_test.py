#Test openspiel vs jax implementation
import pyspiel 
import random
import jax 
import jax.numpy as jnp
from tqdm import tqdm
from games.liars_dice import LiarsDice


seed = 123
_py_rng = random.Random(seed)

def _play_chance(state: pyspiel.State) -> pyspiel.State:
    while state.is_chance_node():
        chance_outcome, chance_proba = zip(*state.chance_outcomes())
        action = _py_rng.choices(chance_outcome, weights=chance_proba)[0]
        state.apply_action(action)
    return state



def play_traj_spiel(num_dice=1, dice_sides=6):
    """Sample a trajectory with OpenSpiel
    """
    rewards = []
    obs = []
    game = pyspiel.load_game(f'liars_dice(players=2,numdice={num_dice},dice_sides={dice_sides})')
    state = game.new_initial_state()
    state = _play_chance(state)
    dices = state.history()
    mid = len(dices)//2
    dices = [dices[:mid], dices[mid:]]
    dices = jnp.array(dices, dtype=jnp.int8)
    dices = jnp.sort(dices, axis=-1)
    actions = []
    legal = []
    while not state.is_terminal():
        obs.append(jnp.array(state.information_state_tensor()))
        legal_actions = state.legal_actions()
        legal.append(jnp.array(state.legal_actions_mask()))
        action = _py_rng.choices(legal_actions)[0] 
        actions.append(action)
        state.apply_action(action)
        state = _play_chance(state)
        rewards.append(jnp.array(state.returns(), dtype=jnp.float32))
    return dices, actions, legal, obs, rewards


def play_traj_jax(num_dice, dice_sides, dices, actions, sp_legal, sp_obs, sp_rewards):
    """Mirror trajectory from cards and action and check if
    the rewards and the obs are the same.
    """
    eps = 1e-8
    unused_key = jax.random.PRNGKey(seed)
    game = LiarsDice(num_dice=num_dice, dice_sides=dice_sides)
    state = game.init(unused_key)
    state = state.replace(_dices = dices)
    state = state.replace(observation = game.observe(state,  state.current_player))
    while not state.terminated:
        assert jnp.abs(state.observation - sp_obs.pop(0)).sum() < eps
        assert jnp.abs(jnp.float32(state.legal_action_mask) - jnp.float32(sp_legal.pop(0))).sum() < eps
        action = actions.pop(0)
        state = game.step(state, action)
        assert jnp.abs(state.rewards - sp_rewards.pop(0)).sum()  <eps

def _test_liars_dice(N, num_dice=1, dice_sides=6):
    print(f'Test openspiel vs jax Liars dice over {N} trajectories')
    print(f'with  {num_dice} dice(s) with {dice_sides} sides')
    for _ in tqdm(range(N)):
        dices, actions, sp_legal, sp_obs, sp_rewards = play_traj_spiel(num_dice, dice_sides)
        play_traj_jax(num_dice, dice_sides,dices, actions, sp_legal, sp_obs, sp_rewards)
    print()
    print('test passed!')

def test_liars_dice():
    N = 1_000
    _test_liars_dice(N, num_dice=1, dice_sides=6)
    print()
    _test_liars_dice(N, num_dice=3, dice_sides=6)

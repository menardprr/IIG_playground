#Test openspiel vs jax implementation
import pyspiel 
import random
import jax 
import jax.numpy as jnp
from tqdm import tqdm
from games.kuhn_poker import KuhnPoker

seed = 123
_py_rng = random.Random(seed)

def _play_chance(state: pyspiel.State) -> pyspiel.State:
    while state.is_chance_node():
        chance_outcome, chance_proba = zip(*state.chance_outcomes())
        action = _py_rng.choices(chance_outcome, weights=chance_proba)[0]
        state.apply_action(action)
    return state

def play_traj_spiel():
    """Sample a trajectory with OpenSpiel
    """
    rewards = []
    obs = []
    game = pyspiel.load_game('kuhn_poker')
    state = game.new_initial_state()
    state = _play_chance(state)
    cards = jnp.array(state.history(), dtype=jnp.int8)
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
    return cards, actions, legal, obs, rewards


def play_traj_jax(cards, actions, sp_legal, sp_obs, sp_rewards):
    """Mirror trajectory from cards and action and check if
    the rewards and the obs are the same.
    """
    eps = 1e-8
    unused_key = jax.random.PRNGKey(seed)
    game = KuhnPoker()
    state = game.init(unused_key)
    state = state.replace(_cards = cards)
    state = state.replace(observation = game.observe(state,  state.current_player))
    while not state.terminated:
        assert jnp.abs(state.observation - sp_obs.pop(0)).sum() < eps
        assert jnp.abs(jnp.float32(state.legal_action_mask) - jnp.float32(sp_legal.pop(0))).sum() < eps
        action = actions.pop(0)
        state = game.step(state, action)
        assert jnp.abs(state.rewards - sp_rewards.pop(0)).sum()  <eps

def test_kuhn_poker():
    N = 10_000
    print(f'Test openspiel vs jax Kuhn poker over {N} trajectories')
    for _ in tqdm(range(N)):
        cards, actions, sp_legal, sp_obs, sp_rewards = play_traj_spiel()
        play_traj_jax(cards, actions, sp_legal, sp_obs, sp_rewards)
    print()
    print('test passed!')
import games.game as ggame

import jax
import jax.numpy as jnp
import chex


FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

J_2 = jnp.int8(2)
J_3 = jnp.int8(3)


PASS = jnp.int8(0)
BET = jnp.int8(1)

_NUM_PLAYER = 2
_MAX_GAME_LENGTH = 3
_MAX_UTILITY = 2.0
_MIN_UTILITY = - 2.0


@chex.dataclass(frozen=True)
class State(ggame.State):
    current_player: chex.ArrayDevice = jnp.int8(0)
    observation: chex.ArrayDevice = jnp.zeros(11, dtype=jnp.bool_)
    rewards: chex.ArrayDevice = jnp.float32([0.0, 0.0])
    terminated: chex.ArrayDevice = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: chex.ArrayDevice = jnp.ones(2, dtype=jnp.bool_)
    _rng_key: chex.PRNGKey = jax.random.PRNGKey(0)
    _step_count:  chex.ArrayDevice = jnp.int8(0)
    # --- Kuhn poker specific ---
    _cards:  chex.ArrayDevice = jnp.int8([-1, -1])
    # [(player 0),(player 1)]
    _bets:  chex.ArrayDevice = jnp.zeros(6, dtype=jnp.bool_)
    # 0(Call)  1(Bet)  2(Fold)  3(Check)
    _pot:  chex.ArrayDevice = jnp.int8([1, 1])

    @property
    def env_id(self) -> str:
        return "kuhn_poker"


class KuhnPoker(ggame.Game):
    def __init__(self):
        super().__init__()

    def _init(self, key: jax.random.KeyArray) -> State:
        return _init(key)

    def _step(self, state: State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return _step(state, action)

    def _observe(self, state: State, player_id: jnp.ndarray) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> str:
        return "kuhn_poker"

    @property
    def num_players(self) -> int:
        return _NUM_PLAYER

    @property
    def max_game_length(self) -> int:
        return _MAX_GAME_LENGTH

    @property
    def min_utility(self) -> float:
        return _MIN_UTILITY

    @property
    def max_utility(self) -> float:
        return _MAX_UTILITY

def _init(rng: chex.PRNGKey) -> State:
    init_card = jax.random.choice(
        rng, jnp.int8([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])
    )
    return State(  # type:ignore
        _cards=init_card,
    )


def _step(state: State, action: jnp.ndarray):
    action = jnp.int8(action)

    #Update pot
    pot = jax.lax.cond(
        action == BET,
        lambda: state._pot.at[state.current_player].add(1),
        lambda: state._pot,
    )
    state = state.replace(_pot=pot)
    #Check if game ends
    terminated = jax.lax.cond(
        (state._pot.min(-1) == J_2) | 
        ((action == PASS) & (state._step_count == J_2))|
        (state._step_count == J_3),
        lambda: TRUE,
        lambda: FALSE
        )

    #Compute rewards
    rewards = jax.lax.cond(
        terminated,
        lambda: _get_rewards(state),
        lambda: state.rewards
        )

    return state.replace(  # type:ignore
        current_player=1 - state.current_player,
        terminated=terminated,
        rewards=rewards,
        #_pot=pot,
        _bets= state._bets.at[2*(state._step_count-1)+action].set(True)
    )


def _get_rewards(state: State):
    rewards = jnp.float32([-1, -1])
    rewards = jax.lax.cond(
        state._pot[0] > state._pot[1],
        lambda _rewards: _rewards.at[0].set(1),
        lambda _rewards: _rewards,
        rewards
    )
    rewards = jax.lax.cond(
        state._pot[1] > state._pot[0],
        lambda _rewards: _rewards.at[1].set(1),
        lambda _rewards: _rewards,
        rewards
    )

    def _showdown(rewards: chex.Array):
        rewards = jax.lax.cond(
            state._cards[0] > state._cards[1],
            lambda _rewards: _rewards.at[0].set(1),
            lambda _rewards: _rewards.at[1].set(1),
            rewards
        ) 
        return rewards


    rewards = jax.lax.cond(
        state._pot[1] == state._pot[0],
        lambda _rewards: _showdown(_rewards),
        lambda _rewards: _rewards,
        rewards
        )
    return state._pot.min(-1) * rewards


def _observe(state: State, player_id) -> jnp.ndarray:
    """
    Index   Meaning
    0~1     index of the current player (num player)
    2~4     private card (num card)
    5~11    betting sequence
    """
    obs = jnp.zeros(11, dtype=jnp.bool_)
    obs = obs.at[player_id].set(TRUE)
    obs = obs.at[2+state._cards[player_id]].set(TRUE)
    obs = obs.at[5:].set(state._bets)
    return obs


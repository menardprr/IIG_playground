# Blend from open spiel Leduc poker 
# https://github.com/deepmind/open_spiel/blob/master/open_spiel/games/leduc_poker.cc
# and Pgx jax Leduc
# https://github.com/sotetsuk/pgx/blob/main/pgx/leduc_holdem.py


import jax
import jax.numpy as jnp
import chex

import games.game as ggame

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

INVALID_ACTION = jnp.int8(-1)
FOLD = jnp.int8(0)
CALL = jnp.int8(1)
RAISE = jnp.int8(2)

MAX_RAISE = jnp.int8(2)

MAX_GAME_LENGTH = 8
MAX_UTILITY = 13.0
MIN_UTILITY = - 13.0

@chex.dataclass(frozen=True)
class State(ggame.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(24, dtype=jnp.bool_)
    rewards: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(3, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Leduc Hold'Em specific ---
    _first_player: jnp.ndarray = jnp.int8(0)
    # [(player 0), (player 1), (public)]
    _cards: jnp.ndarray = jnp.int8([-1, -1, -1])
    # 0(Call)  1(Bet)  2(Fold)  3(Check)
    _last_action: jnp.ndarray = INVALID_ACTION
    _chips: jnp.ndarray = jnp.ones(2, dtype=jnp.int8)
    _round: jnp.ndarray = jnp.int8(0)
    _round_step: jnp.ndarray = jnp.int8(0)
    _raise_count: jnp.ndarray = jnp.int8(0)
    #Sequence of raise/call
    _bets: jnp.ndarray = jnp.zeros(2*MAX_GAME_LENGTH, jnp.bool_)

    @property
    def env_id(self) -> str:
        return "leduc_holdem"


class LeducPoker(ggame.Game):
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
        return "leduc_poker"

    @property
    def num_players(self) -> int:
        return 2
    
    @property
    def max_game_length(self) -> int:
        return MAX_GAME_LENGTH

    @property
    def min_utility(self) -> float:
        return MIN_UTILITY

    @property
    def max_utility(self) -> float:
        return MAX_UTILITY

def _init(rng: jax.random.KeyArray) -> State:
    rng1, rng2 = jax.random.split(rng, 2)
    init_card = jax.random.permutation(
        rng1, jnp.int8([0, 0, 1, 1, 2, 2]), independent=True
    )
    return State(
        _rng_key=rng2,
        _cards=init_card[:3],
        legal_action_mask=jnp.bool_([0, 1, 1]),
        _chips=jnp.ones(2, dtype=jnp.int8),
    )


def _step(state: State, action):
    action = jnp.int8(action)
    chips = jax.lax.switch(
        action,
        [
            lambda: state._chips,  # FOLD,
            lambda: state._chips.at[state.current_player].set(
                state._chips[1 - state.current_player]
            ),  # CALL
            lambda: state._chips.at[state.current_player].set(
                jnp.max(state._chips) + _raise_chips(state)
            ),  # RAISE
        ],
    )

    round_over, terminated, reward = _check_round_over(state, action)
    last_action = jax.lax.select(round_over, INVALID_ACTION, action)
    current_player = jax.lax.select(
        round_over, state._first_player, 1 - state.current_player
    )
    raise_count = jax.lax.select(
        round_over, jnp.int8(0), state._raise_count + jnp.int8(action == RAISE)
    )

    reward *= jnp.min(chips)

    legal_action = jax.lax.switch(
        action,
        [
            lambda: jnp.bool_([0, 0, 0]),  # FOLD
            lambda: jnp.bool_([0, 1, 1]),  # CALL
            lambda: jnp.bool_([1, 1, 1]),  # RAISE
        ],
    )
    legal_action = legal_action.at[RAISE].set(raise_count < MAX_RAISE)


    bets = jax.lax.select(
        action==CALL,
        state._bets.at[8 * (state._round) + 2 * state._round_step].set(TRUE),
        state._bets,
        )
    bets = jax.lax.select(
        action==RAISE,
        bets.at[8 * (state._round) + 2 * state._round_step + 1].set(TRUE),
        bets,
        )
    
    round_step = jax.lax.select(
        round_over,
        jnp.int8(0),
        state._round_step + 1,
        )
    

    return state.replace(  # type:ignore
        current_player=current_player,
        _last_action=last_action,
        legal_action_mask=legal_action,
        terminated=terminated,
        rewards=reward,
        _round=state._round + jnp.int8(round_over),
        _chips=chips,
        _raise_count=raise_count,
        _round_step=round_step,
        _bets=bets, 
    )


def _check_round_over(state, action):
    round_over = (action == FOLD) | (
        (state._last_action != INVALID_ACTION) & (action == CALL)
    )
    terminated = (round_over & (state._round == 1)) | (action == FOLD)

    reward = jax.lax.select(
        terminated & (action == FOLD),
        jnp.float32([-1, -1]).at[1 - state.current_player].set(1),
        jnp.float32([0, 0]),
    )
    reward = jax.lax.select(
        terminated & (action != FOLD),
        _get_unit_reward(state),
        reward,
    )
    return round_over, terminated, reward


def _get_unit_reward(state: State):
    win_by_one_pair = state._cards[state.current_player] == state._cards[2]
    lose_by_one_pair = (
        state._cards[1 - state.current_player] == state._cards[2]
    )
    win = win_by_one_pair | (
        ~lose_by_one_pair
        & (
            state._cards[state.current_player]
            > state._cards[1 - state.current_player]
        )
    )
    reward = jax.lax.select(
        win,
        jnp.float32([-1, -1]).at[state.current_player].set(1),
        jnp.float32([-1, -1]).at[1 - state.current_player].set(1),
    )
    return jax.lax.select(
        state._cards[state.current_player]
        == state._cards[1 - state.current_player],  # Draw
        jnp.float32([0, 0]),
        reward,
    )


def _raise_chips(state):
    """raise amounts is 2 in the first round and 4 in the second round."""
    return (state._round + 1) * 2



def _observe(state: State, player_id) -> jnp.ndarray:
    """
    Index   Meaning
    0~1     current player id (num player)
    2~4     private card (num distinct card)
    5~7     public card (num distinct card)
    8~23    betting sequence of raise/call (max_game_length*2)
    """
    obs = jnp.zeros(24, dtype=jnp.bool_)
    obs = obs.at[player_id].set(TRUE)
    obs = obs.at[2+state._cards[player_id]].set(TRUE)
    obs = jax.lax.select(
        state._round == 1, obs.at[5 + state._cards[2]].set(TRUE), obs
    )
    obs = obs.at[8:].set(state._bets)

    return obs
# Blend from open spiel Liar's Dice
# https://github.com/deepmind/open_spiel/blob/master/open_spiel/games/liars_dice.cc
# and Pgx jax game framework
# https://github.com/sotetsuk/pgx/blob/main/pgx/v1.py


import jax
import jax.numpy as jnp
import chex

from functools import partial

import games.game as ggame

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


# Number of sides per dice
DICE_SIDES = jnp.int8(6)
# Number of dice per player
NUM_DICE = jnp.int8(1)
TOTAL_NUM_DICE = 2 * NUM_DICE


MAX_GAME_LENGTH = TOTAL_NUM_DICE * DICE_SIDES + 1
MAX_UTILITY = 1.0
MIN_UTILITY = - 1.0


@chex.dataclass(frozen=True)
class State(ggame.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(1, dtype=jnp.bool_)
    rewards: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(
        DICE_SIDES*TOTAL_NUM_DICE+1, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Liar's Dice specific ---
    _first_player: jnp.ndarray = jnp.int8(0)
    # Last bet
    _last_bet: jnp.ndarray = jnp.int8(0)
    # dices outcome num_player * num_dice
    _dices: jnp.ndarray = jnp.ones((2, NUM_DICE), dtype=jnp.int8)
    # Sequence of possible bets
    _bets: jnp.ndarray = jnp.zeros(
        DICE_SIDES*TOTAL_NUM_DICE+1, dtype=jnp.bool_)

    @property
    def env_id(self) -> str:
        return "liars_dice"


class LiarsDice(ggame.Game):
    def __init__(self, num_dice=1, dice_sides=6):
        super().__init__()
        self.num_dice = num_dice
        self.dice_sides = dice_sides

    def _init(self, rng: jax.random.KeyArray) -> State:
        return partial(
            _init,
            num_dice=self.num_dice,
            dice_sides=self.dice_sides
        )(rng)

    def _step(self, state: State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return partial(
            _step,
            num_dice=self.num_dice,
            dice_sides=self.dice_sides
        )(state, action)

    def _observe(self, state: State, player_id: jnp.ndarray) -> jnp.ndarray:
        assert isinstance(state, State)
        return partial(
            _observe,
            num_dice=self.num_dice,
            dice_sides=self.dice_sides
        )(state, player_id)

    @property
    def id(self) -> str:
        return "liars_dice"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def max_game_length(self) -> int:
        return 2 * self.num_dice * self.dice_sides + 1

    @property
    def min_utility(self) -> float:
        return MIN_UTILITY

    @property
    def max_utility(self) -> float:
        return MAX_UTILITY


def _init(rng: jax.random.KeyArray, num_dice=1, dice_sides=6) -> State:
    _rng, rng = jax.random.split(rng, 2)
    dices = jax.random.choice(
        _rng,
        dice_sides,
        shape=(2, num_dice),
    )
    dices = jnp.sort(dices, axis=-1)
    num_action = 2 * num_dice * dice_sides + 1
    legal = jnp.ones(num_action, dtype=jnp.bool_)
    legal = legal.at[-1].set(FALSE)  # Cannot call liar first round
    return State(
        _rng_key=_rng,
        _dices=jnp.int8(dices),
        legal_action_mask=legal,
        _bets=jnp.zeros(num_action, jnp.bool_)
    )


def _step(state: State, action, num_dice=1, dice_sides=6) -> State:
    action = jnp.int8(action)
    num_action = 2 * num_dice * dice_sides + 1
    liar = jnp.int8(num_action-1)
    # Game ends if one player calls liar
    terminated = (action >= liar)
    # All bet above current one are possible
    legal = jnp.arange(num_action) > action
    # Record the bet
    _bets = state._bets.at[action].set(True)
    # Record last action
    _last_bet = action
    # Get reward
    rewards = jax.lax.select(
        terminated,
        _get_rewards(state, dice_sides),
        jnp.float32([0, 0])
    )
    return state.replace(
        current_player=1 - state.current_player,
        legal_action_mask=legal,
        _bets=_bets,
        _last_bet=_last_bet,
        rewards=rewards,
        terminated=terminated
    )


def _get_rewards(state: State, dice_sides=6):
    """ Bets have the form <dice>-<face>
    So, in a two-player game where each die has 6 faces, we have
    // Bet ID    Dice   Face
    // 0         1          1
    // 1         1          2
    // ...
    // 5         1          6
    // 6         2          1
    // ...
    // 11        2          6
    //
    Bet ID (2*num_dice) * #num faces encodes the special "liar" action."""
    # Decode the bet from bet id
    bet = state._last_bet
    bet_dice = bet // dice_sides + 1
    bet_side = bet % dice_sides + 1
    # Count the number of match, dice_sides is wild so always a match
    all_dice = jnp.reshape(state._dices, state._dices.shape[:-2] + (-1,))
    wild = jnp.int8(dice_sides)
    num_match = ((all_dice + 1) == bet_side) | ((all_dice + 1) == wild)
    num_match = num_match.sum(axis=-1)
    # Check if the bet is correct
    bet_correct = (num_match >= bet_dice)
    # Current player call liar so loose if bet correct
    rewards = jax.lax.select(
        bet_correct,
        jnp.float32([-1, -1]).at[1 - state.current_player].set(1),
        jnp.float32([-1, -1]).at[state.current_player].set(1),
    )
    return rewards


def _observe(state: State, player_id, num_dice=1, dice_sides=6) -> jnp.ndarray:
    """
    Index                            Meaning
    0~1=:n-1                         current player id (num player)
    n~n+num_dice*dice_sides-1=:m-1   private dices (num_dice*dice_sides)
    m~m+2*num_dice*dice_sides        betting sequence  (2*num_dice*dice_sides+1)
    """
    num_dice_face = num_dice * dice_sides
    num_action = 2 * num_dice * dice_sides + 1
    obs = jnp.zeros(
        2 + num_dice_face + num_action,
        dtype=jnp.bool_)
    obs = obs.at[player_id].set(TRUE)
    dices = state._dices.at[state.current_player].get()
    dices = jax.nn.one_hot(dices, dice_sides, dtype=jnp.bool_)
    dices = jnp.reshape(dices, dices.shape[:-2] + (-1,))
    obs = obs.at[2:2+num_dice_face].set(dices)
    obs = obs.at[2+num_dice_face:].set(state._bets)
    return obs

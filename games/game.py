import abc
from typing import Tuple

import jax
import jax.numpy as jnp
import chex

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


@chex.dataclass(frozen=True)
class State(abc.ABC):
    """Base state class 
    """

    current_player: jnp.ndarray
    observation: jnp.ndarray
    rewards: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    legal_action_mask: jnp.ndarray
    # NOTE: _rng_key is
    #   - used for stochastic env and auto reset
    #   - updated only when actually used
    #   - supposed NOT to be used by agent
    _rng_key: jax.random.KeyArray
    _step_count: jnp.ndarray

    @property
    @abc.abstractmethod
    def env_id(self) -> str:
        """Environment id (e.g. "go_19x19")"""
        ...


class Game(abc.ABC):
    """Game class API.
    """

    def __init__(self):
        ...

    def init(self, key: jax.random.KeyArray) -> State:
        """Return the initial state. Note that no internal state of
        environment changes.

        Args:
            key: pseudo-random generator key in JAX

        Returns:
            State: initial state of environment

        """
        key, subkey = jax.random.split(key)
        state = self._init(subkey)
        state = state.replace(_rng_key=key)
        observation = self.observe(state, state.current_player)
        return state.replace(observation=observation)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Step function."""
        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player

        # If the state is already terminated or truncated, environment does not take usual step,
        # but return the same state with zero-rewards for all players
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(rewards=jnp.zeros_like(
                state.rewards)),  # type: ignore
            lambda: self._step(state.replace(
                _step_count=state._step_count + 1), action),  # type: ignore
        )

        # Taking illegal action leads to immediate game terminal with negative reward
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )

        # All legal_action_mask elements are **TRUE** at terminal state
        # This is to avoid zero-division error when normalizing action probability
        # Taking any action at terminal state does not give any effect to the state
        state = jax.lax.cond(
            state.terminated,
            lambda: state.replace(  # type: ignore
                legal_action_mask=jnp.ones_like(state.legal_action_mask)
            ),
            lambda: state,
        )

        observation = self.observe(state, state.current_player)
        state = state.replace(observation=observation)  # type: ignore

        return state

    def observe(self, state: State, player_id: jnp.ndarray) -> jnp.ndarray:
        """Observation function."""
        obs = self._observe(state, player_id)
        return jax.lax.stop_gradient(obs)

    @abc.abstractmethod
    def _init(self, key: jax.random.KeyArray) -> State:
        """Implement game-specific init function here."""
        ...

    @abc.abstractmethod
    def _step(self, state, action) -> State:
        """Implement game-specific step function here."""
        ...

    @abc.abstractmethod
    def _observe(self, state: State, player_id: jnp.ndarray) -> jnp.ndarray:
        """Implement game-specific observe function here."""
        ...

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """Game id."""
        ...

    @property
    @abc.abstractmethod
    def num_players(self) -> int:
        """Number of players (e.g., 2 in Tic-tac-toe)"""
        ...

    @property
    @abc.abstractmethod
    def max_game_length(self) -> int:
        """Max length of a trajectory"""
        ...

    @property
    def num_actions(self) -> int:
        """Return the size of action space (e.g., 9 in Tic-tac-toe)"""
        state = self.init(jax.random.PRNGKey(0))
        return int(state.legal_action_mask.shape[0])

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Return the matrix shape of observation"""
        state = self.init(jax.random.PRNGKey(0))
        obs = self._observe(state, state.current_player)
        return obs.shape

    @property
    def _illegal_action_penalty(self) -> float:
        """Negative reward given when illegal action is selected."""
        return -1.0

    def _step_with_illegal_action(
        self, state: State, loser: jnp.ndarray
    ) -> State:
        penalty = self._illegal_action_penalty
        reward = (
            jnp.ones_like(state.rewards)
            * (-1 * penalty)
            * (self.num_players - 1)
        )
        reward = reward.at[loser].set(penalty)
        return state.replace(rewards=reward, terminated=TRUE)  # type: ignore


game_ids = [
    "kuhn_poker",
    "leduc_poker",
    "liars_dice",
    "full_liars_dice",
    "full_liars_dice_w1"
]


def game_factory(game_id: str) -> Game:
    assert game_id in game_ids, ("available games: "+" ".join(game_ids))
    if game_id == "kuhn_poker":
        from games.kuhn_poker import KuhnPoker
        return KuhnPoker()
    if game_id == "leduc_poker":
        from games.leduc_poker import LeducPoker
        return LeducPoker()
    if game_id == "liars_dice":
        from games.liars_dice import LiarsDice
        return LiarsDice()
    if game_id == "full_liars_dice":
        from games.full_liars_dice import FullLiarsDice
        return FullLiarsDice()
    if game_id == "full_liars_dice_w1":
        from games.full_liars_dice import FullLiarsDice
        return FullLiarsDice(wilde=1)

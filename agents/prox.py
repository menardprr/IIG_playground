from typing import Tuple
from functools import partial
import os
import torch

import chex
import jax
import jax.numpy as jnp
import optax

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agents.common.exploitability import Exploitability

from agents.common.networks import network_factory, optimizer_factory

import games.game as ggame
from games.game import game_factory

Params = chex.ArrayTree
PRNGKey = chex.PRNGKey

# Optimizer


def get_update_and_apply(optimizer):
    """ Get function that update the params and state of the optimizer"""

    def update_and_apply(params, grads, opt_state):
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return update_and_apply

# Utils function


def normalize_reward(rewards, min_reward, max_reward):
    """ Compute normalized rewards"""
    rewards = jnp.where(rewards > 0, rewards / max_reward,
                        rewards / jnp.abs(min_reward))
    return rewards


def normalize_logit(logit: chex.Array, legal: chex.Array) -> chex.Array:
    """Normalize logit by substrating logsumexp over legal action
    by convention the logit of illegal action is zero
    Ags:
        logit: logits of shape [..., A]
        legal: legal action mask of shape [...., A]
    Returns
        normalized logit of shape [..., A]
        policy of shape [...,A]
    """
    # Compute max of legal logit
    legal = jax.lax.stop_gradient(legal)
    legal_logit = jnp.where(legal, logit, -jnp.inf)
    max_legal_logit = legal_logit.max(axis=-1, keepdims=True)
    max_legal_logit = jax.lax.stop_gradient(max_legal_logit)
    # Comput safe input logit see https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    _logit = jnp.where(legal, logit-max_legal_logit, 0.0)
    # Compute exponential of logit with the right max for numerical stability
    # and replace by zero exp of illegal action
    exp_logit = jnp.where(legal, jnp.exp(_logit), 0.0)
    # Compute normalization constant
    baseline = jnp.log(jnp.sum(exp_logit, axis=-1, keepdims=True))
    # Return the log of the policy with the convention of zero logprob for
    # illegal actions
    logpi = jnp.where(
        legal,
        logit - baseline - max_legal_logit,
        0.0
    )
    pi = exp_logit / exp_logit.sum(axis=-1, keepdims=True)
    return logpi, pi


def generate_get_logit_func(pi_apply):
    """A function generator to compute logit."""

    def _get_logit(
            pi_params: Params,
            info_state: chex.Array,
            action_mask: chex.Array
    ) -> Tuple[chex.Array]:
        """Return logit, log prob and prob."""
        logit = pi_apply(pi_params, info_state)
        logpi, pi = normalize_logit(logit, action_mask)
        return logit, logpi, pi

    return _get_logit


def generate_get_value_func(val_apply):
    """A function generator to compute value."""

    def _get_value(val_params, info_state):
        return val_apply(val_params, info_state)

    return _get_value

# Class for steps


@chex.dataclass(frozen=True)
class EnvStep:
    """The environment step tensor summary."""
    # Indicates whether the state comes after a terminal state,
    # i.e. a valid state or just a padding. Shape: [...]
    # The terminal state being the first one to be marked valid.
    valid: chex.Array = ()
    # The single tensor representing the state observation. Shape: [..., ??]
    obs: chex.Array = ()
    # The legal actions mask for the current player. Shape: [..., A]
    legal: chex.Array = ()
    # The current player id as an int. Shape: [...]
    player_id: chex.Array = ()
    # The rewards of all the players. Shape: [..., P]
    rewards: chex.Array = ()


@chex.dataclass(frozen=True)
class ActorStep:
    """The actor step tensor summary."""
    # The action (as one-hot) of the current player. Shape: [..., A]
    action_oh: chex.Array = ()
    # The log policy of the current player. Shape: [..., A]
    logpi: chex.Array = ()
    # Note - these are rewards obtained *after* the actor step, and thus
    # these are the same as EnvStep.rewards visible before the *next* step.
    # Shape: [..., P]
    rewards: chex.Array = ()
    # Proximal regularization.  Shape: [..., A]
    reg: chex.Array = ()
    # Value predicted by the value network.  Shape: [...]
    value: chex.Array = ()
    # Placeholder for te advantages.  Shape: [...]
    advantage: chex.Array = ()


@chex.dataclass(frozen=True)
class TimeStep:
    """The tensor data for one game transition (env_step, actor_step)."""
    env: EnvStep = EnvStep()
    actor: ActorStep = ActorStep()

# A class to handle parameters


class ProxAgentConfig:
    def __init__(
            self,
            batch_size=256,
            prox_cf=0.1,
            prox_period=100_000,
            prox_stop=-1,
            prox_clip=1.0,
            ent_cf=0.01,
            ent_clip=1.0,
            eps_thr=0.01,
            zero_sum_reg=False,
            adv_clip=2.0,
            gae_lambda=1.0,
            name="Prox"
    ):
        # Name
        self.name = name

        # Batchsize
        self.batch_size = batch_size

        # Proximal regularization parameters
        self.prox_cf = prox_cf
        self.prox_period = prox_period
        self.prox_stop = prox_stop
        self.prox_clip = prox_clip

        # Entropy regularization parameters
        self.ent_cf = ent_cf
        self.ent_clip = ent_clip

        # Epsilon threshold
        self.eps_thr = eps_thr

        # Zero sum regularization (not used)
        self.zero_sum_reg = zero_sum_reg

        # Advantage
        self.adv_clip = adv_clip

        # GAE lambda
        self.gae_lambda = gae_lambda

    def get_learning_params(self):
        """Returns parameters in a dataclass"""
        return LearningParams(
            prox_cf=jnp.float32(self.prox_cf),
            prox_period=jnp.int32(self.prox_period),
            prox_stop=jnp.int32(self.prox_stop),
            prox_clip=jnp.float32(self.prox_clip),
            ent_cf=jnp.float32(self.ent_cf),
            ent_clip=jnp.float32(self.ent_clip),
            eps_thr=jnp.float32(self.eps_thr),
            adv_clip=jnp.float32(self.adv_clip),
        )


@chex.dataclass(frozen=True)
class LearningParams:
    """A structure to gather learning parameters..."""
    prox_cf: chex.Array = jnp.float32(1.0)
    prox_period: chex.Array = jnp.int32(1.0)
    prox_stop: chex.Array = jnp.int32(-1)
    prox_clip: chex.Array = jnp.float32(1.0)
    ent_cf: chex.Array = jnp.float32(1.0)
    ent_clip: chex.Array = jnp.float32(1.0)
    eps_thr: chex.Array = jnp.float32(0.0)
    adv_clip: chex.Array = jnp.float32(1.0)
    learner_step: chex.Array = jnp.int32(0)

# Training params


@chex.dataclass(frozen=True)
class TrainingParams:
    """A structure to gather training parameters..."""
    budget: chex.Array = jnp.int32(1)
    log_interval: chex.Array = jnp.int32(1)
    eval_interval: chex.Array = jnp.int32(1)
    checkpoint_interval: chex.Array = jnp.int32(10_000)


# Trajectory collection
def get_act(get_logit, num_actions):
    """Get the actor step function"""

    def _act(
            rng: PRNGKey,
            pi_params: Params,
            info_state: chex.Array,
            action_mask: chex.Array,
    ) -> Tuple[chex.Array]:
        """Sample an action and return log prob"""
        # Compute logit
        _, logpi, _ = get_logit(pi_params, info_state, action_mask)
        # Sample action
        legal_logpi = jnp.where(action_mask, logpi, -jnp.inf)
        action = jax.random.categorical(rng, legal_logpi, axis=-1)

        return action, logpi

    def act(
            rng: PRNGKey,
            pi_params: Params,
            env_step: EnvStep,
    ) -> Tuple[chex.Array, ActorStep]:
        """Actor act function
        Args:
            rng: random key.
            pi_params: parameters of the policy network.
            env_step: env step.
        Returns:
            Return a vector of action and an ActorStep
        """
        action, logpi = _act(
            rng,
            pi_params,
            env_step.obs,
            env_step.legal,
        )
        action_oh = jax.nn.one_hot(action, num_actions)
        action_oh = action_oh * env_step.legal  # action should be legal
        actor_step = ActorStep(
            logpi=logpi,
            action_oh=action_oh,
            advantage=jnp.zeros(logpi.shape[:-1]),
            value=jnp.zeros(logpi.shape[:-1]),
            reg=jnp.zeros(logpi.shape[:-1]),
        )

        return action, actor_step

    return act


def get_collect_batch_trajectory(
        init,
        step,
        act,
        batch_size,
        max_trajectory_length,
):
    """get the collect batch trajectory function"""

    def scan_step(state: ggame.State, key: PRNGKey, pi_params: Params):
        # Build env step
        env_step = EnvStep(
            obs=state.observation,
            legal=state.legal_action_mask,
            player_id=state.current_player,
            valid=~state.terminated,
            rewards=state.rewards
        )
        # Act
        action, actor_step = act(key, pi_params, env_step)
        state = step(state, action)
        # Build timestep
        timestep = TimeStep(
            env=env_step,
            actor=actor_step.replace(rewards=state.rewards)
        )
        return state, timestep

    def collect_batch_trajectory(rng: PRNGKey, pi_params: Params) -> TimeStep:
        """Collect a batch of trajectories and convert them into timestep."""

        # Sample initial state
        _rng, rng = jax.random.split(rng, 2)
        rngs = jax.random.split(_rng, batch_size)
        state = init(rngs)
        # Sample a trajectory
        rngs = jax.random.split(rng, max_trajectory_length)
        _, timestep = jax.lax.scan(
            partial(scan_step, pi_params=pi_params),
            state,
            rngs
        )
        return timestep

    return collect_batch_trajectory

# Compute GAE


@chex.dataclass(frozen=True)
class GAE_carry:
    """A structure to carry information when compute gae"""
    reward_acum: chex.ArrayDevice = ()
    next_value: chex.ArrayDevice = ()
    next_advantage: chex.ArrayDevice = ()


@chex.dataclass(frozen=True)
class GAE_x:
    """A structure of input to compute gae"""
    reward: chex.ArrayDevice = ()
    value: chex.ArrayDevice = ()
    is_player: chex.ArrayDevice = ()
    valid: chex.ArrayDevice = ()


def get_compute_advantage(
        num_players,
        get_logit,
        get_value,
        normalize_reward,
        gae_lambda,
        zero_sum_reg,
):
    """Function to set a function that compute advantage"""

    def compute_gae(
        gae_carry: GAE_carry,
        gae_x: GAE_x,
    ):
        """A function made to be called by scan to compute GAE"""
        # Extract variables from dataclass
        reward_acum, next_value, next_advantage = gae_carry.values()
        reward, value, is_player, valid = gae_x.values()
        # Compute reward
        reward = reward + reward_acum
        # Compute delta and update advantage if play
        delta = reward + next_value - value
        advantage = delta + gae_lambda * next_advantage
        advantage = jnp.where(is_player, advantage, next_advantage)
        advantage = jnp.where(valid, advantage, 0.0)
        # Update value if play and reset if not valid
        value = jnp.where(is_player, value, next_value)
        value = jnp.where(valid, value, 0.0)
        # Update accumulated reward: if play or invalid reset to zero
        reward_acum = jnp.where(is_player, 0.0, reward)
        reward_acum = jnp.where(valid, reward_acum, 0.0)
        # Construct next GAE carrry
        next_gae_carry = GAE_carry(
            reward_acum=reward_acum,
            next_value=value,
            next_advantage=advantage
        )
        return next_gae_carry, advantage

    def add_reg(
        learning_params: LearningParams,
        pi_prox_params: Params,
        ts: TimeStep
    ):
        """Function to add to the rewards proximal and entropy regularization"""
        # Get logpi
        logpi = ts.actor.logpi
        # Compute prox logpi
        rollout = jax.vmap(get_logit, (None, 0, 0), (0, 0, 0))
        _, prox_logpi, _ = rollout(
            pi_prox_params,
            ts.env.obs,
            ts.env.legal
        )
        # Compute uniform logpi
        ent_logpi = jnp.log(1.0 / ts.env.legal.sum(axis=-1, keepdims=True))
        ent_logpi = ent_logpi * ts.env.legal
        # Compute regularization

        def get_reg(logpi, reg_logpi, cf, clip):
            """Compute regularization: clip(ratio_logpi/lr)"""
            reg = (logpi - reg_logpi) * cf
            reg = jnp.where(ts.actor.action_oh, reg, 0.0).sum(axis=-1)
            reg = ts.env.valid * reg
            reg = jnp.clip(
                reg,
                a_min=-clip,
                a_max=clip
            )
            return reg
        ent_reg = get_reg(
            logpi,
            ent_logpi,
            learning_params.ent_cf,
            learning_params.ent_clip,
        )
        prox_reg = get_reg(
            logpi,
            prox_logpi,
            learning_params.prox_cf,
            learning_params.prox_clip,
        )
        reg = prox_reg + ent_reg
        # Add regularization reward
        rewards = normalize_reward(ts.actor.rewards)
        for player in range(num_players):
            is_player = ts.env.player_id == player
            reg_reward = (- reg) * (is_player + zero_sum_reg * (is_player - 1))
            rewards = rewards.at[..., player].set(
                rewards[..., player] + reg_reward
            )
        return ts.replace(actor=ts.actor.replace(
            rewards=rewards,
            reg=reg,
        ))

    def compute_advantage(
            agent_state: AgentState,
            ts: TimeStep
    ) -> TimeStep:
        """Function that compute advantage (gae) of a timestep given parameter
        of the value network
        Args:
        agent_state: state of the agent 
        ts: a timestep 
        Returns:
        A timestep with updated advantage and value
        """

        # Normalize and add regularization reward
        ts = add_reg(
            agent_state.learning_params,
            agent_state.pi_prox_params,
            ts,
        )
        # Compute value
        rollout = jax.vmap(get_value, (None, 0), 0)
        value = rollout(
            agent_state.val_params,
            ts.env.obs
        )
        value = ts.env.valid * value
        # Compute adavantage player by player
        advantage = jnp.zeros_like(value)
        for player in range(num_players):
            is_player = ts.env.player_id == player
            # Prepare carry and input
            gae_x = GAE_x(
                reward=ts.actor.rewards[..., player],
                value=value,
                is_player=is_player,
                valid=ts.env.valid,
            )
            gae_carry = GAE_carry(
                reward_acum=jnp.zeros_like(value[0, ...]),
                next_value=jnp.zeros_like(value[0, ...]),
                next_advantage=jnp.zeros_like(value[0, ...])
            )
            # Backwardly scan over value to compute gae
            _advantage = jax.lax.scan(
                compute_gae,
                gae_carry,
                gae_x,
                reverse=True
            )[1]
            advantage = jnp.where(is_player, _advantage, advantage)
        return ts.replace(
            actor=ts.actor.replace(
                value=value,
                advantage=advantage,
            ))

    return compute_advantage


# Loss function and update generators
def get_loss_funcs(
        get_logit,
        get_value,
        gae_lambda,
):
    """A function that generate the loss function
    Args:
        get_logit: a function that returns the logit of the policy.
        get_value: a function that returns value.
        gae_lambda: the gae lambda parameter (not used)
    Returns:
        A policy and value loss functions
    """

    def valid_mean(x: chex.Array, valid: chex.Array) -> chex.Array:
        "Compute the mean over valid steps"
        num_valid = valid.sum()
        normalization = (num_valid + (num_valid <= 0.0))
        return (x * valid).sum() / normalization

    def val_loss(
        val_params: Params,
        ts: TimeStep
    ):
        # Compute values
        value = get_value(
            val_params,
            ts.env.obs
        )
        # Extract target and advantage from timestep
        value_target = ts.actor.value
        advantage = ts.actor.advantage
        # Compute target
        value_target = advantage + value_target
        value_target = jax.lax.stop_gradient(value_target)
        # Compute value loss with td-gae-lambda targets
        loss = jnp.square(value - value_target)
        # smaller weight for batch with few valid timestep
        loss = (loss * ts.env.valid).mean()
        val_mean = valid_mean(value, ts.env.valid)
        return loss, val_mean

    def pi_loss(
        pi_params: Params,
        learning_params: LearningParams,
        ts: TimeStep
    ) -> float:

        # Compute logpi
        _, logpi, _ = get_logit(
            pi_params,
            ts.env.obs,
            ts.env.legal,
        )
        logpi = jnp.where(ts.actor.action_oh, logpi, 0.0).sum(axis=-1)

        # Log regularization
        reg_mean = valid_mean(ts.actor.reg, ts.env.valid)

        # Compute and log entropy
        ent = - logpi * ts.env.valid
        ent_mean = valid_mean(ent, ts.env.valid)

        # Compute advantage
        adv = ts.actor.advantage
        adv = jnp.clip(adv, -learning_params.adv_clip,
                       learning_params.adv_clip)
        adv = jax.lax.stop_gradient(adv)
        adv_mean = valid_mean(adv, ts.env.valid)

        # Epsilon mask loss
        eps = learning_params.eps_thr
        eps_mask_up = (adv > 0) & (logpi > jnp.log(1-eps))
        eps_mask_down = (adv < 0) & (logpi < jnp.log(eps))
        eps_mask = jax.lax.stop_gradient(~(eps_mask_up | eps_mask_down))

        # Compute loss
        # smaller weight for batch with few valid timestep
        loss = (ts.env.valid * eps_mask * (- adv) * logpi).mean()
        return loss, (adv_mean, reg_mean, ent_mean)

    return pi_loss, val_loss

# Update parameters


@chex.dataclass(frozen=True)
class AgentState:
    """A structure of the current agent state"""
    pi_params: Params = ()
    pi_prox_params: Params = ()
    val_params: Params = ()
    pi_opt_state: optax.OptState = ()
    val_opt_state: optax.OptState = ()
    learning_params: LearningParams = ()


def get_update_agent_func(
    pi_loss,
    val_loss,
    pi_update_and_apply,
    val_update_and_apply,
):
    """A function that generates the update parameters function
    Args:
        pi_loss: a function that computes the policy loss and its gradient
        val_loss: a function that computes the value loss and its gradient
        pi_update_and_apply: a function that updates the optimizer state 
            and applies the gradients of the policy network 
        val_update_and_apply: a function that updates the optimizer state 
            and applies the gradients of the value network
    """

    def update_pi_parameters(
        params: Params,
        opt_state: optax.OptState,
        learning_params: LearningParams,
        ts: TimeStep
    ):
        """A function that update policy params by one step of gradient
            descent on the policy loss"""
        # Compute loss
        (loss, stats), grads = pi_loss(
            params,
            learning_params,
            ts
        )
        # Update `params` using the computed gradient.
        params, opt_state = pi_update_and_apply(params, grads, opt_state)

        return params, opt_state, (loss, *stats)

    def update_val_parameters(
        params: Params,
        opt_state: optax.OptState,
        ts: TimeStep
    ):
        """A function that update value parameters by one step of gradient
            descent on the value loss"""
        # Compute loss
        loss, grads = val_loss(
            params,
            ts
        )
        # Update `params` using the computed gradient.
        params, opt_state = val_update_and_apply(params, grads, opt_state)

        return params, opt_state, loss

    def update_batch(
            agent_state: AgentState,
            ts: TimeStep
    ):

        # Update policy network
        pi_params, pi_opt_state, pi_stats = update_pi_parameters(
            agent_state.pi_params,
            agent_state.pi_opt_state,
            agent_state.learning_params,
            ts
        )

        # Update value netwok
        val_params, val_opt_state, val_stats = update_val_parameters(
            agent_state.val_params,
            agent_state.val_opt_state,
            ts
        )

        # Update agent state
        agent_state = agent_state.replace(
            pi_params=pi_params,
            val_params=val_params,
            pi_opt_state=pi_opt_state,
            val_opt_state=val_opt_state
        )

        return agent_state, pi_stats + val_stats

    def update_prox_parameters(
            agent_state: AgentState,
    ):
        """Update the prox parameters with polyak average or hard update"""
        # Polyak update
        l_params = agent_state.learning_params
        polyak = 1.0 / l_params.prox_period
        polyak = jax.lax.select(
            (l_params.prox_stop >= 0) & (
                l_params.learner_step >= l_params.prox_stop),
            0.0,
            polyak
        )
        pi_prox_params = optax.incremental_update(
            agent_state.pi_params,
            agent_state.pi_prox_params,
            polyak
        )
        # Update agent state
        agent_state = agent_state.replace(pi_prox_params=pi_prox_params)
        return agent_state

    def update_agent(
            rng: PRNGKey,
            agent_state: AgentState,
            ts: TimeStep
    ):
        """Update policy and value parameters with a batch"""

        # Update policy and value network
        ts = jax.tree_map(lambda x: jnp.reshape(
            x, (-1,) + x.shape[2:]), ts)  # Flatten time step
        agent_state, stats = update_batch(
            agent_state,
            ts
        )

        # Update prox network
        agent_state = update_prox_parameters(agent_state)

        return (rng, agent_state), stats

    return update_agent


# Update learning parameters
def get_udpate_learning_params(config: ProxAgentConfig):

    def update_learning_params(learning_params: LearningParams):
        """Update the learning parameters: lr, ix....
        learner_step +=1
        """
        step = learning_params.learner_step
        step = step + 1
        return learning_params.replace(
            learner_step=step,
        )

    return update_learning_params

# Get step function


def get_step_func(
    collect_batch_trajectory,
    compute_advantage,
    update_agent,
    update_learning_params,
):
    """A function that generates the step function
    Args:
        collect_batch_trajectory: a function that collect trajectories
        compute_advantage: a function that computes advantages
        update_agent: a function that updates the optimizer state 
            and parameters 
        update_learning_params: a function that updates the learning parameters 
    """
    def step(rng: PRNGKey, agent_state: AgentState):
        """One step of the algorithm, that plays the game and improves params."""

        rng1, rng2 = jax.random.split(rng)

        # Collect trajectories
        timestep = collect_batch_trajectory(
            rng1,
            agent_state.pi_params
        )

        # Compute advantages
        timestep = compute_advantage(
            agent_state,
            timestep
        )

        # Update agent parameters
        (_, agent_state), stats = update_agent(
            rng2,
            agent_state,
            timestep
        )

        # Update learning params
        agent_state = agent_state.replace(
            learning_params=update_learning_params(
                agent_state.learning_params)
        )

        return agent_state, stats

    return step


class ProxAgent():
    """Prox algorithm."""

    def __init__(
        self,
        game_name,
        config_kwargs=None,
        training_kwargs=None,
        pi_network_kwargs=None,
        val_network_kwargs=None,
        pi_optimizer_kwargs=None,
        val_optimizer_kwargs=None,
        writer_path=None,
        checkpoint_path=None,
        verbose=False,
        seed=42
    ):

        # Random key
        self._rng = jax.random.PRNGKey(seed)

        # Writer
        self.verbose = verbose
        self.writer_path = writer_path
        if writer_path is not None:
            self.writer = SummaryWriter(writer_path)
        else:
            self.writer = None

        # Game parameters
        game = game_factory(game_name)
        self._game = game
        self.action_size = game.num_actions
        self.info_state_size = game.observation_shape[0]
        self.max_trajectory_length = game.max_game_length
        self.num_players = game.num_players

        # Training
        self.training = TrainingParams(**training_kwargs)

        # Config
        self.config = ProxAgentConfig(**config_kwargs)
        self.update_learning_params = jax.jit(
            get_udpate_learning_params(self.config))

        # Policy Network
        pi_network_kwargs['num_actions'] = self.action_size
        pi_net = network_factory('PolicyNet', pi_network_kwargs)

        init_inputs = jnp.ones((self.config.batch_size, self.info_state_size))
        _rng = self._next_rng_key()
        pi_params = pi_net.init(_rng, init_inputs)
        pi_prox_params = pi_net.init(_rng, init_inputs)

        self.get_logit = jax.jit(generate_get_logit_func(pi_net.apply))
        self.act = get_act(
            self.get_logit,
            self.action_size,
        )

        # Value network
        val_net = network_factory('ValueNet', val_network_kwargs)

        init_inputs = jnp.ones((self.config.batch_size, self.info_state_size))
        val_params = val_net.init(self._next_rng_key(), init_inputs)

        self.get_value = jax.jit(generate_get_value_func(val_net.apply))

        # Optimizers
        def get_optimizer(_optimizer_kwargs, _params):
            if _optimizer_kwargs["schedule"] is not None:
                transition_steps = self.training.budget
                _optimizer_kwargs["schedule"]["transition_steps"] = transition_steps
                schedule = optax.linear_schedule(
                    **_optimizer_kwargs["schedule"]
                )
                _optimizer_kwargs["optimizer_kwargs"]["learning_rate"] = schedule
            optimizer_type = _optimizer_kwargs["optimizer_type"]
            optimizer_kwargs = _optimizer_kwargs["optimizer_kwargs"]
            optimizer = optimizer_factory(optimizer_type, optimizer_kwargs)
            if _optimizer_kwargs["clip_gradient"] is not None:
                optimizer = optax.chain(
                    optimizer,
                    optax.clip(_optimizer_kwargs["clip_gradient"]),
                )
            return optimizer.init(_params), get_update_and_apply(optimizer)

        pi_opt_state, pi_update_and_apply = get_optimizer(
            pi_optimizer_kwargs,
            pi_params
        )
        val_opt_state, val_update_and_apply = get_optimizer(
            val_optimizer_kwargs,
            val_params
        )

        # Create agent state
        self.agent_state = AgentState(
            pi_params=pi_params,
            pi_prox_params=pi_prox_params,
            val_params=val_params,
            pi_opt_state=pi_opt_state,
            val_opt_state=val_opt_state,
            learning_params=self.config.get_learning_params()
        )

        # Trajectory collection
        game_init = jax.vmap(game.init)
        game_step = jax.vmap(game.step)
        self.collect_batch_trajectory = get_collect_batch_trajectory(
            game_init,
            game_step,
            self.act,
            self.config.batch_size,
            self.max_trajectory_length
        )

        # Reward transformation
        self.normalize_rewards = partial(
            normalize_reward,
            min_reward=game.min_utility,
            max_reward=game.max_utility,
        )

        # Advantage computation
        self.compute_advantage = get_compute_advantage(
            self.num_players,
            self.get_logit,
            self.get_value,
            self.normalize_rewards,
            self.config.gae_lambda,
            zero_sum_reg=self.config.zero_sum_reg,
        )

        # Loss function
        _pi_loss, _val_loss = get_loss_funcs(
            self.get_logit,
            self.get_value,
            self.config.gae_lambda
        )
        pi_loss = jax.value_and_grad(_pi_loss, has_aux=True)
        val_loss = jax.value_and_grad(_val_loss, has_aux=True)

        # Update
        self.update_agent = get_update_agent_func(
            pi_loss,
            val_loss,
            pi_update_and_apply,
            val_update_and_apply,
        )

        # Step
        self.step = jax.jit(get_step_func(
            self.collect_batch_trajectory,
            self.compute_advantage,
            self.update_agent,
            self.update_learning_params
        ))

        # Exploitability with open spiel
        self.expl = Exploitability(
            game_name,
            self.get_logit,
        )

        # Checkpoint
        self.checkpoint_path = checkpoint_path
        self.n_step = 0
        if checkpoint_path is not None:
            if os.path.isdir(checkpoint_path):
                # Load last checkpoint
                print("Load last checkpoint at " + checkpoint_path)
                self.load_checkpoint()
            else:
                # Create checkpoint directory
                os.mkdir(checkpoint_path)

    def _next_rng_key(self) -> chex.PRNGKey:
        """Get the next rng subkey from class rngkey.
        Must *not* be called from under a jitted function!
        Returns:
            A fresh rng_key.
        """
        self._rng, subkey = jax.random.split(self._rng)
        return subkey

    def fit(self):
        """Train the agent"""
        for n_step in tqdm(range(self.n_step, self.training.budget)):
            self.agent_state, stats = self.step(
                self._next_rng_key(),
                self.agent_state
            )
            if n_step % self.training.log_interval == 0:
                stats = tuple(x.item() for x in stats)
                pi_loss, adv_mean, reg_mean, ent_mean, val_loss, val_mean = stats
                if self.writer:
                    self.writer.add_scalar("loss/pi_loss", pi_loss, n_step)
                    self.writer.add_scalar("loss/val_loss", val_loss, n_step)
                    self.writer.add_scalar("value/val_mean", val_mean, n_step)
                    self.writer.add_scalar("value/adv_mean", adv_mean, n_step)
                    self.writer.add_scalar("value/ent_mean", ent_mean, n_step)
                    self.writer.add_scalar("value/reg_mean", reg_mean, n_step)
            if n_step % self.training.eval_interval == 0:
                self.log_exploitability(
                    "cur", self.agent_state.pi_params, n_step)
                self.log_exploitability(
                    "prox", self.agent_state.pi_prox_params, n_step)
            if n_step % self.training.checkpoint_interval == 0:
                if self.checkpoint_path:
                    self.checkpoint(n_step)

    def log_exploitability(self, name, params, n_step):
        nash_conv, player_improvements = self.expl.compute_exploitability(
            params)
        if self.writer:
            self.writer.add_scalar(f"nash_{name}/exp", nash_conv, n_step)
        if self.verbose:
            print(f"Nashconv_{name}: {nash_conv}")
        for player in range(self.num_players):
            if self.writer:
                self.writer.add_scalar(
                    f"nash_{name}/_imp_player_{player}",
                    player_improvements[player],
                    n_step
                )

    def _get_state(self):
        """"Get the current global state of the agent"""
        return dict(
            # The agent state
            agent_state=self.agent_state,
            # Writer path
            writer_path=self.writer_path,
            # Checkpoint path
            checkpoint_path=self.checkpoint_path,
            # The randomness key
            rng=self._rng
        )

    def _set_state(self, state):
        """"Set the global state of the agent"""
        # The agent state
        self.agent_state = state["agent_state"]
        # Writer path
        self.writer_path = state["writer_path"]
        if self.writer_path is not None:
            self.writer = SummaryWriter(self.writer_path)
            # if os.path.exists(self.writer_path):
            #     shutil.rmtree(self.writer_path, ignore_errors=True)
        # Checkpoint path
        self.checkpoint_path = state["checkpoint_path"]
        # The randomness key
        self._rng = state["rng"]
        # Set n step
        self.n_step = self.agent_state.learning_params.learner_step

    def save(self, path):
        """Save the agent global state"""
        torch.save(self._get_state(), path)

    def load(self, path):
        """Load the agent global state"""
        self._set_state(torch.load(path))

    def load_checkpoint(self):
        """Load last checkpoint"""
        list_state = os.listdir(self.checkpoint_path)
        last_state = max(list_state, key=lambda x: int(x.split('.')[-2]))
        state_path = os.path.join(self.checkpoint_path, last_state)
        print("Load: " + state_path)
        self.load(state_path)

    def checkpoint(self, n_step):
        """Checkpoint the agent"""
        checkpoint_name = self.config.name + "." + str(n_step) + ".jx"
        self.save(os.path.join(self.checkpoint_path, checkpoint_name))

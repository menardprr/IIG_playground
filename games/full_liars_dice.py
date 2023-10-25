# Custom implementation of the Liar's dice game

import jax
import jax.numpy as jnp
import chex

from functools import partial

import games.game as ggame

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


MAX_UTILITY = 1.0
MIN_UTILITY = - 1.0

DICE_SIDES = 6
MAX_NUM_DICE = 5



def full_to_round(obs, legal, max_num_dice=5, dice_sides=6):
    """
    Convert observation and legal from full Liar's dice to one round Liar's dice
    """
    #Get number of dice
    num_dice_oh = obs[2:2+max_num_dice]
    num_dice = num_dice_oh * (jnp.arange(num_dice_oh.shape[0]) + 1)
    num_dice = num_dice.sum().item()
    num_dice_face = num_dice * dice_sides
    num_action = 2 * num_dice * dice_sides + 1
    #Initialize observation for one round Liar dice
    _obs = jnp.zeros(
        2 + num_dice_face + num_action,
        dtype=jnp.bool_)
    #Player Id
    _obs = _obs.at[:2].set(obs[:2])
    #Private dices
    idx_dice = 2 + 2 * max_num_dice
    idx_bet = idx_dice + (dice_sides + 1) * max_num_dice
    dice_count = obs.at[idx_dice:idx_dice+(dice_sides+1)*(max_num_dice+1)].get()
    dice_count = dice_count.reshape((dice_sides + 1, max_num_dice+1))
    dice_count = dice_count.at[:-1,:].get()
    dice_count = jnp.argmax(dice_count, axis=-1)
    dice = jnp.repeat(jnp.arange(dice_sides), dice_count)
    dice = jnp.sort(dice, axis=-1)
    dice = jax.nn.one_hot(dice, dice_sides, dtype=jnp.bool_)
    dice = jnp.reshape(dice, dice.shape[:-2] + (-1,))
    _obs = _obs.at[2:2+num_dice_face].set(dice)
    #Bets
    _obs = _obs.at[2+num_dice_face:].set(obs[idx_bet:idx_bet+num_action])
    #Legal 
    _legal = legal.at[:num_action].get()
    return _obs, _legal

def round_to_full(obs, legal, num_dice=1, max_num_dice=5, dice_sides=6):
    """
    Convert observation and legal from one round Liar's dice to full Liar's dice
    """
    #Initialize observation for full Liar dice
    max_num_actions = 2 * max_num_dice * dice_sides + 1
    _obs = jnp.zeros(
        2+2*max_num_dice+(dice_sides+1)*max_num_dice+max_num_actions,
        dtype=jnp.bool_)
    #Player Id
    _obs = _obs.at[:2].set(obs[:2])
    #Num of dice
    num_dice_0 = jax.nn.one_hot(num_dice-1, max_num_dice, dtype=jnp.bool_) 
    _obs = _obs.at[2:2+max_num_dice].set(num_dice_0)
    _obs = _obs.at[2+max_num_dice:2+2*max_num_dice].set(num_dice_0) # num_dice_0 == num_dice_1
    #Private dices
    dice = obs.at[2:2+(num_dice * dice_sides)].get()
    dice = jnp.reshape(dice, (num_dice,dice_sides))
    dice = jnp.argmax(dice, axis=-1)
    dice = jnp.int8(dice)
    _dice = dice_sides * jnp.ones(max_num_dice, dtype=jnp.int8)
    _dice = _dice.at[:num_dice].set(dice)
    _dice = jnp.sort(_dice, axis=-1)
    __dice = jnp.zeros((dice_sides+1,), dtype=jnp.int8)
    _dice = __dice.at[_dice].add(1)
    _dice = jax.nn.one_hot(_dice, max_num_dice+1, dtype=jnp.bool_)
    _dice = jnp.reshape(_dice, (-1,))
    idx = 2 + 2 * max_num_dice
    _obs = _obs.at[idx:idx+(dice_sides+1)*(max_num_dice+1)].set(_dice)
    #Bets
    idx = idx + (dice_sides+1) * max_num_dice
    _bets = jnp.zeros(max_num_actions, dtype=jnp.bool_)
    num_action = 2 * num_dice * dice_sides + 1
    num_dice_face = num_dice * dice_sides
    _bets = _bets.at[:num_action].set(obs.at[2+num_dice_face:].get())
    _obs = _obs.at[idx:].set(_bets)
    #Legal 
    _legal = jnp.zeros(max_num_actions, dtype=jnp.bool_)
    _legal = _legal.at[:num_action].set(legal)
    return _obs, _legal


@chex.dataclass(frozen=True)
class State(ggame.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(1, dtype=jnp.bool_)
    rewards: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(DICE_SIDES*2*MAX_NUM_DICE+1, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Liar's Dice specific ---
    _first_player: jnp.ndarray = jnp.int8(0)
    #Liar action 
    _liar: jnp.ndarray = jnp.int8(0)
    #Last bet
    _last_bet: jnp.ndarray = jnp.int8(0)
    #Dices outcome: num_player * dice_sides
    _dices: jnp.ndarray = jnp.ones((2, DICE_SIDES),dtype=jnp.int8)
    #Number of dice of each player: num_player
    _num_dice: jnp.ndarray = jnp.ones((2,),dtype=jnp.int8)
    #Sequence of possible bets 
    _bets: jnp.ndarray = jnp.zeros(DICE_SIDES*2*MAX_NUM_DICE+1, dtype=jnp.bool_)

    @property
    def env_id(self) -> str:
        return "ful_liars_dice"


class FullLiarsDice(ggame.Game):
    def __init__(self, max_num_dice=5, dice_sides=6, wilde=6):
        super().__init__()
        self.max_num_dice = max_num_dice
        self.dice_sides = dice_sides
        self.wilde = wilde

    def _init(self, rng: jax.random.KeyArray) -> State:
        return partial(
            _init,
            max_num_dice=self.max_num_dice, 
            dice_sides=self.dice_sides
            )(rng)

    def _step(self, state: State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return partial(
            _step,
            max_num_dice=self.max_num_dice,
            dice_sides=self.dice_sides,
            wilde = self.wilde
            )(state, action)

    def _observe(self, state: State, player_id: jnp.ndarray) -> jnp.ndarray:
        assert isinstance(state, State)
        return partial(
            _observe,
            max_num_dice=self.max_num_dice,
            dice_sides=self.dice_sides
            )(state, player_id)
    
    def full_init(self, key:jax.random.KeyArray, num_dice=None) -> State:
        """Initialize a state with maximum number of dice for each player
        Not jitable a priori :/.
        """
        key, subkey = jax.random.split(key)
        if num_dice is None:
            num_dice = self.max_num_dice
        state = _init(
            subkey,
            max_num_dice=self.max_num_dice,
            dice_sides=self.dice_sides,
            init_num_dice=jnp.array([num_dice, num_dice])
        )  
        state = state.replace(_rng_key=key)  
        observation = self.observe(state, state.current_player)
        return state.replace(observation=observation)

    def full_step(self, state: State, action: jnp.ndarray) -> State:
        """Full step with reset and update of the number of dice when
        a round ends. Not jitable a priori :/.
        """
        state = self.step(state, action)
        new_num_dice = jax.lax.select(
            state.rewards[0] < 0.0, 
            state._num_dice,
            jnp.flip(state._num_dice)
        ).at[0].add(-1)
        rewards= state.rewards
        terminated = jnp.any(new_num_dice <= 0, axis=-1)    
        if (state.terminated & (~terminated)).item():
            state = _init(
                state._rng_key,
                max_num_dice=self.max_num_dice,
                dice_sides=self.dice_sides,
                init_num_dice=new_num_dice) 
            observation = self.observe(state, state.current_player)
            return state.replace(observation=observation,rewards=rewards)
        return state

    @property
    def id(self) -> str:
        return "full_liars_dice"

    @property
    def num_players(self) -> int:
        return 2
    
    @property
    def max_game_length(self) -> int:
        return 2 * self.max_num_dice * self.dice_sides + 1

    @property
    def min_utility(self) -> float:
        return MIN_UTILITY

    @property
    def max_utility(self) -> float:
        return MAX_UTILITY

def _init(
        rng: jax.random.KeyArray, 
        max_num_dice=5, 
        dice_sides=6,
        init_num_dice=None,
    ) -> State:
    
    #Generate numbe of dice
    _rng, rng = jax.random.split(rng)
    num_dice = jax.random.choice(
        _rng,
        jnp.arange(1, max_num_dice+1),
        shape=(2,),
    )
    if init_num_dice is not None:
        num_dice = init_num_dice
    #Dice mask
    mask_dice = num_dice[:,None] > jnp.arange(max_num_dice)[None,:]
    #Generate private dice
    _rng, rng = jax.random.split(rng)
    dices = jax.random.choice(
        _rng,
        dice_sides,
        shape=(2, max_num_dice),
    )
    #Remove and sort
    dices = jnp.where(mask_dice, dices, dice_sides)
    #dices = jnp.sort(dices, axis=-1)
    #From dice to counts
    _dices = jnp.zeros((2, dice_sides+1), dtype=jnp.int8)
    for i in range(2):
        _dices = _dices.at[i,dices[i,:]].add(1)
    #_dices = jax.nn.one_hot(_dices, max_num_dice, dtype=jnp.int8)
    dices = _dices
    #Legal action mask
    max_num_actions = 2 * max_num_dice * dice_sides + 1
    num_action = num_dice.sum() * dice_sides + 1
    liar = num_action-1  #Liar action
    legal = jnp.arange(max_num_actions) < liar #Cannot call liar first round
    return State(
        _rng_key=rng,
        _num_dice=num_dice,
        _dices=dices,
        _liar=liar,
        legal_action_mask=legal,
        _bets=jnp.zeros(max_num_actions, jnp.bool_)
    )

def _step(state: State, action, max_num_dice=5, dice_sides=6, wilde=6) -> State:
    action = jnp.int8(action)
    max_num_actions = 2 * max_num_dice * dice_sides + 1
    liar = state._liar
    #Game ends if one player calls liar
    terminated = (action >= liar)
    #All bet above current one and below liar are possible
    legal = (jnp.arange(max_num_actions) > action) & (jnp.arange(max_num_actions) <= liar)
    #Record the bet
    _bets = state._bets.at[action].set(True)
    #Record last action 
    _last_bet = action
    #Get reward 
    rewards = jax.lax.select(
        terminated,
        _get_rewards(state, dice_sides, wilde),
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


def _get_rewards(state: State, dice_sides=6, wilde=6):
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
    #Decode the bet from bet id
    bet = state._last_bet
    bet_dice = bet // dice_sides + 1
    bet_side = bet % dice_sides
    #Count the number of match, 0 is wild so always a match
    wilde = jnp.int8(wilde-1)
    num_match = state._dices[:, bet_side] + (bet_side != wilde) * state._dices[:, wilde]
    num_match = num_match.sum()
    #Check if the bet is correct
    bet_correct = (num_match >= bet_dice) 
    #Current player call liar so loose if bet correct
    rewards = jax.lax.select(
        bet_correct,
        jnp.float32([-1, -1]).at[1 - state.current_player].set(1),
        jnp.float32([-1, -1]).at[state.current_player].set(1),
    )
    return rewards

def _observe(state: State, player_id, max_num_dice=5, dice_sides=6) -> jnp.ndarray:
    """
    Size                            Meaning
    2                               current player id
    max_num_dice                    num dice player 1
    max_num_dice                    num dice player 1
    (dice_sides+1)*(max_num_dice+1) private dices counts 
    2*max_num_dice*dice_sides+1     betting sequence 
    """
    max_num_actions = 2 * max_num_dice * dice_sides + 1

    obs = jnp.zeros(
        2+2*max_num_dice+(dice_sides+1)*max_num_dice+max_num_actions,
        dtype=jnp.bool_)
    obs = obs.at[player_id].set(TRUE)
    num_dice_0 = jax.nn.one_hot(state._num_dice[0]-1, max_num_dice, dtype=jnp.bool_) 
    obs = obs.at[2:2+max_num_dice].set(num_dice_0)
    num_dice_1 = jax.nn.one_hot(state._num_dice[1]-1, max_num_dice, dtype=jnp.bool_) 
    obs = obs.at[2+max_num_dice:2+2*max_num_dice].set(num_dice_1)
    dices = state._dices.at[state.current_player].get()
    dices = jax.nn.one_hot(dices, max_num_dice+1, dtype=jnp.bool_)
    dices = jnp.reshape(dices, (-1,))
    idx = 2 + 2 * max_num_dice
    obs = obs.at[idx:idx+(dice_sides+1)*(max_num_dice+1)].set(dices)
    idx = idx + (dice_sides+1) * max_num_dice
    obs = obs.at[idx:].set(state._bets)
    return obs





# def unif_logit(observation, legal):
#     return jnp.where(legal, 1.0, -jnp.inf)

class HumanVsBot:
    """ A class to play a game again a bot define by the
    function `get_bot_logit`
    """
    def __init__(
        self,
        get_bot_logit,
        max_num_dice=5,
        wilde=1,
        seed=13,
        ):

        #Rng key
        self._key = jax.random.PRNGKey(seed)

        #Game
        self.game = FullLiarsDice(
            max_num_dice=max_num_dice,
            dice_sides=6,
            wilde=wilde,
            )

        #Bot
        self.get_bot_logit = get_bot_logit

        #Player idx
        self.human_idx = 1
        self.bot_idx = 0

        #To print dice and stuff
        # from https://stackoverflow.com/a/52672324
        s ="+ - - - - +"
        m1="|  o   o  |"
        m2="|  o      |"
        m3="|    o    |"
        m4="|      o  |"
        m5="|         |"
        m6="|    ?    |"

        DICE = [
            [m5, m6, m5],
            [m5, m3, m5],
            [m2, m5, m4],
            [m2, m3, m4],
            [m1, m5, m1],
            [m1, m3, m1],
            [m1, m1, m1]
            ]
        def die(i):
            return [s, *DICE[i], s]

        def bet(i):
            row = "     " + str(i) + "   * "
            if i > 9:
                row = "     " + str(i) + "  * "
            v = "           "
            return [v,v,row,v,v]

        def join_row(*rows):
            return ['   '.join(r) for r in zip(*rows)]

        def print_dice(dice):
            for line in join_row(*map(die, dice)):
                print(line)
        self._print_dice = print_dice

        def print_bet(i,j):
            for line in join_row(*[bet(i),die(j)]):
                print(line)
        self._print_bet = print_bet

    def _next_rng_key(self):
        """Get the next rng subkey from class rngkey.
        Must *not* be called from under a jitted function!
        Returns:
            A fresh rng_key.
        """
        self._key, key = jax.random.split(self._key)
        return key


    def print_dice_player(self, state, player_idx, side=None):
        """Print dice of player `player_idx` """
        dice_count = state._dices[player_idx]
        dice_count = dice_count.at[:-1].get()
        dice_sides = self.game.dice_sides
        dice = jnp.repeat(jnp.arange(dice_sides), dice_count)
        dice = jnp.sort(dice, axis=-1) + 1
        self._print_dice(list(dice))
        if side is not None:
            dice_match = (dice == side) | (dice == self.game.wilde)
            match_line = ["***********" if m else "           " for m in dice_match]
            print("   ".join(match_line))


    def print_line(self):
            print("---".join(["-----------" for _ in range(self.game.max_num_dice)]))

    def print_dice(self, state, side=None):
        print("Your dice:")
        self.print_dice_player(state, self.human_idx, side)
        print("Bot dice:")
        if side is None:
            self._print_dice(state._num_dice[self.bot_idx].item()*[0])
        else:
            self.print_dice_player(state, self.bot_idx, side)

    def print_action(self, action):
        if action == self.liar:
            print("Liar!")
        else:
            bet = self.action_to_bet(action)
            self._print_bet(*bet)

    def bet_to_action(self, bet):
        return (bet[0]-1) * self.game.dice_sides + (bet[1]-1)

    def action_to_bet(self, bet):
        bet_dice = bet // self.game.dice_sides + 1
        bet_side = bet % self.game.dice_sides + 1
        return [bet_dice, bet_side]

    def get_human_action(self):
        """Ask the action to the player"""
        print()
        bet = input("Your bet ? ")
        if "liar" in bet:
            return self.liar
        b = [int(x) for x in bet.split(" ")][:2]
        return self.bet_to_action(b)

    def get_bot_action(self, observation, legal):
        """Sample action for the bot"""
        #Compute logit
        logit = self.get_bot_logit(observation, legal)
        #Sample action
        legal_logit = jnp.where(legal, logit, -jnp.inf)
        action = jax.random.categorical(
            self._next_rng_key(),
            legal_logit,
            axis=-1
            )
        return action

    def play(self):
        
        #Sample player order
        self.human_idx = jax.random.choice(self._next_rng_key(),2).item()
        self.bot_idx = (1 - self.human_idx) % 2

        #Init game
        state = self.game.full_init(self._next_rng_key())

        #Time in the game
        start_round = True

        #Liar
        action = -1
        human_win = True
        self.liar = state._liar

        while not state.terminated:
            print()
            if start_round:
                #Print private dice at the beginning of a round
                self.print_dice(state)
                print()
                start_round = False
            #Select action
            _action = action
            if state.current_player == self.human_idx:
                action = self.get_human_action()
            else:
                action = self.get_bot_action(
                    state.observation,
                    state.legal_action_mask
                )
                print("Bot plays:")
            #Print action
            self.print_action(action)
            #Play action
            _state = state
            state = self.game.full_step(state, action)
            #Check if end of round
            if action == self.liar:
                print()
                #Reveal dice
                self.print_dice(_state, self.action_to_bet(_action)[1])
                print()
                if state.rewards[self.human_idx] < 0:
                    print("You lose one die :/")
                    #Loser starts
                    self.human_idx = 0
                    self.bot_idx = 1
                    human_win = False
                else:
                    print("Bot lose one die :)")
                    #Loser starts
                    self.human_idx = 1
                    self.bot_idx = 0
                    human_win = True
                #Update liar
                start_round = True
                self.liar = state._liar
                input("Press Enter to continue...")
                print()
                self.print_line()
                print()
        if human_win:
            print()
            print("You win! :)")
        else:
            print()
            print("You lose! :/")



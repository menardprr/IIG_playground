# Network
import haiku as hk
import optax


def optimizer_factory(optimizer_type, optimizer_kwargs):
    """A factory of optimizer"""
    if optimizer_type == "adam":
        return optax.adam(**optimizer_kwargs)
    elif optimizer_type == "sgd":
        return optax.sgd(**optimizer_kwargs)
    else:
        raise NotImplementedError()


def apply_transform_net(net_module, network_kwargs):
    """Init a module network with kwargs"""
    def _net(info_input):
        module = net_module(**network_kwargs)
        return module(info_input)
    return hk.without_apply_rng(hk.transform(_net))


def network_factory(network_type, network_kwargs):
    """A factory of network"""
    if network_type == "PolicyNet":
        return apply_transform_net(PolicyNet, network_kwargs)
    elif network_type == "ValueNet":
        return apply_transform_net(ValueNet, network_kwargs)
    else:
        raise NotImplementedError()


class PolicyNet(hk.Module):
    """A simple network with a policy head"""

    def __init__(self, num_actions, hidden_layers_sizes):
        super().__init__()
        self._num_actions = num_actions
        self._hidden_layers_sizes = hidden_layers_sizes

    def __call__(self, info_state):
        """Process a batch of observations."""
        torso = hk.nets.MLP(self._hidden_layers_sizes, activate_final=True)
        hidden = torso(info_state)
        policy_logit = hk.Linear(self._num_actions)(hidden)
        return policy_logit


class ValueNet(hk.Module):
    """A simple network with a value head"""

    def __init__(self, hidden_layers_sizes):
        super().__init__()
        self._hidden_layers_sizes = hidden_layers_sizes

    def __call__(self, info_state):
        """Process a batch of observations."""
        torso = hk.nets.MLP(self._hidden_layers_sizes, activate_final=True)
        hidden = torso(info_state)
        value = hk.Linear(1)(hidden)
        return value.squeeze(-1)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional


# -------------------- Neural Network Modules --------------------
def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        fc_dims: List[int],
        log_std_init: float = -0.5,
        activation_fn: nn.Module = nn.Tanh,
    ) -> None:
        super(Actor, self).__init__()

        self.activation_fn = activation_fn

        layers = []
        in_features = input_dim
        for out_features in fc_dims:
            layers.append(layer_init(nn.Linear(in_features, out_features)))
            layers.append(self.activation_fn())
            in_features = out_features
        layers.append(layer_init(nn.Linear(in_features, output_dim), std=0.01))
        # layers.append(nn.Tanh())

        self.mean = nn.Sequential(*layers)
        self.logstd = nn.Parameter(
            torch.ones(output_dim) * log_std_init, requires_grad=True
        )

    def forward(
        self,
        state: torch.Tensor,
        params: Optional[Dict] = None,
        prefix: str = "actor",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action mean and log standard deviation.
        If params is provided, it should contain 'mean.<layer>' and 'logstd' keys.
        """
        if params is None:
            action_mean = self.mean(state)
            action_std = self.logstd.exp().expand_as(action_mean)
        else:
            # Functional forward: iterate over mean layers
            action_mean = self._functional_sequential(
                self.mean, state, params, prefix=f"{prefix}.mean"
            )
            action_std = params[f"{prefix}.logstd"].exp().expand_as(action_mean)
        return action_mean, action_std

    def _functional_sequential(
        self, module: nn.Sequential, x: torch.Tensor, params: Dict, prefix: str
    ) -> torch.Tensor:
        """Apply a Sequential module using parameters from the dictionary."""
        for idx, submodule in enumerate(module):
            if isinstance(submodule, nn.Linear):
                w_key = f"{prefix}.{idx}.weight"
                b_key = f"{prefix}.{idx}.bias"
                x = F.linear(x, params[w_key], params[b_key])
            elif isinstance(submodule, self.activation_fn):
                # x = self.activation_fn(x)
                x = submodule(x)
            else:
                raise TypeError(f"Unsupported layer type: {type(submodule)}")
        return x


class Critic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        fc_dims: List[int],
        activation_fn: nn.Module = nn.Tanh,
    ) -> None:
        super(Critic, self).__init__()

        self.activation_fn = activation_fn

        layers = []
        in_features = input_dim
        for out_features in fc_dims:
            layers.append(layer_init(nn.Linear(in_features, out_features)))
            layers.append(self.activation_fn())
            in_features = out_features
        layers.append(layer_init(nn.Linear(in_features, 1), std=1.0))

        self.critic = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        params: Optional[Dict] = None,
        prefix: str = "critic",
    ) -> torch.Tensor:
        if params is None:
            value = self.critic(state)
        else:
            value = self._functional_sequential(
                self.critic, state, params, prefix=f"{prefix}.critic"
            )
        return value

    def _functional_sequential(
        self, module: nn.Sequential, x: torch.Tensor, params: Dict, prefix: str
    ) -> torch.Tensor:
        for idx, submodule in enumerate(module):
            if isinstance(submodule, nn.Linear):
                w_key = f"{prefix}.{idx}.weight"
                b_key = f"{prefix}.{idx}.bias"
                x = F.linear(x, params[w_key], params[b_key])
            elif isinstance(submodule, self.activation_fn):
                # x = self.activation_fn(x)
                x = submodule(x)
            else:
                raise TypeError(f"Unsupported layer type: {type(submodule)}")
        return x


class ActorCriticNetwork(nn.Module):
    """Shared MLP for policy (Gaussian) and value (critic) functions.
    Supports functional parameter passing for MAML inner loops.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        policy_kwargs: dict[str, List[int]] = {
            "feature": [],
            "pi": [64, 64],
            "vf": [64, 64],
            "activation_fn": nn.Tanh,
        },
    ):
        super().__init__()
        self.action_dim = action_dim
        feature_fc_dims = policy_kwargs["feature"]
        actor_fc_dims = policy_kwargs["pi"]
        critic_fc_dims = policy_kwargs["vf"]
        self.activation_fn = policy_kwargs["activation_fn"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Shared feature extraction layers
        if len(feature_fc_dims) != 0:
            layers = []
            in_features = state_dim
            for out_features in feature_fc_dims:
                layers.append(layer_init(nn.Linear(in_features, out_features)))
                layers.append(self.activation_fn())
                in_features = out_features
            self.feature_extractor = nn.Sequential(*layers)
            input_dim = feature_fc_dims[-1]
        else:
            self.feature_extractor = nn.Flatten()
            input_dim = state_dim

        # Actor head
        self.actor = Actor(
            input_dim, action_dim, actor_fc_dims, activation_fn=self.activation_fn
        )

        # Critic head
        self.critic = Critic(
            input_dim, critic_fc_dims, activation_fn=self.activation_fn
        )

        self.to(self.device)

    def forward(
        self,
        observation: np.ndarray,
        params: Optional[Dict] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional alternative parameters.
        Returns:
            action_mean: torch.Tensor (batch_size, action_dim)
            action_logstd: torch.Tensor (action_dim,)
            value: torch.Tensor (batch_size,)
        """
        actions, logprobs = self.get_action(observation, params, deterministic)
        values = self.get_value(observation, params)

        return actions, values, logprobs

    def get_action(
        self,
        observation: np.ndarray,
        params: Optional[Dict] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        state = state.unsqueeze(0) if state.dim() == 1 else state
        features = self._extract_features(state, params)

        action_mean, action_std = self.actor(features, params, "actor")
        dist = Normal(action_mean, action_std)

        if deterministic:
            actions = dist.mean
        else:
            actions = dist.rsample()
        logprobs = dist.log_prob(actions).sum(-1)

        return actions, logprobs

    def get_value(
        self, observation: np.ndarray, params: Optional[Dict] = None
    ) -> torch.Tensor:
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        state = state.unsqueeze(0) if state.dim() == 1 else state
        features = self._extract_features(state, params)

        values = self.critic(features, params, "critic")

        return values

    def evaluate_action(
        self, observation: np.ndarray, action: np.ndarray, params: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (values, logprobs, entropy)
        """
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        state = state.unsqueeze(0) if state.dim() == 1 else state
        features = self._extract_features(state, params)

        action_mean, action_std = self.actor(features, params, "actor")
        dist = Normal(action_mean, action_std)

        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        logprobs = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        values = self.critic(features, params, "critic")

        return values, logprobs, entropy

    def _extract_features(
        self, x: torch.Tensor, params: Optional[Dict] = None
    ) -> torch.Tensor:
        """Apply feature extractor, optionally with alternative parameters."""
        if params is None or isinstance(self.feature_extractor, nn.Flatten):
            return self.feature_extractor(x)
        else:
            # Functional forward through the feature extractor (Linear + Tanh)
            for idx, submodule in enumerate(self.feature_extractor):
                if isinstance(submodule, nn.Linear):
                    w_key = f"feature_extractor.{idx}.weight"
                    b_key = f"feature_extractor.{idx}.bias"
                    x = F.linear(x, params[w_key], params[b_key])
                elif isinstance(submodule, self.activation_fn):
                    x = submodule(x)
                else:
                    raise TypeError(
                        f"Unsupported layer in feature_extractor: {type(submodule)}"
                    )
            return x

    def get_actor_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """Return only Actor parameters"""
        params = OrderedDict()
        for name, param in self.named_parameters():
            # Include only actor and feature extractor, exclude value
            if "critic" not in name:
                params[name] = param
        return params

    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """Return a flat dictionary of all parameters (names -> tensors)."""
        return OrderedDict({name: param for name, param in self.named_parameters()})

    def load_parameters_dict(self, params: Dict[str, torch.Tensor]) -> None:
        """Load parameters from a dictionary (for cloning or evaluation)."""
        own_state = self.state_dict()
        for name, param in params.items():
            if name in own_state:
                own_state[name].copy_(param)


class PolicyNetwork(nn.Module):
    """Policy Network (Gaussian).
    Supports functional parameter passing for MAML inner loops.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        policy_kwargs: dict[str, List[int]] = {
            "feature": [],
            "pi": [64, 64],
            "vf": [64, 64],
            "activation_fn": nn.Tanh,
        },
    ):
        super().__init__()

        self.action_dim = action_dim
        actor_fc_dims = policy_kwargs["pi"]
        self.activation_fn = policy_kwargs["activation_fn"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = state_dim

        # Actor head
        self.actor = Actor(
            input_dim, action_dim, actor_fc_dims, activation_fn=self.activation_fn
        )

        self.to(self.device)

    def forward(
        self,
        observation: np.ndarray,
        params: Optional[Dict] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional alternative parameters.
        Returns:
            action_mean: torch.Tensor (batch_size, action_dim)
            action_logstd: torch.Tensor (action_dim,)
            value: torch.Tensor (batch_size,)
        """
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        state = state.unsqueeze(0) if state.dim() == 1 else state

        action_mean, action_std = self.actor(state, params, "actor")
        dist = Normal(action_mean, action_std)

        if deterministic:
            actions = dist.mean
        else:
            actions = dist.rsample()
        logprobs = dist.log_prob(actions).sum(-1)

        return actions, logprobs

    def evaluate_action(
        self, observation: np.ndarray, action: np.ndarray, params: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (values, logprobs, entropy)
        """
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        state = state.unsqueeze(0) if state.dim() == 1 else state

        action_mean, action_std = self.actor(state, params, "actor")
        dist = Normal(action_mean, action_std)

        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        logprobs = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return logprobs, entropy

    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """Return a flat dictionary of all parameters (names -> tensors)."""
        return OrderedDict({name: param for name, param in self.named_parameters()})

    def load_parameters_dict(self, params: Dict[str, torch.Tensor]) -> None:
        """Load parameters from a dictionary (for cloning or evaluation)."""
        own_state = self.state_dict()
        for name, param in params.items():
            if name in own_state:
                own_state[name].copy_(param)


class ValueNetwork(nn.Module):
    """Value (critic) Network.
    Supports functional parameters.
    """

    def __init__(
        self,
        state_dim: int,
        policy_kwargs: dict[str, List[int]] = {
            "feature": [],
            "pi": [64, 64],
            "vf": [64, 64],
            "activation_fn": nn.Tanh,
        },
    ):
        super().__init__()
        critic_fc_dims = policy_kwargs["vf"]
        self.activation_fn = policy_kwargs["activation_fn"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Critic head
        self.critic = Critic(
            state_dim, critic_fc_dims, activation_fn=self.activation_fn
        )

        self.to(self.device)

    def forward(
        self,
        observation: np.ndarray,
        params: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional alternative parameters.
        Returns:
            action_mean: torch.Tensor (batch_size, action_dim)
            action_logstd: torch.Tensor (action_dim,)
            value: torch.Tensor (batch_size,)
        """
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        state = state.unsqueeze(0) if state.dim() == 1 else state

        values = self.critic(state, params, "critic")

        return values

    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """Return a flat dictionary of all parameters (names -> tensors)."""
        return OrderedDict({name: param for name, param in self.named_parameters()})

    def load_parameters_dict(self, params: Dict[str, torch.Tensor]) -> None:
        """Load parameters from a dictionary (for cloning or evaluation)."""
        own_state = self.state_dict()
        for name, param in params.items():
            if name in own_state:
                own_state[name].copy_(param)


class LinearValueNetwork(nn.Module):
    """
    ## Description

    A linear state-value function, whose parameters are found by minimizing
    least-squares.
    Linear baseline based on handcrafted features, as described in [1]
    (Supplementary Material 2).

    ## Credit

    Adapted from Tristan Deleu's implementation.

    ## References

    [1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel,
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016
        (https://arxiv.org/abs/1604.06778)

    """

    def __init__(self, input_size, reg=1e-5):
        """
        ## Arguments

        * `inputs_size` (int) - Size of input.
        * `reg` (float, *optional*, default=1e-5) - Regularization coefficient.
        """
        super().__init__()

        self.linear = nn.Linear(2 * input_size + 4, 1, bias=False)
        self.reg = reg
        self.device = torch.device("cpu")

    def _features(self, states):
        length = states.size(0)
        ones = torch.ones(length, 1).to(states.device)
        al = (
            torch.arange(length, dtype=torch.float32, device=states.device).view(-1, 1)
            / 100.0
        )
        return torch.cat([states, states**2, al, al**2, al**3, ones], dim=1)

    def fit(self, states, returns):
        """
        ## Description

        Fits the parameters of the linear model by the method of least-squares.

        ## Arguments

        * `states` (tensor) - States collected with the policy to evaluate.
        * `returns` (tensor) - Returns associated with those states (ie, discounted rewards).
        """
        features = self._features(states)
        reg = self.reg * torch.eye(features.size(1))
        reg = reg.to(states.device)
        A = features.t() @ features + reg
        b = features.t() @ returns
        if hasattr(torch, "linalg") and hasattr(torch.linalg, "lstsq"):
            coeffs = torch.linalg.lstsq(A, b).solution
        else:
            raise NotImplementedError()

        self.linear.weight.data = coeffs.data.t()

    def forward(self, state):
        """
        ## Description

        Computes the value of a state using the linear function approximator.

        ## Arguments

        * `state` (Tensor) - The state to evaluate.
        """
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        features = self._features(state)
        return self.linear(features)

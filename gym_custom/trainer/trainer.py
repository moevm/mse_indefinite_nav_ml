from typing import Type
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.utils.annotations import override
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.utils.typing import TrainerConfigDict
from gym_custom.policy.policy import CustomPPOTorchPolicy


class CustomPPOTrainer(PPOTrainer):
    @override(PPOTrainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        return CustomPPOTorchPolicy

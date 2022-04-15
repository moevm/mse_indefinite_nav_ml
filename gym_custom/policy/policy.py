import torch
import numpy as np
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.annotations import override
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, \
    Union, TYPE_CHECKING
from ray.rllib.utils.typing import GradInfoDict, ModelGradients, \
    ModelWeights, TensorType, TensorStructType, TrainerConfigDict
if TYPE_CHECKING:
    from ray.rllib.evaluation import Episode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

class CustomPPOTorchPolicy(PPOTorchPolicy):
    @override(PPOTorchPolicy)
    def compute_actions(
            self,
            obs_batch: Union[List[TensorStructType], TensorStructType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Union[List[TensorStructType],
                                     TensorStructType] = None,
            prev_reward_batch: Union[List[TensorStructType],
                                     TensorStructType] = None,
            info_batch: Optional[Dict[str, list]] = None,
            episodes: Optional[List["Episode"]] = None,
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            **kwargs) -> \
            Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:

        with torch.no_grad():
            seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)
            input_dict = self._lazy_tensor_dict({
                SampleBatch.CUR_OBS: obs_batch,
                "is_training": False,
            })
            if prev_action_batch is not None:
                input_dict[SampleBatch.PREV_ACTIONS] = \
                    np.asarray(prev_action_batch)
            if prev_reward_batch is not None:
                input_dict[SampleBatch.PREV_REWARDS] = \
                    np.asarray(prev_reward_batch)
            state_batches = [
                convert_to_torch_tensor(s, self.device)
                for s in (state_batches or [])
            ]
            input_dict[SampleBatch.INFOS] = np.asarray(info_batch)
            return self._compute_action_helper(input_dict, state_batches,
                                               seq_lens, explore, timestep)

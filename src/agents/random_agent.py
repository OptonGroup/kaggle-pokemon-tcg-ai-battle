"""Random agent — competition baseline."""
import random
from cg.api import Observation, to_observation_class


def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        # initial deck selection — handled by caller
        return []
    return random.sample(range(len(obs.select.option)), obs.select.maxCount)

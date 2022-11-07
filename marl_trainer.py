import pandas as pd
import matplotlib.pyplot as plt

from environment import MultiAgentSfcPartitioningEnv
from common.resources import Topology
from common.datasets import dummy_payload

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.qmix import QMixConfig

from gym.spaces import Tuple

# create the SFC dataset
dataset = dummy_payload(n_sfcs=10000, min_n_vnfs=2, max_n_vnfs=5)

# create the topology
topology = Topology('MESH_LARGE')


if __name__ == '__main__':
    # Create and register the environment.
    def env_creator(config):
        return MultiAgentSfcPartitioningEnv(env_config=config)
    env_config = {'topology': topology, 'dataset': dataset, "disable_env_checking": True}
    env_name = "MultiAgentSfcPartitioningEnv"
    register_env(env_name, env_creator)

    env = MultiAgentSfcPartitioningEnv(env_config=env_config)

    ray.init()

    grouping = {"group_1": list(range(len(topology.G.nodes)))}
    obs_space = Tuple([env.observation_space[0] for _ in range(10)])
    act_space = Tuple([env.action_space[0] for _ in range(10)])

    register_env(env_name,
                 lambda config: MultiAgentSfcPartitioningEnv(config).with_agent_groups(grouping,
                                                                                       obs_space=obs_space,
                                                                                       act_space=act_space
                                                                                       )
                 )

    qmix_config = (
        QMixConfig()
        .framework(framework="torch")
        .environment(env=env_name,
                     env_config=env_config,
                     disable_env_checking=True)
        .rollouts(num_envs_per_worker=1)
    )
    qmix_config.simple_optimizer = True  # Avoid GPU engagement.
    trainer = qmix_config.build()

    for _ in range(10):
        results = trainer.train()

    rewards_history = pd.DataFrame(results['hist_stats']['episode_reward'])
    rewards_history.plot()
    plt.show()

    # trainer.train()["episode_reward_mean"]
    trainer.stop()
    ray.shutdown()

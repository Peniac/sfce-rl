import pandas as pd
import matplotlib.pyplot as plt

from environment import MultiAgentSfcPartitioningEnv
from common.resources import Topology
from common.datasets import dummy_payload
from utils import save_object

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.qmix import QMixConfig

from gym.spaces import Tuple

# create the SFC dataset
dataset = dummy_payload(n_sfcs=4, min_n_vnfs=2, max_n_vnfs=5)

# create the topology
topology = Topology('MESH_LARGE')
# topology = Topology('MESH_THREE')


if __name__ == '__main__':

    env_config = {'topology': topology,
                  'dataset': dataset,
                  'disable_env_checking': True,
                  'agents': [idx for idx, pop in enumerate(topology.G.nodes)]}

    env_name = "MultiAgentSfcPartitioningEnv"

    ray.init()

    env = MultiAgentSfcPartitioningEnv(env_config=env_config)

    grouping = {"group_1": list(range(len(topology.G.nodes)))}
    obs_space = Tuple([env.observation_space[0] for _ in range(len(topology.G.nodes))])
    act_space = Tuple([env.action_space[0] for _ in range(len(topology.G.nodes))])

    register_env(env_name,
                 lambda config: MultiAgentSfcPartitioningEnv(config)
                 .with_agent_groups(grouping,
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

    # SFCs # Epsilon timesteps
    # 2    # 1000
    # 3    # > 8000
    # 4    #
    qmix_config.exploration_config['epsilon_timesteps'] = 8000
    qmix_config.exploration_config['final_epsilon'] = 0.00
    qmix_config.simple_optimizer = True  # Avoid GPU engagement.
    qmix_config.model['lstm_cell_size'] = 16

    trainer = qmix_config.build()

    mean_rewards = {}
    for j in range(10):
        # approx 250 episodes and 1000 time-steps per training iteration
        results = trainer.train()
        print(f'Iteration {j+1}')
        episodes_total = results['episodes_total']
        print(f'Episodes total: {episodes_total}')
        episodes_reward_mean = results['episode_reward_mean']
        print(f'Episode reward mean {episodes_reward_mean}')
        print(trainer.get_policy().get_exploration_state())
        print('\n')
        mean_rewards[j+1] = results['episode_reward_mean']


    # rewards_history = pd.DataFrame(results['hist_stats']['episode_reward'])
    # rewards_history.plot()
    # plt.show()

    mean_rewards = pd.Series(mean_rewards, name='episode reward mean')
    save_object(mean_rewards, 'mean_rewards_'+str(len(dataset.payload)))
    mean_rewards.plot()
    plt.show()

    trainer.stop()
    ray.shutdown()

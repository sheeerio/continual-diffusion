import gym
import numpy as np
import torch
from pyvirtualdisplay import Display
import zarr
from ding.policy import create_policy
from ding.envs import DingEnvWrapper
from ding.entry import serial_pipeline
import os
from easydict import EasyDict


virtual_display = Display(visible=0, size=(1024, 768))
virtual_display.start()

output_path = "nmaze.zarr"
num_episodes = 250  
max_steps_per_episode = 700 

coinrun_dqn_config = EasyDict(
    dict(
        exp_name='coinrun_dqn_experiment',
        env=dict(
            env_id='coinrun',
            collector_env_num=4,
            evaluator_env_num=4,
            n_evaluator_episode=4,
            stop_value=10,
        ),
        policy=dict(
            type='dqn',          
            on_policy=False,      
            cuda=True,
            model=dict(
                obs_shape=[3, 64, 64],
                action_shape=15,
                encoder_hidden_size_list=[128, 128, 512],
                dueling=False,
            ),
            discount_factor=0.99,
            learn=dict(
                update_per_collect=20,
                batch_size=32,
                learning_rate=0.0005,
                target_update_freq=500,
                # ======== Path to the pretrained checkpoint (ckpt) ========
                learner=dict(
                    hook=dict(
                        load_ckpt_before_run='./iteration_130000.pth.tar'
                    )
                ),
                resume_training=True,
            ),
            collect=dict(n_sample=100,),
            eval=dict(evaluator=dict(eval_freq=5000,)),
            other=dict(
                eps=dict(
                    type='exp',
                    start=1.0,
                    end=0.05,
                    decay=250000,
                ),
                replay_buffer=dict(replay_buffer_size=100000,),
            ),
        ),
    )
)

coinrun_dqn_create_config = EasyDict(
    dict(
        env=dict(
            type='procgen',
            import_names=['dizoo.procgen.envs.procgen_env'],
        ),
        env_manager=dict(type='subprocess',),
        policy=dict(type='dqn'), 
        log=dict(
            type='tensorboard',
            exp_name='coinrun_dqn_experiment',
        ),
    )
)

main_config = coinrun_dqn_config
create_config = coinrun_dqn_create_config

def train_dqn_policy():
    print("Training DQN policy...")
    serial_pipeline((main_config, create_config), seed=0)
    model_path = "./iteration_130000.pth.tar" 
    return model_path

pretrained_checkpoint_path = "/content/iteration_130000.pth.tar"

file_exists = os.path.exists(pretrained_checkpoint_path)
if file_exists:
    print("DQN policy already trained. Skipping training.")
    trained_model_path = pretrained_checkpoint_path
else:
    trained_model_path = train_dqn_policy()

print("Loading the trained DQN policy...")
policy = create_policy(main_config.policy, enable_field=['eval'])

checkpoint = torch.load(trained_model_path, map_location='cpu')
policy.eval_mode.load_state_dict(checkpoint)

# policy.eval_mode.eval()

env = gym.make(
    'procgen:procgen-coinrun-v0',
    start_level=0,
    num_levels=1,
    render_mode='rgb_array',
    # new_step_api=True
)
env = DingEnvWrapper(env)

observations = []
actions = []
episode_ends = []

action_space = env.action_space
action_dim = action_space.n 

print("Generating Maze dataset using trained DQN policy...")
for episode in range(num_episodes):
    total = 0
    observation = env.reset()
    for step in range(max_steps_per_episode):
        with torch.no_grad():
            obs_tensor = torch.tensor(observation).permute(2, 0, 1).float()

            action_output = policy.eval_mode.forward({'obs': obs_tensor})

            action = int(action_output['obs']['action'].cpu().numpy()[0])

        action_one_hot = np.zeros(action_dim, dtype=np.float32)
        action_one_hot[action] = 1.0 

        observation_, reward, terminated, *_ = env.step(action)
        total += reward

        observations.append(observation)
        actions.append(action_one_hot)

        observation = observation_

        if terminated:
          print(step)
          break
    print(f"Episode {episode + 1}, Total Reward: {total}")

    episode_ends.append(len(observations))

images = np.array(observations, dtype=np.uint8)  # Shape: (N, 64, 64, 3)
actions = np.array(actions, dtype=np.float32)    # Shape: (N,)
episode_ends = np.array(episode_ends, dtype=np.int32)  

print("Saving dataset to Zarr format...")
root = zarr.open(output_path, mode='w')
root.create_dataset('images', data=images, chunks=(1000, 64, 64, 3), dtype=np.uint8)
root.create_dataset('actions', data=actions, chunks=(1000, 15), dtype=np.float32)

meta = root.create_group('meta')
meta.create_dataset('episode_ends', data=episode_ends, dtype=np.int32)

print(f"Dataset saved at {output_path}.")
env.close()

virtual_display.stop()
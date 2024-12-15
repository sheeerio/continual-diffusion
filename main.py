import torch
import torch.nn as nn
import gym
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm
from pyvirtualdisplay import Display
from diffusers import DDPMScheduler
from packaging import version
from model import get_resnet, replace_bn_with_gn, vision_encoder, ConditionalUnet1D
from data import normalize_data, MazeImageDataset

virtual_display = Display(visible=0, size=(1024, 768))
virtual_display.start()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset_path = 'maze.zarr' 
pred_horizon = 16
obs_horizon = 16
action_horizon = 1
action_dim = 15  

dataset = MazeImageDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)

stats = dataset.stats
print("Dataset initialized.")

vision_feature_dim = 512
image_feature_dim_reduced = 42  
global_cond_dim = image_feature_dim_reduced  
diffusion_step_embed_dim = 256

print(f"Expected global_cond_dim: {global_cond_dim}")

vision_encoder = get_resnet('resnet18', weights=None)  
vision_encoder = replace_bn_with_gn(vision_encoder)
vision_encoder = vision_encoder.to(device)
print("Vision encoder initialized and moved to device.")

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=global_cond_dim,
    diffusion_step_embed_dim=diffusion_step_embed_dim
).to(device)
print("Noise prediction network initialized and moved to device.")

cond_encoder = nn.Linear(vision_feature_dim, image_feature_dim_reduced).to(device)
print("Condition encoder initialized and moved to device.")

nets = {
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net,
    'cond_encoder': cond_encoder
}
ema_nets = nets

for net_name, net in ema_nets.items():
    net.eval()
    print(f"Model {net_name} set to evaluation mode.")

env = gym.make('procgen:procgen-maze-v0', start_level=0, num_levels=1)
print("Environment initialized.")

gym_version = gym.__version__
print(f"Gym version: {gym_version}")

if version.parse(gym_version) >= version.parse("0.26.0"):
    reset_kwargs = {}
    step_kwargs = {}
    reset_return_info = True
    step_return_info = True
else:
    reset_kwargs = {}
    step_kwargs = {}
    reset_return_info = False
    step_return_info = False

if reset_return_info:
    obs, info = env.reset(**reset_kwargs)
    print("Environment reset with info.")
else:
    obs = env.reset(**reset_kwargs)
    print("Environment reset without info.")

print(f"Initial observation type: {type(obs)}")
if isinstance(obs, dict):
    print(f"Initial observation keys: {obs.keys()}")
print(f"Initial observation shape: {obs.shape if isinstance(obs, np.ndarray) else 'N/A'}")
print(f"Initial observation dtype: {obs.dtype if isinstance(obs, np.ndarray) else 'N/A'}")

done = False
step_idx = 0
max_steps = 100
rewards = []
imgs = []
obs_deque = collections.deque(maxlen=obs_horizon)

for _ in range(obs_horizon):
    obs_deque.append(obs)

def visualize_images_as_animation(imgs):
    valid_imgs = []
    for idx, img in enumerate(imgs):
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] == 3:
                if img.dtype not in [np.uint8, np.float32, np.float64]:
                    try:
                        img = img.astype(np.float32)
                        print(f"Converted image {idx} to float32.")
                    except Exception as e:
                        print(f"Failed to convert image {idx}: {e}")
                        continue
                valid_imgs.append(img)
            else:
                print(f"Image {idx} has unexpected shape: {img.shape}")
        else:
            print(f"Image {idx} is not a NumPy array: {type(img)}")

    if not valid_imgs:
        raise ValueError("No valid images to display.")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Maze Animation")
    ax.axis('off')
    img_plot = ax.imshow(valid_imgs[0])

    def update(frame):
        img_plot.set_data(frame)
        return [img_plot]

    ani = animation.FuncAnimation(fig, update, frames=valid_imgs, blit=True, interval=100)
    return HTML(ani.to_html5_video())

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,  
    beta_start=0.0001,         
    beta_end=0.02,             
    beta_schedule="linear"     
)
print("Noise scheduler initialized.")

with torch.no_grad():  
    with tqdm(total=max_steps, desc="Eval Maze") as pbar:
        while not done and step_idx < max_steps:
            images = []
            for obs_item in obs_deque:
                if isinstance(obs_item, dict):
                    image = obs_item.get('image') 
                    if image is None:
                        print("Warning: 'image' key not found in observation.")
                        continue
                else:
                    image = obs_item 

                if not isinstance(image, np.ndarray):
                    try:
                        image = np.array(image)
                        print("Converted obs_item to NumPy array.")
                    except Exception as e:
                        print(f"Failed to convert obs_item to NumPy array: {e}")
                        continue

                if image.ndim == 3 and image.shape[-1] == 3:
                    images.append(image)
                else:
                    print(f"Unexpected image shape: {image.shape}")
                    continue

            if not images:
                print("No valid images found in obs_deque.")
                break

            try:
                images = np.stack(images)  # Shape: (obs_horizon, H, W, C)
                images = images.transpose(0, 3, 1, 2)  # Convert to (obs_horizon, C, H, W)
                images = normalize_data(images, stats=stats['image'])  # Normalize to [-1, 1]
                images = torch.from_numpy(images).unsqueeze(0).to(device, dtype=torch.float32)  # (1, obs_horizon, C, H, W)
                images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])  # (obs_horizon, C, H, W)
            except Exception as e:
                print(f"Failed to process images: {e}")
                break

            try:
                image_features = ema_nets['vision_encoder'](images)  # (obs_horizon, 512)
                image_features = image_features.view(1, obs_horizon, -1).mean(dim=1)  # (1, 512)
            except Exception as e:
                print(f"Failed to extract image features: {e}")
                break

            try:
                obs_cond = ema_nets['cond_encoder'](image_features)  # (1, 42)
                assert obs_cond.shape[-1] == global_cond_dim, f"Expected global_cond_dim={global_cond_dim}, got {obs_cond.shape[-1]}"
            except Exception as e:
                print(f"Failed in condition encoding: {e}")
                break

            noisy_action = torch.randn((1, pred_horizon, action_dim), device=device)

            try:
                noise = torch.randn_like(noisy_action)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
                noisy_actions = noise_scheduler.add_noise(noisy_action, noise, timesteps)  # [1,16,15]
            except Exception as e:
                print(f"Failed to add noise: {e}")
                break

            try:
                naction = noisy_actions
                for k in noise_scheduler.timesteps:
                    noise_pred = ema_nets['noise_pred_net'](naction, k, global_cond=obs_cond)
                    naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
            except Exception as e:
                print(f"Failed during denoising: {e}")
                break

            try:
                action_pred = naction.argmax(dim=-1)  # (1, 16)
                action_pred = action_pred.squeeze(0).cpu().numpy()  # (16,)
            except Exception as e:
                print(f"Failed to convert to one-hot actions: {e}")
                break

            for i in range(pred_horizon):
                try:
                    action = int(action_pred[i]) 
                    step_result = env.step(action)

                    if version.parse(gym_version) >= version.parse("0.26.0"):
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_result

                    print(f"Step {step_idx}: Action={action}, Reward={reward}, Done={done}")

                    obs_deque.append(obs)
                    rewards.append(reward)

                    if isinstance(obs, dict):
                        img = obs.get('image')  
                        if img is None:
                            print("Warning: 'image' key not found in observation.")
                            continue
                    else:
                        img = obs 

                    if isinstance(img, np.ndarray):
                        if img.ndim == 3 and img.shape[-1] == 3:
                            imgs.append(img)
                        else:
                            print(f"Unexpected image shape when appending: {img.shape}")
                    else:
                        print(f"Observation image is not a NumPy array: {type(img)}")

                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)

                    if step_idx >= max_steps or done:
                        done = True
                        break
                except Exception as e:
                    print(f"Failed during action execution: {e}")
                    done = True
                    break

def visualize_images_as_animation(imgs):
    valid_imgs = []
    for idx, img in enumerate(imgs):
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] == 3:
                if img.dtype not in [np.uint8, np.float32, np.float64]:
                    try:
                        img = img.astype(np.float32)
                        print(f"Converted image {idx} to float32.")
                    except Exception as e:
                        print(f"Failed to convert image {idx}: {e}")
                        continue
                valid_imgs.append(img)
            else:
                print(f"Image {idx} has unexpected shape: {img.shape}")
        else:
            print(f"Image {idx} is not a NumPy array: {type(img)}")

    if not valid_imgs:
        raise ValueError("No valid images to display.")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Maze Animation")
    ax.axis('off')
    img_plot = ax.imshow(valid_imgs[0])

    def update(frame):
        img_plot.set_data(frame)
        return [img_plot]

    ani = animation.FuncAnimation(fig, update, frames=valid_imgs, blit=True, interval=100)
    return HTML(ani.to_html5_video())

try:
    if imgs:
        print("Starting visualization.")
    else:
        print("No images collected to visualize.")
except Exception as e:
    print(f"Visualization failed: {e}")

virtual_display.stop()
print("Virtual display stopped.")
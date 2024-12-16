import argparse
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
from model import get_resnet, replace_bn_with_gn, ConditionalUnet1D
from data import normalize_data, MazeImageDataset

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy in Gym Environment")
    parser.add_argument('--env', type=str, required=True, help='Environment name (e.g., procgen:procgen-maze-v0)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--eval', action='store_true', help='Flag to indicate evaluation mode')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum number of steps for evaluation')
    parser.add_argument('--dataset_path', type=str, default='maze.zarr', help='Path to the dataset file')
    parser.add_argument('--pred_horizon', type=int, default=16, help='Prediction horizon')
    parser.add_argument('--obs_horizon', type=int, default=16, help='Observation horizon')
    parser.add_argument('--action_horizon', type=int, default=1, help='Action horizon')
    parser.add_argument('--output_video', type=str, default=None, help='Path to save the evaluation video (optional)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for evaluation')
    args = parser.parse_args()
    return args

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

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    vision_encoder = get_resnet('resnet18', weights=None)
    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_encoder = vision_encoder.to(device)

    noise_pred_net = ConditionalUnet1D(
        input_dim=15,  # action_dim
        global_cond_dim=42,  # image_feature_dim_reduced
        diffusion_step_embed_dim=256
    ).to(device)

    cond_encoder = nn.Linear(512, 42).to(device)

    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net,
        'cond_encoder': cond_encoder
    })

    nets.load_state_dict(checkpoint, strict=False)

    for net in nets.values():
        net.eval()

    return nets

def main():
    args = get_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize virtual display if running in headless environment
    virtual_display = Display(visible=0, size=(1024, 768))
    virtual_display.start()

    # Load dataset
    dataset = MazeImageDataset(
        dataset_path=args.dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon
    )
    stats = dataset.stats
    print("Dataset initialized.")

    # Load model
    nets = load_model(args.model_path, device)
    print("Model loaded and set to evaluation mode.")

    # Initialize environment
    env = gym.make(args.env, start_level=0, num_levels=1)
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
    rewards = []
    imgs = []
    obs_deque = collections.deque(maxlen=args.obs_horizon)

    for _ in range(args.obs_horizon):
        obs_deque.append(obs)

    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,  
        beta_start=0.0001,         
        beta_end=0.02,             
        beta_schedule="linear"     
    )
    print("Noise scheduler initialized.")

    with torch.no_grad():  
        with tqdm(total=args.max_steps, desc="Eval Maze") as pbar:
            while not done and step_idx < args.max_steps:
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
                    image_features = nets['vision_encoder'](images)  # (obs_horizon, 512)
                    image_features = image_features.view(1, args.obs_horizon, -1).mean(dim=1)  # (1, 512)
                except Exception as e:
                    print(f"Failed to extract image features: {e}")
                    break

                try:
                    obs_cond = nets['cond_encoder'](image_features)  # (1, 42)
                    assert obs_cond.shape[-1] == 42, f"Expected global_cond_dim=42, got {obs_cond.shape[-1]}"
                except Exception as e:
                    print(f"Failed in condition encoding: {e}")
                    break

                noisy_action = torch.randn((1, args.pred_horizon, 15), device=device)

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
                        noise_pred = nets['noise_pred_net'](naction, k, global_cond=obs_cond)
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

                for i in range(args.pred_horizon):
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

                        if step_idx >= args.max_steps or done:
                            done = True
                            break
                    except Exception as e:
                        print(f"Failed during action execution: {e}")
                        done = True
                        break

    if args.eval:
        try:
            if imgs:
                print("Starting visualization.")
                animation_html = visualize_images_as_animation(imgs)
                if args.output_video:
                    with open(args.output_video, "wb") as f:
                        f.write(animation_html.data.encode())
                    print(f"Saved animation video to {args.output_video}")
                else:
                    display(animation_html)
            else:
                print("No images collected to visualize.")
        except Exception as e:
            print(f"Visualization failed: {e}")

    # Stop the virtual display
    virtual_display.stop()
    print("Virtual display stopped.")

if __name__ == "__main__":
    main()

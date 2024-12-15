import torch
import torch.utils.data
import numpy as np
import zarr
from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from torch.optim import AdamW
import torchvision
import torch.nn as nn
from data import dataset
from model import ConditionalUnet1D

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,       
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear"
)

num_epochs = 10


vision_feature_dim = 512
image_feature_dim_reduced = 42
action_dim = 15  
lowdim_obs_dim = action_dim  
obs_dim = vision_feature_dim + lowdim_obs_dim 
global_cond_dim = image_feature_dim_reduced  
diffusion_step_embed_dim = 256

print(f"Expected global_cond_dim: {global_cond_dim}")

def get_resnet(name: str, weights: Union[str, None]=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    func = getattr(torchvision.models, name)
    if weights is not None:
        resnet = func(weights=weights, **kwargs)
    else:
        resnet = func(pretrained=False, **kwargs)
    resnet.fc = nn.Identity() 
    return resnet

def replace_bn_with_gn(root_module: nn.Module, features_per_group: int = 16) -> nn.Module:
    """
    Replace all BatchNorm2d layers with GroupNorm.
    """
    bn_module_names = [name for name, module in root_module.named_modules() if isinstance(module, nn.BatchNorm2d)]

    for name in bn_module_names:
        parent_name, _, module_name = name.rpartition('.')
        parent_module = root_module if parent_name == '' else root_module.get_submodule(parent_name)

        bn_module = getattr(parent_module, module_name)
        num_channels = bn_module.num_features
        num_groups = max(1, num_channels // features_per_group)

        gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

        setattr(parent_module, module_name, gn)

    return root_module

vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)
vision_encoder = vision_encoder.to(device)

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=global_cond_dim,
    diffusion_step_embed_dim=diffusion_step_embed_dim 
).to(device)

cond_encoder = nn.Linear(vision_feature_dim, image_feature_dim_reduced).to(device)

nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net,
    'cond_encoder': cond_encoder
})

ema = EMAModel(
    parameters=nets.parameters(),
    model=nets,
    power=0.75
)

optimizer = AdamW(
    params=nets.parameters(),
    lr=1e-4,
    weight_decay=1e-6
)

lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

# Training Loop
with tqdm(range(num_epochs), desc='Epoch') as epoch_bar:
    for epoch_idx in epoch_bar:
        epoch_loss = []
        for nbatch in dataloader:
            images = nbatch['image'].to(device).float()  # Shape: (B, obs_horizon, C, H, W)
            actions = nbatch['action'].to(device).float()  # Shape: (B, pred_horizon, 15)

            batch_size, obs_horizon, C, H, W = images.shape
            images = images.view(batch_size * obs_horizon, C, H, W)  # (B*obs_horizon, C, H, W)

            with torch.no_grad(): 
                image_features = nets['vision_encoder'](images)  # (B*obs_horizon, 512)

            image_features = image_features.view(batch_size, obs_horizon, -1).mean(dim=1)  # (B, 512)

            image_features_reduced = nets['cond_encoder'](image_features)  # (B, 42)

            obs_cond = image_features_reduced  # (B, 42)

            assert obs_cond.shape[-1] == global_cond_dim, f"Expected global_cond_dim={global_cond_dim}, got {obs_cond.shape[-1]}"

            naction = actions

            noise = torch.randn_like(naction).to(device) 
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()  # (B,)

            noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps) 

            noise_pred = nets['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond) 

            noise_pred = torch.clamp(noise_pred, -1e3, 1e3)
            noise = torch.clamp(noise, -1e3, 1e3)

            loss = nn.functional.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            ema.step(nets)

            epoch_loss.append(loss.item())

        avg_loss = np.mean(epoch_loss)
        epoch_bar.set_postfix(loss=avg_loss)
        epoch_bar.update(1)
        print(f"Epoch {epoch_idx + 1}/{num_epochs} - Avg Loss: {avg_loss:.6f}")
import os
import sys
import yaml
import time
import torch
import torch.nn as nn
import random
import argparse

import numpy as np
import torchio as tio
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch import autocast
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from torchmetrics.image import StructuralSimilarityIndexMeasure

from models.pytorch_vqgan import VQModel
from cellpainting_loader_2ch import get_dataloader

from vqloss.vq_loss import VQLPIPSWithDiscriminator
from vqloss.my_lpips import LPIPS

from utils import denormalize_image, create_output_dir, save_config_file_copy_at_output



def load_model(checkpoint_path):
    """
    Loads a pretrained VAE and replaces the first and last conv layers for 2-channel input/output.
    """

    # Default model configuration from taming-transformers repo
    ddconfig = {"double_z": False,
                "z_channels": 256,
                "resolution": 256,
                "in_channels": 2,
                "out_ch": 2,
                "ch": 128,
                "ch_mult": (1, 1, 2, 2, 4),
                "num_res_blocks": 2,
                "attn_resolutions": [16],
                "dropout": 0.0}
    embed_dim = 256
    n_embed = 1024

    vqgan = VQModel(ddconfig = ddconfig,
                    embed_dim = embed_dim,
                    n_embed = n_embed)
    
    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vqgan.load_state_dict(checkpoint)

    """
    # workaround to load model trained using pytorch lightning
    state_dict = checkpoint["state_dict"]

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if "loss" in k:
            continue  # Skip parameters of the loss function
        cleaned_state_dict[k] = v
    
    vqgan.load_state_dict(cleaned_state_dict)

    # replacing conv_in and conv_out layers to support 2-channel input/output
    # # conv_in
    old_conv_in = vqgan.encoder.conv_in
    
    vqgan.encoder.conv_in = nn.Conv2d(
        in_channels=2,  # 2-channel input
        out_channels=old_conv_in.out_channels,
        kernel_size=old_conv_in.kernel_size,
        stride=old_conv_in.stride,
        padding=old_conv_in.padding,
        dilation=old_conv_in.dilation,
        groups=old_conv_in.groups,
        bias=(old_conv_in.bias is not None),
    )

    # # copy the original weight into the new conv_in layer
    with torch.no_grad():
        vqgan.encoder.conv_in.weight.copy_(old_conv_in.weight[:, :2, :, :])
        if old_conv_in.bias is not None:
            vqgan.encoder.conv_in.bias.copy_(old_conv_in.bias)

    # # conv_out
    old_conv_out = vqgan.decoder.conv_out
    vqgan.decoder.conv_out = nn.Conv2d(
        in_channels=old_conv_out.in_channels,
        out_channels=2,  # 2-channel output
        kernel_size=old_conv_out.kernel_size,
        stride=old_conv_out.stride,
        padding=old_conv_out.padding,
        dilation=old_conv_out.dilation,
        groups=old_conv_out.groups,
        bias=(old_conv_out.bias is not None),
    )

   # # copy the original weight into the new conv_out layer
    with torch.no_grad():
        # Copy only the first 2 output channels, all input channels
        vqgan.decoder.conv_out.weight.copy_(old_conv_out.weight[:2, :, :, :])
        if old_conv_out.bias is not None:
            vqgan.decoder.conv_out.bias.copy_(old_conv_out.bias[:2])

    """
    
    # set all model parameters to require gradients
    for param in vqgan.parameters():
        param.requires_grad = False

    print("✅ custom vqgan prepared")
    
    # Print trainable parameters for verification
    trainable_params = sum(p.numel() for p in vqgan.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in vqgan.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return vqgan


def add_zero_channel(img):
    """Convert 2-channel to RGB by using channels as R,G and setting B=0"""
    # img shape: (B, 2, H, W) -> (B, 3, H, W)
    zeros = torch.zeros_like(img[:, 0:1, :, :])
    return torch.cat([img[:, 0:1, :, :], img[:, 1:2, :, :], zeros], dim=1)


def load_dataset(batch_size, num_workers):
    
    val_dataloader = get_dataloader(names_file = "/home/tiagofroes/workplace/hf_vae/data_files/reconstruction_full_val.txt",
                                    images_root = "/home/tiagofroes/workplace/DATA/reconstruction",
                                    split = "val",
                                    to_normal = True,
                                    batch_size = batch_size,
                                    num_workers = num_workers)

    return val_dataloader


def test(test_loader, model, device, args):

    # Get training parameters
    output_dir = args.output_dir

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Base output path
    output_path = f'./outputs/{output_dir}'

    # Model to gpu
    model = model.to(device)
    model.eval()

    # Validation metrics
    l1_loss = torch.nn.L1Loss().to(device)
    l2_loss = torch.nn.MSELoss().to(device)
    ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    lpips = LPIPS().to(device)
    lpips.eval()

    # init metrics
    val_metrics_dict = {"l1_score" : 0.0,
                        "l2_score" : 0.0,
                        "2ch_p_score" : 0.0,
                        "3ch_p_score" : 0.0,
                        "ssim_score" : 0.0,
                        "codebook_loss" : 0.0}
        
    progress_bar = tqdm(enumerate(test_loader), total = len(test_loader), ncols = 50)
    progress_bar.set_description(f"Test: ")

    # test loop
    for step, batch in progress_bar:
        images = batch["images"].to(device)

        with torch.no_grad(), autocast("cuda", enabled = True):

            # Forward pass
            reconstructed_image, codebook_loss = model(images)
            val_metrics_dict["codebook_loss"] += codebook_loss.item()

            # Calculate metrics
            val_metrics_dict["l1_score"] += l1_loss(reconstructed_image, images).item()
            val_metrics_dict["l2_score"] += l2_loss(reconstructed_image, images).item()


            for channel_index in range(reconstructed_image.shape[1]):
                    val_metrics_dict["2ch_p_score"] += lpips(reconstructed_image[:,channel_index,:,:].unsqueeze(dim = 1),
                                                             images[:,channel_index,:,:].unsqueeze(dim = 1)).item()/reconstructed_image.shape[1]


            lpips_reconstructed_image = add_zero_channel(reconstructed_image)
            lpips_images = add_zero_channel(images)
            val_metrics_dict["3ch_p_score"] += lpips(lpips_reconstructed_image.contiguous(), lpips_images.contiguous()).item()

                    
            ssim_loss.update(reconstructed_image.contiguous(), images.contiguous())


        n_channels = images.shape[1]

        reconstructed_image = reconstructed_image.squeeze().cpu()
        images = images.squeeze().cpu()

        # putting back to [0, 1] range
        reconstructed_image = denormalize_image(reconstructed_image)
        images = denormalize_image(images)

        for i in range(n_channels):

            ch = reconstructed_image[i]
            save_image(ch.float(), f'{output_path}/generated/{step}_ch{i}_reconstruction.png')
            ch = images[i]
            save_image(ch.float(), f'{output_path}/generated/{step}_ch{i}_original.png')


    val_metrics_dict["ssim_score"] = ssim_loss.compute().item()

    # get mean, log and store validation metrics
    for metric in val_metrics_dict.keys():
        if metric != "ssim_score":
            val_metrics_dict[metric] /= len(test_loader)


    # print and save results
    results = (f"Test Results - L1: {val_metrics_dict.get('l1_score', 0):.4f}, "
           f"L2: {val_metrics_dict.get('l2_score', 0):.4f}, "
           f"3ch_Perceptual: {val_metrics_dict.get('3ch_p_score', 0):.4f}, "
           f"2ch_Perceptual: {val_metrics_dict.get('2ch_p_score', 0):.4f}, "
           f"SSIM: {val_metrics_dict.get('ssim_score', 0):.4f}, ")
    print(results)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    save_path = f"./outputs/{args.output_dir}/test_results.txt"

    # Save to file
    with open(save_path, "w") as file:
        file.write(results)

    print(f"Saved results to {save_path}")


def main(args):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output folder function
    create_output_dir(args.output_dir)    

    test_loader = load_dataset(batch_size = 1,
                               num_workers = 8)

    # Load VAE
    # stabilityai/sd-vae-ft-mse
    # madebyollin/sdxl-vae-fp16-fix
    vqgan = load_model(checkpoint_path= args.checkpoint_path)
    
    total_start = time.time()
    
    test(test_loader = test_loader,
         model = vqgan,
         device = device,
         args = args)
    
    total_time = time.time() - total_start
    print(f"Train completed, total time: {total_time:.1f}s")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training script for deep learning model")
    
    parser.add_argument('--output_dir', type=str, required=True, help='Output path for checkpoints and generated data')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')

    args = parser.parse_args()

    # save training parameters using mlflow
    args_dict = dict(vars(args))
    
    print("\n\n\n\ntest parameters:")
    for var in args_dict.keys():
        print(f"\t--{var}: {args_dict[var]}")
    print("\n\n\n\n")
    
    main(args)
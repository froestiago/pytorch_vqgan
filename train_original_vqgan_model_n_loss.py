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

from torch.optim.lr_scheduler import CosineAnnealingLR

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

    
    # set all model parameters to require gradients
    for param in vqgan.parameters():
        param.requires_grad = True

    """

    print("✅ custom vqgan prepared")
    
    # Print trainable parameters for verification
    trainable_params = sum(p.numel() for p in vqgan.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in vqgan.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return vqgan


def load_dataset(batch_size, num_workers):

    train_dataloader = get_dataloader(names_file = "/home/tiagofroes/workplace/hf_vae/data_files/reconstruction_full_train.txt",
                                      images_root = "/home/tiagofroes/workplace/DATA/reconstruction",
                                      split = "train",
                                      to_normal = True,
                                      batch_size = batch_size,
                                      num_workers = num_workers)
    
    val_dataloader = get_dataloader(names_file = "/home/tiagofroes/workplace/hf_vae/data_files/reconstruction_full_val.txt",
                                    images_root = "/home/tiagofroes/workplace/DATA/reconstruction",
                                    split = "val",
                                    to_normal = True,
                                    batch_size = batch_size,
                                    num_workers = num_workers)

    return train_dataloader, val_dataloader


def train(train_loader, val_loader, model, args):

    # Get training parameters
    n_epochs = args.n_epochs
    n_steps_warm_up = args.n_steps_warm_up
    val_interval = args.val_interval
    save_interval = args.save_interval
    learning_rate = args.learning_rate
    output_dir = args.output_dir

    # Set device
    device = torch.device("cuda")

    # Base output path
    output_path = f'./outputs/{output_dir}'

    # Model to gpu
    model = model.to(device)

    # init loss (as in taming-transformers repo)
    vq_loss = VQLPIPSWithDiscriminator(
        disc_start=0,  # Start discriminator after warm-up epochs
        codebook_weight=args.quant_loss_weight,
        pixelloss_weight=1.0,
        disc_num_layers=2,
        disc_in_channels=2,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge"
    ).to("cuda")

    global_step = 0

    # Validation metrics - data_range should be 2.0 for [-1, 1] normalized images
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    val_l1_loss = torch.nn.L1Loss().to(device)
    val_l2_loss = torch.nn.MSELoss().to(device)

    # Optimizers
    generator_optimizer = torch.optim.AdamW(params=model.parameters(), lr = learning_rate, weight_decay = 0.01)
    discriminator_optimizer = torch.optim.AdamW(params=vq_loss.discriminator.parameters(), lr = args.learning_rate)
    
    # scalers
    scaler_generator = GradScaler()
    scaler_discriminator = GradScaler()

    lr_scheduler = CosineAnnealingLR(generator_optimizer, T_max=n_epochs)

    # Main loop
    for epoch in range(4, n_epochs+1):

        model.train()

        # printable metrics
        epoch_total_loss = 0.0
        epoch_img_rec_loss = 0.0
        epoch_p_loss = 0.0
        epoch_quant_loss = 0.0
        epoch_discriminator_loss = 0.0
        epoch_generator_loss = 0.0

        progress_bar = tqdm(total=len(train_loader), ncols = 200, disable = False)
        progress_bar.set_description(f"Epoch {epoch}")
        print_every_train = len(train_loader) // 100

        # Train loop
        for step, batch in enumerate(train_loader):
            global_step += 1

            original_image = batch["images"].to(device)
            
            with autocast("cuda", enabled = True):

                reconstructed_image, codebook_loss = model(original_image)

                # Get the last layer of the model
                last_layer = model.decoder.conv_out.weight

                loss, log = vq_loss(codebook_loss = codebook_loss,
                                    inputs = original_image,
                                    reconstructions = reconstructed_image,
                                    optimizer_idx = 0,
                                    global_step = global_step,
                                    split = "train",
                                    last_layer = last_layer)
                
            epoch_total_loss += log["train/total_loss"].item()
            epoch_img_rec_loss += log["train/rec_loss"].item()
            epoch_p_loss += log["train/p_loss"].item()
            epoch_quant_loss += log["train/quant_loss"].item()    
            epoch_generator_loss += log["train/g_loss"].item()                

            # Backpropagation for generator
            generator_optimizer.zero_grad(set_to_none=True)
            scaler_generator.scale(loss).backward()
            scaler_generator.step(generator_optimizer)
            scaler_generator.update()

            # Forward pass for discriminator update (optimizer_idx=1)
            if global_step >= n_steps_warm_up:
                with autocast("cuda", enabled=True):
                    d_loss, d_log = vq_loss(
                        codebook_loss=codebook_loss,
                        inputs=original_image,
                        reconstructions=reconstructed_image,
                        optimizer_idx=1,
                        global_step=global_step
                    )

                epoch_discriminator_loss += d_log["train/disc_loss"].item()

                # Backpropagation for discriminator
                discriminator_optimizer.zero_grad(set_to_none=True)
                scaler_discriminator.scale(d_loss).backward()
                scaler_discriminator.step(discriminator_optimizer)
                scaler_discriminator.update()

            if step % print_every_train == 0 or step == len(train_loader) - 1:
                progress_bar.set_postfix({"total_loss": epoch_total_loss / (step + 1),
                                          "img_rec_loss": epoch_img_rec_loss / (step + 1),
                                          "codebook_loss": epoch_quant_loss / (step + 1),
                                          "p_loss": epoch_p_loss / (step + 1),
                                          "generator_loss": epoch_generator_loss / (step + 1),
                                          "disc_loss": epoch_discriminator_loss / ((step + 1) if epoch >= n_steps_warm_up else 1)})
                progress_bar.update(print_every_train)

        # Close progress bar for this epoch
        progress_bar.close()

        print(f"Global step: {global_step}")

        # Validation condition
        if (epoch + 1) % val_interval ==0:
            torch.cuda.empty_cache()
            
            model.eval()
            vq_loss.eval()

            val_l1 = 0.0
            val_l2 = 0.0
            val_img_rec_loss = 0.0
            val_p_loss = 0.0    
            val_codebook_loss = 0.0

            # get rand index to save img
            random_index = random.randint(0, len(val_loader)-1)

            # Validation loop
            print(f"\n\nValidating epoch {epoch}...")
            with torch.no_grad():
                for step, batch in enumerate(val_loader):

                    original_image = batch["images"].to(device)
                    
                    with autocast("cuda", enabled=True):

                        reconstructed_image, codebook_loss = model(original_image)
                        val_codebook_loss += codebook_loss

                        val_l1 += val_l1_loss(reconstructed_image, original_image).item()
                        val_l2 += val_l2_loss(reconstructed_image, original_image).item()

                        loss, log = vq_loss(codebook_loss=codebook_loss,
                                            inputs=original_image,
                                            reconstructions=reconstructed_image,
                                            optimizer_idx=0,
                                            global_step=global_step,
                                            last_layer=last_layer,  # Pass the last layer here
                                            split="val")
                        
                    val_img_rec_loss += log["val/rec_loss"].item()
                    val_p_loss += log["val/p_loss"].item()
                    val_codebook_loss += log["val/quant_loss"].item()

                    ssim_fn.update(reconstructed_image.contiguous(), original_image.contiguous())

                    # Save reconstructed image
                    if step == random_index:

                        # random batch index
                        batch_index = random.randint(0, original_image.shape[0] - 1)

                        # Denormalize images from [-1, 1] to [0, 1] for proper saving
                        output_to_save = denormalize_image(reconstructed_image[batch_index].cpu().squeeze())
                        original_to_save = denormalize_image(original_image[batch_index].cpu().squeeze())

                        for i in range(original_image.shape[1]):

                            ch = output_to_save[i]
                            save_image(ch.float(), f'{output_path}/validation/epoch_{epoch}_ch{i}_reconstruction.png')

                            ch = original_to_save[i]
                            save_image(ch.float(), f'{output_path}/validation/epoch_{epoch}_ch{i}_original.png')

            # Compute final metrics
            avg_img_rec = val_img_rec_loss / len(val_loader)
            avg_p_loss = val_p_loss / len(val_loader)
            avg_codebook_loss = val_codebook_loss / len(val_loader)
            avg_ssim = ssim_fn.compute()
            avg_l1 = val_l1 / len(val_loader)
            avg_l2 = val_l2 / len(val_loader)

            print(f"\n\nValidation Epoch {epoch}: img_rec: {avg_img_rec:.4f}, L1: {avg_l1:.4f}, L2: {avg_l2:.4f}, ssim: {avg_ssim:.4f}, p_loss: {avg_p_loss:.4f}, codebook_loss: {avg_codebook_loss:.4f}\n\n")

            torch.cuda.empty_cache()

        lr_scheduler.step()
        current_lr = generator_optimizer.param_groups[0]["lr"]
        print(f"Updated  learning rate = {current_lr}")

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f'{output_path}/checkpoints/epoch_{epoch}_weights.pt')


def main(args):
    
    # Create output dir with subdirs
    create_output_dir(args.output_dir)

    # Save a copy of the config file in the output dir
    save_config_file_copy_at_output(args)

    # Load Datasets
    train_loader, val_loader = load_dataset(batch_size = args.batch_size,
                                            num_workers= args.num_workers)

    # Load model    
    vqgan = load_model(checkpoint_path= args.checkpoint_path)

    total_start = time.time()
    
    train(train_loader = train_loader,
          val_loader = val_loader,
          model = vqgan,
          args = args)
    
    total_time = time.time() - total_start
    print(f"Train completed, total time: {total_time:.1f}s")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training script for deep learning model")

    # Dataset parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers to load data")

    # Training parameters
    parser.add_argument('--checkpoint_path', type=str, default = "./", help='Path to the model checkpoint (downloaded from taming transformers repo)')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of total training epochs')
    parser.add_argument('--n_steps_warm_up', type=int, default=10, help='Number of epochs before the discriminator starts to work')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval')
    parser.add_argument('--save_interval', type=int, default=10, help='Checkpoint save interval')
    parser.add_argument('--output_dir', type=str, required=True, help='Output path for checkpoints and generated data')
    parser.add_argument('--learning_rate', type=float, default=1.e-5, help='Learning rate for the optimizer')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients before optimizer step')
    parser.add_argument('--quant_loss_weight', type=float, default=0.25, help='Weight for the quantization commitment loss')


    args = parser.parse_args()

    # Call the main function
    main(args)
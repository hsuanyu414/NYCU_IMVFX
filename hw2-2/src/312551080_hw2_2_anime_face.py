#!/usr/bin/env python
# coding: utf-8

# # Set up the environment
# 
# 

# # Random seed

# Set the random seed to a certain value for reproducibility.

# In[ ]:


import random

import torch
import numpy as np

def same_seeds(seed):
  # Python built-in random module
  random.seed(seed)
  # Numpy
  np.random.seed(seed)
  # Torch
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

# Set random seed for reproducibility
same_seeds(999)


# # Import packages

# First, we need to import packages that will be used later.

# In[ ]:


# Import packages

import numpy as np
import cv2
import einops
import imageio
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, Grayscale, Resize, Normalize
from torchvision.datasets import ImageFolder
import os


# # Hyperparameters and Initialization

# Define some inputs for the run later.

# In[ ]:


# You may replace the workspace directory if you want
workspace_dir = '.'

# Create the anime_face_log folder if not exist
log_dir = f"{workspace_dir}/anime_face_log"
os.makedirs(log_dir, exist_ok=True)

# Root directory for the anime_face dataset
dataset_path = f"{workspace_dir}/anime_face_dataset"

# The path to save the model
model_store_path = f"{workspace_dir}/anime_face.pt"

# Batch size during training
batch_size = 128

# Number of training epochs
n_epochs = 100

# Learning rate for optimizers
lr = 0.0005

# Number of the forward steps
n_steps = 1000

# Initial beta
start_beta = 1e-4

# End beta
end_beta = 0.02

# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# List to keep track of loss
loss_list = []


# # Dataset

# ### Create the dataset and data loader.

# In[ ]:


dataset = ImageFolder(root=dataset_path, transform=Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)]
))
dataloader = DataLoader(dataset, batch_size, shuffle=True)


# ### Show some images

# In[ ]:


# Show images
def show_images(images, title="", save=False, filename="images.png"):
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()
    print("The shape of images: ", images.shape)
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, fontsize=24)
    rows = int(len(images) ** 0.5)
    cols = round(len(images) / rows)
    index = 0
    for row in range(rows):
        for col in range(cols):
            fig.add_subplot(rows, cols, index + 1)
            if index < len(images):
                frame = plt.gca()
                frame.axes.get_yaxis().set_visible(False)
                frame.axes.get_xaxis().set_visible(False)
                temp = np.transpose(images[index], (1, 2, 0))
                temp = ((temp - temp.min()) / (temp.max() - temp.min())-0.5)*2
                plt.imshow((temp+1)/2)
                index += 1
    if save:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

# Show images of next batch
def show_images_of_next_batch(loader):
    dataiter = iter(dataloader)
    data = next(dataiter)
    features, labels = data
    show_images(features, "Images in a batch")

# show_images_of_next_batch(dataloader)


# # Model

# Here, we constructed a noise predictor using a simple U-Net architecture to generate handwritten numbers.
# 
# If you want to achieve better results, you need to design more complex models yourself.
# 
# Feel free to modify your own model structure!

# ### DDPM

# In[ ]:


# Define the class of DDPM
class DDPM(nn.Module):
    def __init__(self, image_shape=(3, 64, 64), n_steps=200, start_beta=1e-4, end_beta=0.02, device=None):
        super(DDPM, self).__init__()
        self.device = device
        self.image_shape = image_shape
        self.n_steps = n_steps
        self.noise_predictor = UNet(n_steps).to(device)
        self.betas = torch.linspace(start_beta, end_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    # Forward process
    # Add the noise to the images
    def forward(self, x0, t, eta=None):
        n, channel, height, width = x0.shape
        alpha_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, channel, height, width).to(self.device)

        noise = alpha_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - alpha_bar).sqrt().reshape(n, 1, 1, 1) * eta
        # noise = noise.clamp(-1, 1) # MODIFIED
        # noise = ((noise-noise.min())/(noise.max()-noise.min()))*2-1
        return noise

    # Backward process
    # Predict the noise that was added to the images during the forward process
    def backward(self, x, t):
        return self.noise_predictor(x, t)


# ### Time embedding

# In[ ]:


# Create the time embedding
def time_embedding(n, d):
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])
    return embedding


# ### Noise predictor

# In[ ]:


# Define the class of U-Net
class UNet(nn.Module):
    def __init__(self, n_steps=1000, time_embedding_dim=256):
        factor = 1
        super(UNet, self).__init__()

        # Time embedding
        self.time_step_embedding = nn.Embedding(n_steps, time_embedding_dim)
        self.time_step_embedding.weight.data = time_embedding(n_steps, time_embedding_dim)
        self.time_step_embedding.requires_grad_(False)

        # The first half
        self.time_step_encoder1 = nn.Sequential(
            nn.Linear(time_embedding_dim, 1),
            nn.SiLU(),
            nn.Linear(1, 1)
        )

        self.block1 = nn.Sequential(
            nn.LayerNorm((3, 64, 64)),
            nn.Conv2d(3, 32*factor, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32*factor, 32*factor, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.down1 = nn.Conv2d(32*factor, 32*factor, 4, 2, 1)

        self.time_step_encoder2 = nn.Sequential(
            nn.Linear(time_embedding_dim, 32*factor),
            nn.SiLU(),
            nn.Linear(32*factor, 32*factor)
        )

        self.block2 = nn.Sequential(
            nn.LayerNorm((32*factor, 32, 32)),
            nn.Conv2d(32*factor, 64*factor, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64*factor, 64*factor, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.down2 = nn.Conv2d(64*factor, 64*factor, 4, 2, 1)

        self.time_step_encoder3 = nn.Sequential(
            nn.Linear(time_embedding_dim, 64*factor),
            nn.SiLU(),
            nn.Linear(64*factor, 64*factor)
        )

        self.block3 = nn.Sequential(
            nn.LayerNorm((64*factor, 16, 16)),
            nn.Conv2d(64*factor, 128*factor, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128*factor, 128*factor, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128*factor, 128*factor, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128*factor, 128*factor, 4, 2, 1)
        )

        # The bottleneck
        self.time_step_encoder_mid = nn.Sequential(
            nn.Linear(time_embedding_dim, 128*factor),
            nn.SiLU(),
            nn.Linear(128*factor, 128*factor)
        )

        self.block_mid = nn.Sequential(
            nn.LayerNorm((128*factor, 7, 7)),
            nn.Conv2d(128*factor, 128*factor, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128*factor, 128*factor, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        # The second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128*factor, 128*factor, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128*factor, 128*factor, 2, 1)
        )

        self.time_step_encoder4 = nn.Sequential(
            nn.Linear(time_embedding_dim, 256*factor),
            nn.SiLU(),
            nn.Linear(256*factor, 256*factor)
        )

        self.block4 = nn.Sequential(
            nn.LayerNorm((256*factor, 16, 16)),
            nn.Conv2d(256*factor, 64*factor, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64*factor, 64*factor, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.up2 = nn.ConvTranspose2d(64*factor, 64*factor, 4, 2, 1)

        self.time_step_encoder5 = nn.Sequential(
            nn.Linear(time_embedding_dim, 128*factor),
            nn.SiLU(),
            nn.Linear(128*factor, 128*factor)
        )

        self.block5 = nn.Sequential(
            nn.LayerNorm((128*factor, 32, 32)),
            nn.Conv2d(128*factor, 32*factor, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32*factor, 32*factor, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.up3 = nn.ConvTranspose2d(32*factor, 32*factor, 4, 2, 1)

        self.time_step_encoder6 = nn.Sequential(
            nn.Linear(time_embedding_dim, 64*factor),
            nn.SiLU(),
            nn.Linear(64*factor, 64*factor)
        )
        self.block6 = nn.Sequential(
            nn.LayerNorm((64*factor, 64, 64)),
            nn.Conv2d(64*factor, 32*factor, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32*factor, 32*factor, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.LayerNorm((32*factor, 64, 64)),
            nn.Conv2d(32*factor, 32*factor, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32*factor, 32*factor, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.final_layer = nn.Conv2d(32*factor, 3, 3, 1, 1)

    def forward(self, x, t):
        t = self.time_step_embedding(t)
        n = len(x)
        output1 = self.block1(x + self.time_step_encoder1(t).reshape(n, -1, 1, 1))
        output2 = self.block2(self.down1(output1) + self.time_step_encoder2(t).reshape(n, -1, 1, 1))
        output3 = self.block3(self.down2(output2) + self.time_step_encoder3(t).reshape(n, -1, 1, 1))
        output_mid = self.block_mid(self.down3(output3) + self.time_step_encoder_mid(t).reshape(n, -1, 1, 1))
        output4 = torch.cat((output3, self.up1(output_mid)), dim=1)
        output4 = self.block4(output4 + self.time_step_encoder4(t).reshape(n, -1, 1, 1))
        output5 = torch.cat((output2, self.up2(output4)), dim=1)
        output5 = self.block5(output5 + self.time_step_encoder5(t).reshape(n, -1, 1, 1))
        output6 = torch.cat((output1, self.up3(output5)), dim=1)
        output6 = self.block6(output6 + self.time_step_encoder6(t).reshape(n, -1, 1, 1))
        output = self.final_layer(output6)
        return output


# In[ ]:


# Build the DDPM
ddpm_anime = DDPM(n_steps=n_steps, start_beta=start_beta, end_beta=end_beta, device=device)

# Print the model
print(ddpm_anime)


# ### Show forward process

# In[ ]:


# Sample the first image from the next batch, then demonstrate the forward process.
def show_forward(ddpm, loader, device):
    fig = plt.figure(figsize=(6, 1))

    for batch in loader:

        images = batch[0]

        fig.add_subplot(161)
        temp = np.transpose(images[0], (1, 2, 0))
        plt.title('original')
        plt.imshow((temp+1)/2)
        plt.axis('off')

        tensor_image = ddpm(images[:1].to(device), [int(0.1 * ddpm.n_steps) - 1])
        image = tensor_image.detach().cpu().numpy()
        fig.add_subplot(162)
        temp = np.transpose(image[0], (1, 2, 0))
        plt.title('10%')
        plt.imshow((temp+1)/2)
        plt.axis('off')

        tensor_image = ddpm(images[:1].to(device), [int(0.25 * ddpm.n_steps) - 1])
        image = tensor_image.detach().cpu().numpy()
        fig.add_subplot(163)
        temp = np.transpose(image[0], (1, 2, 0))
        plt.title('25%')
        plt.imshow((temp+1)/2)
        plt.axis('off')

        tensor_image = ddpm(images[:1].to(device), [int(0.5 * ddpm.n_steps) - 1])
        image = tensor_image.detach().cpu().numpy()
        fig.add_subplot(164)
        temp = np.transpose(image[0], (1, 2, 0))
        plt.title('50%')
        plt.imshow((temp+1)/2)
        plt.axis('off')

        tensor_image = ddpm(images[:1].to(device), [int(0.75 * ddpm.n_steps) - 1])
        image = tensor_image.detach().cpu().numpy()
        fig.add_subplot(165)
        temp = np.transpose(image[0], (1, 2, 0))
        plt.title('75%')
        plt.imshow((temp+1)/2)
        plt.axis('off')

        tensor_image = ddpm(images[:1].to(device), [int(1 * ddpm.n_steps) - 1])
        image = tensor_image.detach().cpu().numpy()
        fig.add_subplot(166)
        temp = np.transpose(image[0], (1, 2, 0))
        plt.title('100%')
        plt.imshow((temp+1)/2)
        plt.axis('off')
        break

# show_forward(dppm_anime, dataloader, device)


# ### Generate new images

# In[ ]:


"""
Provided with a DDPM model, a specified number of samples to generate, and a chosen device,
this function returns a set of freshly generated samples while also saving the .gif of the reverse process
"""
def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=25, gif_name="sampling_anime_face.gif", channel=3, height=64, width=64):
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, channel, height, width).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)
            
            if t > 0:
                z = torch.randn(n_samples, channel, height, width).to(device)
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                x = x + sigma_t * z

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])
                    

                # Reshaping batch (n, c, h, w) to be a square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)

    if channel == 1:
        for i in range(len(frames)):
            frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2RGB)

    # Storing the gif
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):    
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(frames[-1])
    
    return x



# ### Training loop

# In[ ]:


import torch.optim
def trainer(ddpm, dataloader, n_epochs, optim, loss_funciton, device, model_store_path):
    best_loss = float("inf")
    n_steps = ddpm.n_steps
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)
    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="green"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(dataloader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="blue")):
            # Load data
            x0 = batch[0].to(device)
            n = len(x0)

            # Pick random noise for each of the images in the batch
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)
        
            # Compute the noisy image based on x0 and the time step
            noises = ddpm(x0, t, eta)
            
            # Get model estimation of noise based on the images and the time step
            eta_theta = ddpm.backward(noises, t.reshape(n, -1))
            
            # Optimize the Mean Squared Error (MSE) between the injected noise and the predicted noise
            loss = loss_funciton(eta_theta, eta)

            # First, initialize the optimizer's gradient and then update the network's weights
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # Aggregate the loss values from each iteration to compute the loss value for an epoch
            epoch_loss += loss.item() * len(x0) / len(dataloader.dataset)

            # Save Losses for plotting later
            loss_list.append(loss.item())

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"
        # Show images generated at the epoch
        show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}", save=True, filename=f"anime_face_log/epoch_{epoch + 1}.png")

        # If the current loss is better than the previous one, then store the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), model_store_path)
            log_string += " <Store the best model.>"

        print(log_string)
        lr_scheduler.step()


# # It's your turn

# You need to train the diffusion model to generate grayscale images or color ones by the Anime Face dataset.
# 
# If you choose to implement the color images generation model, you will get additional bonus.
# 
# Please note that the size of the input images should be 64x64.
# 
# 

# In[ ]:


######################################################################################
# TODO: Design the diffusion process for the Anime Face dataset
# Implementation B.1-4
######################################################################################


trainer(ddpm_anime, dataloader, n_epochs=n_epochs, optim=Adam(ddpm_anime.parameters(), lr), loss_funciton=nn.MSELoss(), device=device, model_store_path=model_store_path)


# #Plot loss values

# In[ ]:


######################################################################################
# TODO: Plot the loss values of DDPM for the Anime Face dataset
# Implementation B.1-5
######################################################################################
plt.plot(loss_list)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend(['Loss'], loc='upper right')
plt.savefig('loss_anime.jpg')
plt.close()


# #Plot generated images in 5*5 grid

# In[ ]:


######################################################################################
# TODO: Store your generate images in 5*5 grid for the Anime Face dataset
# Implementation B.1-6
######################################################################################
ddpm_anime.load_state_dict(torch.load(model_store_path))
nums_samples = 25
samples = generate_new_images(ddpm_anime, n_samples=nums_samples, device=device)
show_images(samples, "Final result", save=True, filename="result_anime.jpg")



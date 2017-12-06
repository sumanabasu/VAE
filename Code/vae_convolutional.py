# Convolutional Ecoder-Decoder
# CIFAR10 pytorch downloader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision
import numpy as np

# CIFAR10 dataset
dataset = datasets.CIFAR10(root='/home/ml/sbasu11/Documents/VAE/CIFAR10/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=100, 
                                          shuffle=True)

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# VAE model
class VAE(nn.Module):
    def __init__(self, z_dim = 20, linear_dim = 128):
        super(VAE, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, stride = 2, padding = 1)    
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 2, padding = 1)
        self.lineare = nn.Linear(linear_dim, z_dim)

        self.lineard = nn.Linear(z_dim, linear_dim)
        self.tconv1 = nn.ConvTranspose2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 2, padding = 0)
        self.tconv2 = nn.ConvTranspose2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2, padding = 1)
        self.tconv3 = nn.ConvTranspose2d(in_channels = 16, out_channels = 3, kernel_size = 2, stride = 2, padding = 1)

        self.relu = nn.ReLU(True)
        self. tanh = nn.Tanh()

    def encoder(self, x):
        #print x.size()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        mu = self.lineare(x)
        log_var = self.lineare(x)
        return mu, log_var

    def decoder(self, z):
        z = self.relu(self.lineard(z))
        z = z.view(-1, 8, 4, 4)
        z = self.relu(self.tconv1(z))
        z = self.relu(self.tconv2(z))
        z = self.relu(self.tconv3(z))
        z = self.tanh(z)
        return z

    
    def reparametrize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        eps = to_var(torch.randn(mu.size(0), mu.size(1)))
        z = mu + eps * torch.exp(log_var/2)    # 2 for convert var to std
        return z
                     
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var
    
    def sample(self, z):
        return self.decoder(z)
    
vae = VAE()

if torch.cuda.is_available():
    vae.cuda()
    
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
iter_per_epoch = len(data_loader)
data_iter = iter(data_loader)

# fixed inputs for debugging
fixed_z = to_var(torch.randn(100, 20))
fixed_x, _ = next(data_iter)
torchvision.utils.save_image(fixed_x.cpu(), './data/real_images.png')
fixed_x = to_var(fixed_x.view(fixed_x.size(0), 3, 32, 32))

for epoch in range(50):
    for i, (images, _) in enumerate(data_loader):
        #print images.size()
        
        images = to_var(images.view(images.size(0), 3, 32, 32))
        out, mu, log_var = vae(images)
        
        # Compute reconstruction loss and kl divergence
        reconst_loss = F.binary_cross_entropy(out, images, size_average=False)
        kl_divergence = torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var -1))
        
        # Backprop + Optimize
        total_loss = reconst_loss + kl_divergence
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "Reconst Loss: %.4f, KL Div: %.4f" 
                   %(epoch+1, 50, i+1, iter_per_epoch, total_loss.data[0], 
                     reconst_loss.data[0], kl_divergence.data[0]))
    
    # Save the reconstructed images
    print "saving1"
    reconst_images, _, _ = vae(fixed_x)
    reconst_images = reconst_images.view(reconst_images.size(0), 3, 32, 32)
    torchvision.utils.save_image(reconst_images.data.cpu(), 
        './data/reconst_images_%d.png' %(epoch+1))
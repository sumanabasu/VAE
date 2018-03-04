from __future__ import division

import torch
import argparse
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
#import torch.distributions
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt

class Encoder(torch.nn.Module):
    def __init__(self, D_in, D_out, D_layers=[]):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_layers[0])
        self.linear2 = torch.nn.Linear(D_layers[0], D_layers[1])
        self.linear3 = torch.nn.Linear(D_layers[1], D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

class Decoder(torch.nn.Module):
    def __init__(self, D_in, D_out, D_layers=[]):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_out, D_layers[1])
        self.linear2 = torch.nn.Linear(D_layers[1], D_layers[0])
        self.linear3 = torch.nn.Linear(D_layers[0], D_in)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

class VAE(torch.nn.Module):

    def __init__(self, K, N, temperature, hidden_layers, iscuda):
        """
        Categorical Variational Autoencoder
        K: Number of Cateories or Classes
        N: Number of Categorical distributions 
        N x K: Dimension of latent variable
        hidden_layers: A list containing number of nodes in each hidden layers
                        of both encoder and decoder
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, K*N, hidden_layers)
        self.decoder = Decoder(input_dim, K*N, hidden_layers[::-1])
        self.K=K
        self.N=N
        self.temperature = temperature
        self.iscuda = iscuda

    def _sample_latent(self, tou):
        """
        Return the latent normal sample y ~ gumbel_softmax(x)
        """
        eps = 1e-20
        
        # generates a h_enc.size() shaped batch of reparameterized 
        # Gumbel samples with location = 0, sclae = 1
        U = torch.from_numpy(np.random.uniform(0, 1, size=self.hidden.size())).float()

        # for doing operation between Variable and Tensor, a tensor has to be wrapped
        # insider Variable. However, set requires_grad as False so that back propagation doesn't
        # pass through it
        # gumbel sample is -log(-log(U))
        g = Variable(-torch.log(-torch.log(U + eps) + eps), requires_grad=False)
        if self.iscuda:
            g = g.cuda()

        # Gumbel-Softmax samples are - softmax((probs + gumbel(0,1).sample)/temperature)
        y = self.hidden + g
        softmax = torch.nn.Softmax(dim=-1) # -1 indicates the last dimension

        return softmax(y/1.0)

    def forward(self, x):
        # dynamic binarization of input
        t = Variable(torch.rand(x.size()), requires_grad=False)
        if self.iscuda:
            t = t.cuda()

        net = t < x
    
        h_enc = self.encoder(net.float())
        tou = Variable(torch.from_numpy(np.array([self.temperature])), requires_grad=False)
        self.hidden = h_enc.view(-1, self.N, self.K)
        bsize = self.hidden.size()[0]
        self.latent = self._sample_latent(tou)
        x_hat = self.decoder(self.latent.view(bsize,-1))
        return x_hat

    def loss_fn(self, x, x_hat):
        """
        Total Loss = Reconstruction Loss + KL Divergence
        x = input to forward()
        x_hat = output of forward()
        Reconstruction Loss = binary cross entropy between inputs and outputs
        KL Divergence = KL Divergence between gumbel softmax distributions with 
                        self.hidden and uniform log-odds
        """
        eps = 1e-20 # to avoid log of 0

        # Reconstruction Loss
        # Instantiate Bernoulli distribution with x_hat as log odds for each pixel
        #Then, binary_cross_entropy = log probability evaluated at x
        softmax = torch.nn.Softmax(dim=-1)
        x_prob = softmax(x_hat)
        #p_x = torch.distributions.Bernoulli(probs=x_hat)
        recons_loss = torch.sum(x * torch.log(x_prob + eps), dim=1)
        #recons_loss = torch.sum(p_x.log_prob(x), dim=1)

        # KL Divergence = entropy (self.latent) - cross_entropy(self.latent, uniform log-odds)
        q_y = softmax(self.hidden) # convert hidden layer values to probabilities
        kl1 = q_y * torch.log(q_y + eps) # entropy (self.latent)
        kl2 = q_y * np.log((1.0/self.K) + eps)

        KL_divergence = torch.sum(torch.sum(kl1 - kl2, 2),1)
        
        # total loss = reconstruction loss + KL Divergence
        loss = -torch.mean(recons_loss - KL_divergence)
        self.recons_loss = -torch.mean(recons_loss).data[0]
        self.kl_loss = -torch.mean(-KL_divergence).data[0]
        return loss 
        #return -torch.mean(recons_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE MNIST Example')      
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    args = parser.parse_args()
    iscuda = not args.no_cuda and torch.cuda.is_available()
    print "Using Cuda: ", iscuda

    #iscuda = True
    input_dim = 28 * 28
    batch_size = 100
    hidden1_size = 512
    hidden2_size = 256
    K = 10 #number of classes
    N = 20 #number of categorical distributions
    tau0 = 1
    epochs = 85 

    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    vae = VAE(K, N, tau0, [hidden1_size, hidden2_size], iscuda)
    if iscuda:
        vae.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=1e-2)
    l = 0
    rl = 0
    kl = 0
    for e in range(1,epochs):
        l = 0
        rl = 0
        kl = 0
        for i, data in enumerate(dataloader, 0):
            inputs, _ = data
            if iscuda:
                inputs = inputs.cuda()

            inputs = Variable(inputs.resize_(batch_size, input_dim))
            optimizer.zero_grad()
            
            outputs = vae(inputs)

            loss = vae.loss_fn(inputs, outputs)
            l += loss.data[0]
            
            rl += vae.recons_loss
            kl += vae.kl_loss

            loss.backward()
            optimizer.step()

        print "debug-epoch: ", e, ", error: ", l/i, ", recons error: ", rl/i, ", kl divergence: ", kl/i

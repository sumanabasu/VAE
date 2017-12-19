from __future__ import division

import torch
import argparse
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt

class Encoder(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

class Decoder(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.sigmoid(self.linear3(x))

class VAE(torch.nn.Module):

    def __init__(self, encoder, decoder, K, N, iscuda):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.softmax = torch.nn.Softmax()
        self.K=K
        self.N=N
        self.iscuda = iscuda

    def _sample_latent(self, h_enc, temperature):
        """
        Return the latent normal sample y ~ gumbel_softmax(x)
        """

        eps = 1e-20
        
        h_enc = h_enc.view(-1, self.K)

        self.qy = self.softmax(h_enc)
        self.logqy = torch.log(self.qy + eps)

        U = torch.from_numpy(np.random.uniform(0, 1, size=h_enc.size())).float()
        # for doing operation between Variable and Tensor, a tensor has to be wrapped 
        # insider Variable. However, set requires_grad as False so that back propagation doesn't 
        # pass through it
        g = Variable(-torch.log(-torch.log(U + eps) + eps), requires_grad=False)
        if self.iscuda:
            g = g.cuda()
        y = self.softmax((h_enc + g)/temperature)
        y = y.view(-1, self.N * self.K)

        return y

    def forward(self, x, temperature):
        h_enc = self.encoder(x)
        z = self._sample_latent(h_enc, temperature)
        return self.decoder(z)

    def total_loss(self, inputs, outputs):
        kltmp = self.qy * (self.logqy - np.log(1/self.K))
        kltmp = kltmp.view(-1, self.N, self.K)
        KL = torch.sum(kltmp) # sum over dimensions 1 and 2
        batch_size = kltmp.size()[0]
        KL /= (batch_size * self.N)
        
        BCE = torch.mean(F.binary_cross_entropy(outputs, inputs))

        return torch.mean(BCE-KL)

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
    N = 30 #number of categorical distributions
    tau0 = 1
    np_temp=tau0
    epochs = 85 
    ANNEAL_RATE=0.00003
    MIN_TEMP=0.5

    transform = transforms.Compose(
        [transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    print('Number of samples: ', len(mnist))
    
    encoder = Encoder(input_dim, hidden1_size, hidden2_size, K*N)
    decoder = Decoder(K*N, hidden2_size, hidden1_size, input_dim)
    vae = VAE(encoder, decoder, K, N, iscuda)
    if iscuda:
        vae.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            inputs, classes = data
            if iscuda:
                inputs = inputs.cuda()
            inputs, classes = Variable(inputs.resize_(batch_size, input_dim)), Variable(classes)
            optimizer.zero_grad()
            
            outputs = vae(inputs, np_temp)
            np_temp=np.maximum(tau0*np.exp(-ANNEAL_RATE*(i * epoch)),MIN_TEMP)

            loss = vae.total_loss(inputs, outputs)
            loss.backward()
            optimizer.step()
            l = loss.data[0]
        print("epoch: {}, loss:{}".format(epoch, l))

    plt.imsave("mnist_gumbel.png", vae(inputs, tou).data[0].numpy().reshape(28, 28), cmap='gray')

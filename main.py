import matplotlib.pyplot as plt
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torchvision

cuda = True if torch.cuda.is_available() else False

# Parameters
LATENT_DIM = 128
IMG_SIZE = 32
IMG_SHAPE = (1, IMG_SIZE, IMG_SIZE)
lr = 0.0002
BETAS = (0.5, 0.999)
N_CRITIC = 5
N_EPOCHS = 10
SAMPLE_INTERVAL = 400
# Loss weight for gradient penalty
LAMBDA_GP = 10
BATCH_SIZE = 64


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             *block(LATENT_DIM, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(IMG_SHAPE))),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.shape[0], *IMG_SHAPE)
#         return img
#
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(int(np.prod(IMG_SHAPE)), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#         )
#
#     def forward(self, img):
#         img_flat = img.view(img.shape[0], -1)
#         validity = self.model(img_flat)
#         return validity
#
#

def initialize_weights(net):
    for m in net.modules():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
          nn.init.normal_(m.weight.data, 1.0, 0.02)
          nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, dim=LATENT_DIM):
        super().__init__()
        self.dim = LATENT_DIM
        # self.model = model

        self.linear = nn.Sequential(
            nn.Linear(self.dim, 4 * 4 * 4 * self.dim),
            # nn.BatchNorm1d(num_features=4*4*4*self.dim),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4 * self.dim, out_channels=2 * self.dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=2 * self.dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=2 * self.dim, out_channels=self.dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=self.dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.dim, out_channels=1, kernel_size=2, stride=2),
            nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 4 * self.dim, 4, 4)
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 1
        self.inner_dim = LATENT_DIM
        # self.model = model

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=self.inner_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.inner_dim, out_channels=2 * self.inner_dim, kernel_size=3, stride=2,
                      padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=2 * self.inner_dim, out_channels=4 * self.inner_dim, kernel_size=3, stride=2,
                      padding=1),
            nn.LeakyReLU()
        )

        self.FC = nn.Linear(4 * 4 * 4 * self.inner_dim, 1)
        self.Sigmoid = nn.Sigmoid()

        initialize_weights(self)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 4 * 4 * 4 * self.inner_dim)
        x = self.FC(x)
        # if self.model == 'DCGAN':
        #     x = self.Sigmoid(x)
        return x.view(-1, 1)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(generator, discriminator, optimizer_G, optimizer_D, data_loader, n_epochs):
    # ----------
    #  Training
    # ----------
    D_losses, G_losses = [], []
    batches_done = 0
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(data_loader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], LATENT_DIM))))

            # Generate a batch of images pytorch
            fake_imgs = generator(z)

            # Real images pytorch
            real_validity = discriminator(real_imgs)
            # Fake images pytorch
            fake_validity = discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % N_CRITIC == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images pytorch
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images pytorch
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                if i % 100 == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, n_epochs, i, len(data_loader), d_loss.item(), g_loss.item())
                    )

                if batches_done % SAMPLE_INTERVAL == 0:
                    save_image(fake_imgs.data[:25], "./images/%d.png" % batches_done, nrow=5, normalize=True)

                batches_done += N_CRITIC
                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())
    plt.figure()
    epochs_range = np.arange(1, n_epochs)
    plt.plot(epochs_range, D_losses)
    plt.plot(epochs_range, G_losses)
    plt.title('Generator and Discriminator losses')
    plt.legend()
    plt.show()



# Load data
train_data = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        "./data/FashionMNIST",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=BETAS)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=BETAS)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
train(generator, discriminator, optimizer_G, optimizer_D, train_data, n_epochs=2)


print('End')

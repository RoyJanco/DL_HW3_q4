import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torch import Tensor
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torchvision
from matplotlib import pyplot as plt
import torchvision.utils as vutils

cuda = True if torch.cuda.is_available() else False
# Decide which device we want to run on
global device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU available')
else:
    device = torch.device('cpu')
    print('GPU not available, training on CPU.')

# Parameters
MODEL = 'WGAN' # Choose between 'WGAN' or 'DCGAN'
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
DCGAN_FEATURES = 64  # Used for the DCGAN


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



class DCGAN_Generator(nn.Module):
    def __init__(self):
        super(DCGAN_Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( LATENT_DIM, DCGAN_FEATURES * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(DCGAN_FEATURES * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(DCGAN_FEATURES * 8, DCGAN_FEATURES * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DCGAN_FEATURES * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( DCGAN_FEATURES * 4, DCGAN_FEATURES * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DCGAN_FEATURES * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( DCGAN_FEATURES * 2, 1, 4, 2, 1, bias=False),
            nn.Tanh(),

            # nn.BatchNorm2d(DCGAN_FEATURES),
            # nn.ReLU(True),
            # # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d( DCGAN_FEATURES, 1, 4, 2, 1, bias=False),
            # nn.Tanh()
            # # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        input = input.view(-1, LATENT_DIM, 1, 1)
        return self.main(input)


class DCGAN_Discriminator(nn.Module):
    def __init__(self):
        super(DCGAN_Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(1, DCGAN_FEATURES, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(DCGAN_FEATURES, DCGAN_FEATURES * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DCGAN_FEATURES * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(DCGAN_FEATURES * 2, DCGAN_FEATURES * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DCGAN_FEATURES * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(DCGAN_FEATURES * 4, 1, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(DCGAN_FEATURES * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(DCGAN_FEATURES * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.main(input)
        out = out.view(-1, 1)
        return out


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
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
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


def train_WGAN(generator, discriminator, optimizer_G, optimizer_D, data_loader, n_epochs):
    # ----------
    #  Training
    # ----------
    D_losses, G_losses = [], []
    batches_done = 0
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(data_loader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor)).to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], LATENT_DIM)))).to(device)

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

                if i % 20 == 0:
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
    # epochs_range = np.arange(1, n_epochs+1)
    plt.plot(D_losses, label='Discriminator')
    plt.plot(G_losses, label='Generator')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator losses')
    plt.show()


def train_DCGAN(generator, discriminator, optimizer_G, optimizer_D, data_loader, n_epochs):
    # ----------
    #  Training
    # ----------
    D_losses, G_losses = [], []
    batches_done = 0
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(data_loader):


            # Configure input
            real_imgs = Variable(imgs.type(Tensor)).to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], LATENT_DIM)))).to(device)

            # Generate a batch of images pytorch
            fake_imgs = generator(z)

            # Real images pytorch
            real_validity = discriminator(real_imgs)
            # Fake images pytorch
            fake_validity = discriminator(fake_imgs)

            # Adversarial loss. We minimize this term:
            d_loss = -(torch.mean(torch.log(real_validity)) + torch.mean(torch.log(1 - fake_validity)))

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % 1 == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images pytorch
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images pytorch
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(torch.log(fake_validity))

                g_loss.backward()
                optimizer_G.step()

                if i % 20 == 0:
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
    # epochs_range = np.arange(1, n_epochs+1)
    plt.plot(D_losses, label='Discriminator')
    plt.plot(G_losses, label='Generator')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator losses')
    plt.show()


def train_generic(model, generator, discriminator, optimizer_G, optimizer_D, data_loader, n_epochs):
    # ----------
    #  Training
    # ----------
    D_losses, G_losses = [], []
    batches_done = 0
    if model == 'WGAN':
        disc_iterations = N_CRITIC
    else:
        disc_iterations = 1
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(data_loader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor)).to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], LATENT_DIM)))).to(device)

            # Generate a batch of images pytorch
            fake_imgs = generator(z)

            # Real images pytorch
            real_validity = discriminator(real_imgs)
            # Fake images pytorch
            fake_validity = discriminator(fake_imgs)
            if model == 'WGAN':
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty
            else:
                # Adversarial loss. We minimize this term:
                d_loss = -(torch.mean(torch.log(real_validity)) + torch.mean(torch.log(1 - fake_validity)))

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every disc_iterations steps
            if i % disc_iterations == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images pytorch
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images pytorch
                fake_validity = discriminator(fake_imgs)
                if model == 'WGAN':
                    g_loss = -torch.mean(fake_validity)
                else:
                    g_loss = -torch.mean(torch.log(fake_validity))


                g_loss.backward()
                optimizer_G.step()

                if i % 20 == 0:
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
    # epochs_range = np.arange(1, n_epochs+1)
    plt.plot(-D_losses, label='Negative discriminator loss')
    plt.plot(G_losses, label='Generator loss')
    plt.legend()
    plt.xlabel('Generator iterations')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator losses')
    plt.show()


def train_DCGAN_2(generator, discriminator, optimizer_G, optimizer_D, data_loader, n_epochs):
    # Training Loop
    real_label = 1
    fake_label = 0
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(n_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(data_loader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizer_D.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizer_G.step()

            # Output training stats
            if i % 20 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, n_epochs, i, len(data_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if i % SAMPLE_INTERVAL == 0:
                save_image(fake.data[:25], "./images/%d.png" % i, nrow=5, normalize=True)

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == n_epochs-1) and (i == len(data_loader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

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

# WGAP-GP Training
# Initialize generator and discriminator
# generator = Generator()
# discriminator = Discriminator()
#
# if cuda:
#     generator.cuda()
#     discriminator.cuda()
#
# # Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=BETAS)
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=BETAS)
#
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# train(generator, discriminator, optimizer_G, optimizer_D, train_data, n_epochs=2)

model = 'WGAN'
if model == 'WGAN':
    # WGAP-GP Training
    generator = Generator()
    discriminator = Discriminator()
else:
    # DCGAN Training
    generator = DCGAN_Generator()
    discriminator = DCGAN_Discriminator()

generator.to(device)
discriminator.to(device)
# if cuda:
#     generator.cuda()
#     discriminator.cuda()

print(generator)
print(discriminator)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=BETAS)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=BETAS)

# train_DCGAN(generator, discriminator, optimizer_G, optimizer_D, train_data, n_epochs=1)
train_generic(model, generator, discriminator, optimizer_G, optimizer_D, train_data, n_epochs=1)


print('End')

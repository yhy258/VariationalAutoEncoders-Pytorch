import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.utils import save_image
from torchvision import datasets, transforms

from vae_model_frame import *

import numpy as np

from tqdm import tqdm


# Simple Implementation
class StandardVAE(VAE_Frame):
    def __init__(self, input_shape=(3, 32, 32)):
        super().__init__()
        self.input_shape = input_shape
        img_in_channels = input_shape[0]

        if input_shape[1] == 32:
            hid_dims = [32, 64, 128]
            latent_dim = 64
            self.start_width = 4

        elif input_shape[1] == 64:
            hid_dims = [64, 128, 256, 512]
            latent_dim = 128
            self.start_width = 4

        else:
            raise NotImplementedError("input_shape parameter only 32, 64")

        self.latent_dim = latent_dim

        # Encoder
        self.encoder_modules = []

        in_channels = img_in_channels

        for h_dim in hid_dims:
            self.encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        self.encoder_modules = nn.Sequential(*self.encoder_modules)
        self.encode_mu = nn.Linear(hid_dims[-1] * self.start_width * self.start_width, latent_dim)
        self.encode_logvar = nn.Linear(hid_dims[-1] * self.start_width * self.start_width, latent_dim)

        # Decoder
        self.proj_decode = nn.Linear(latent_dim, hid_dims[-1] * self.start_width * self.start_width)

        self.decoder_modules = []
        hid_dims.reverse()

        for i in range(len(hid_dims) - 1):
            self.decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hid_dims[i],
                                       hid_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hid_dims[i + 1]),
                    nn.ReLU())
            )

        self.decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hid_dims[-1],
                                   hid_dims[-1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hid_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hid_dims[-1], out_channels=img_in_channels,
                          kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        )
        self.decoder_modules = nn.Sequential(*self.decoder_modules)

    def encoder(self, x: Tensor) -> List[Tensor]:
        x = self.encoder_modules(x)
        x = x.view(x.size(0), -1)
        mu = self.encode_mu(x)
        log_var = self.encode_logvar(x)
        return [mu, log_var]

    def decoder(self, z: Tensor) -> Tensor:
        z = self.proj_decode(z)
        z = z.view(z.size(0), -1, self.start_width, self.start_width)
        return self.decoder_modules(z)

    def reparameterize_trick(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return std * eps + mu

    def generate(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]

    def sample(self, samples_num: input, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> List[
        Tensor]:
        latent = torch.randn(samples_num, self.latent_dim).to(device)
        return self.decoder(latent)

    def forward(self, x: Tensor) -> List[Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize_trick(mu, log_var)
        output = self.decoder(z)
        return output, mu, log_var

    def loss_function(self, pred: Tensor, real_img: Tensor, mu: Tensor, log_var: Tensor):
        # VAE 논문 마지막 부분에 나와있음. (ELBO를 수식적으로 유도한 결과)
        recons_loss = nn.BCELoss(size_average=False)(pred, real_img) / pred.size(0)
        kld_loss = ((mu ** 2 + log_var.exp() - 1 - log_var) / 2).mean()
        loss = recons_loss + kld_loss
        return loss, recons_loss, kld_loss

# Example
if __name__ == "__main__":

    dataset = datasets.CIFAR10(root='./data', train=True,
                               transform=transforms.Compose([transforms.ToTensor()]),
                               download=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae_model = StandardVAE(input_shape=(3, 32, 32)).to(DEVICE)
    optim = torch.optim.Adam(params=vae_model.parameters(), lr=3e-4)

    for epoch in range(30):
        print("Epoch : {}/30".format(epoch + 1))
        losses = []
        recons_losses = []
        kld_losses = []

        for img, _ in tqdm(dataloader):
            img = img.to(DEVICE)
            pred, mu, log_var = vae_model(img)
            total_loss, recons_loss, kld_loss = vae_model.loss_function(pred, img, mu, log_var)

            losses.append(total_loss.item())
            recons_losses.append(recons_loss.item())
            kld_losses.append(kld_loss.item())

            optim.zero_grad()
            total_loss.backward()
            optim.step()

        print("Total Losses : {:.5f} Reconstruction Losses : {:.5f} KL Divergence : {:.5f}".format(np.mean(losses),
                                                                                                   np.mean(
                                                                                                       recons_losses),
                                                                                                   np.mean(kld_losses)))

    """
        Reconstruction
    """
    imgs, _ = next(iter(dataloader))
    imgs = imgs.to(DEVICE)
    recons_datas = vae_model.generate(imgs)
    print(recons_datas.size())
    save_image(recons_datas.view(32, 3, 32, 32)[:25],
               "Standard_VAE_Reconstruction.png", nrow=5, normalize=True)

    """
        Sampling
    """
    samples = vae_model.sample(25)
    save_image(samples.view(32, 3, 32, 32)[:25], "Standard_VAE_Sampling.png",
               nrow=5, normalize=True)





import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms

import numpy as np

from vq_vae_modules import *
from vae_model_frame import *

from tqdm import tqdm

# Simple Implementation
class VQ_VAE(VAE_Frame):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, commitment_weight, embedding_dim,
                 num_embeddings, input_shape=(3, 32, 32), decay=0.99):
        super().__init__()
        self.input_shape = input_shape

        img_in_channels = input_shape[0]

        self.encoder_modules = Encoder(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.proj_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, 1, stride=1)

        self.decoder_modules = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

        if decay > 0:
            self.vq_model = EMA_VQ(num_embeddings, embedding_dim, commitment_weight, decay)
        else:
            self.vq_model = VQ(num_embeddings, embedding_dim, commitment_weight)

    def encoder(self, x: Tensor) -> List[Tensor]:
        z = self.encoder_modules(x)
        z = self.proj_vq_conv(z)
        loss, quantized, _ = self.vq_model(z)
        return loss, quantized

    def decoder(self, quantized: Tensor) -> Tensor:
        x_recon = self.decoder_modules(quantized)

        return x_recon

    def generate(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]

    def forward(self, x: Tensor) -> List[Tensor]:
        loss, quantized = self.encoder(x)
        x_recon = self.decoder(quantized)
        return x_recon, loss

    def loss_function(self, latent_loss, pred: Tensor, real_img: Tensor, data_variance):
        # VAE 논문 마지막 부분에 나와있음. (ELBO를 수식적으로 유도한 결과)
        recons_loss = F.mse_loss(pred, real_img) / data_variance
        loss = recons_loss + latent_loss
        return loss, recons_loss


if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = datasets.CIFAR10(root="data", train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(0.5, 1.0)
                               ]))
    data_variance = np.var(dataset.data / 255.0)

    ## Create a dataloader
    # Batch size during training
    batch_size = 256

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae_model = VQ_VAE(num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, commitment_weight=0.25,
                       embedding_dim=64, num_embeddings=512, input_shape=(3, 32, 32)).to(DEVICE)
    optim = torch.optim.Adam(params=vae_model.parameters(), lr=1e-3, amsgrad=False)

    for epoch in range(40):  # 10000
        print("Epoch : {}/40".format(epoch + 1))
        losses = []
        recons_losses = []
        kld_losses = []

        for img, _ in tqdm(dataloader):
            img = img.to(DEVICE)
            pred, latent_loss = vae_model(img)
            total_loss, recons_loss = vae_model.loss_function(latent_loss, pred, img, data_variance)

            losses.append(total_loss.item())
            recons_losses.append(recons_loss.item())

            optim.zero_grad()
            total_loss.backward()
            optim.step()

        print("Total Losses : {:.5f} Reconstruction Losses : {:.5f}".format(np.mean(losses), np.mean(recons_losses)))


    """
        Reconstruction
    """
    imgs, _ = next(iter(dataloader))
    imgs = imgs.to(DEVICE)
    recons_datas = vae_model.generate(imgs)
    print(recons_datas.size())
    save_image(recons_datas.view(batch_size, 3, 32, 32)[:25],
               "/VQ_VAE_Reconstruction.png", nrow=5, normalize=True)





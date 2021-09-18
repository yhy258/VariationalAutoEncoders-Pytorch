import torch
import torch.nn as nn
import torch.nn.functional as F


class VQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_weight):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding_table = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.embedding_table.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)  # 일정 구간에서의 uniform
        self.commitment_weight = commitment_weight

    def forward(self, x):  # [B, C, H, W]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        flat_x = x.view(-1, self.embedding_dim)  # [B*H*W., C]

        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding_table.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding_table.weight.t()))  # (x-y)^2
        # distance output shape => [B*H*W, num_embeddings)] by broadcasting -> embedding_dim 상에서 서로 모두 비교 -> BHW 각각에 대해 가장 가까운 embedding indices 골라오기

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [B*H*W, 1, 1]
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings,
                                device=x.device)  # [B*H*W, num_embeddings]
        encodings.scatter_(1, encoding_indices, 1)  # one hot encodding

        quantized = torch.matmul(encodings, self.embedding_table.weight).view(
            x.size())  # indexing한 결과. (num_embeddings (category)에 대해서 내가 뽑았던 얘들만 가져오기.)
        # x.size () = [B, H, W , C] Channel이 뒤에 있음을 기억하자!

        z_e_loss = F.mse_loss(quantized.detach(), x)  # detach -> stop gradient
        z_q_loss = F.mse_loss(quantized, x.detach())

        loss = z_q_loss + self.commitment_weight * z_e_loss

        """
            중요한 부분!! 이렇게 하는 이유.
            우리는 NN을 기준으로 한 Indexing을 통해 quantization을 진행하는데, 이렇게 되면 역전파 시 gradient가 진행 될 수 없음.
            decoder의 gradient를 그대로 encoder의 outpu(z_e)에 부여해주기 위해 이런식으로 코드를 짠다.
            이렇게 짠다 하더라도 어차피 가장 가까운 embedding 요소에 대해 gradient를 부여하기 때문에 해당 gradient의 경향성은 유효하다.
        """
        quantized = x + (quantized - x).detach()

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encodings


# In paper, Appendix A.
class EMA_VQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_weight, decay, eps=1e-5):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding_table = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding_table.weight.data.normal_()

        self.commitment_weight = commitment_weight

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))  # ema cluster sizde

        self.ema_weight = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))
        self.ema_weight.data.normal_()  # 초기화

        self.decay = decay
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()

        flat_x = x.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True) + torch.sum(self.embedding_table.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding_table.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding_table.weight).view(x.size())

        # training시 EMA로 적용
        if self.training:
            # 논문에서 ema_weight == m
            # cluster size == N
            self._ema_cluster_size = self._ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size.data)  # 매칭된 것들 갯수.
            # 클러스터 사이즈에 라플라스 smoothing 적용.
            self._ema_cluster_size = ((self._ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n)
            dw = torch.matmul(encodings.t(), flat_x)
            self.ema_weight = nn.Parameter(self.ema_weight * self.decay + (1 - self.decay) * dw)
            self.embedding_table.weight = nn.Parameter(self.ema_weight / self._ema_cluster_size.unsqueeze(1))  #

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        loss = self.commitment_weight * e_latent_loss

        quantized = x + (quantized - x).detach()

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encodings


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)
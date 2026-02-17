import torch
import torch.nn as nn
import torch.nn.functional as F
import filter


class Model(nn.Module):

    def __init__(self, in_channels, out_features: int = 3, features: int = 16):
        super(Model, self).__init__()

        self.features = features

        self.conv_layer = nn.Sequential(
            self._conv2d_block(in_channels=in_channels, out_channels=features, kernel_size=5, stride=1, padding=2),
            self._conv2d_block(in_channels=features, out_channels=features*4, stride=2),
            self._conv2d_block(in_channels=features*4, out_channels=features*8, stride=2),
            self._conv2d_block(in_channels=features*8, out_channels=features*16, stride=4),
            self._conv2d_block(in_channels=features*16, out_channels=features*64, stride=1),
        )

        self.shared_mpls = nn.Sequential(
            self._shared_mlp(features*64*2, features*64),
            self._shared_mlp(features*64, features*16),
            self._shared_mlp(features*16, features),
            self._shared_mlp(features, out_features),

        )

    def _conv2d_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):

        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True
            ),  
            nn.LeakyReLU(0.2, inplace=True)
        )

    @staticmethod
    def _shared_mlp(in_channels, out_channels):
        
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        batch_size = x.shape[0]
        x = self.conv_layer(x)

        x = x.view(batch_size, self.features*64, -1)
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x = torch.concat([x_max, x_avg], dim=1)

        x = self.shared_mpls(x)

        x[:,0:1] = F.tanh(x[:,0:1])

        return x


if __name__ == "__main__":

    random = torch.randn(1, 1, 64, 64)
    print(random.shape)
    model = Model(in_channels=1, out_features=5)
    out = model(random)
    print(out.shape)
    torch.save(model.state_dict(), "Model_Weights.pth")

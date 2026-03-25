import torch
import torch.nn as nn
import torch.nn.functional as F
import filter


class Model(nn.Module):

    def __init__(self, in_channels: int = 3, out_features: int = 4, features: int = 16):
        super(Model, self).__init__()

        self.features = features

        self.conv_layer = nn.Sequential(
            self._conv2d_block(in_channels=in_channels, out_channels=features, kernel_size=5, stride=1, padding=2),
            self._conv2d_block(in_channels=features, out_channels=features*4, kernel_size=5, stride=3),
            self._conv2d_block(in_channels=features*4, out_channels=features*8, kernel_size=5, stride=3),
            self._conv2d_block(in_channels=features*8, out_channels=features*32, kernel_size=5, stride=1),
        )

        self.strokes_mpls = nn.Sequential(
            self._stroke_block(in_channels=out_features, out_channels=features*64),
            self._stroke_block(in_channels=features*64, out_channels=features*64),
        )

        # in_channel = (conv layers out channels) * 2 + (strokes mlps out channels) * 2
        first_in_channel = (features * 32) * 2 + (features * 64) * 2

        self.shared_mpls = nn.Sequential(
            self._shared_mlp(first_in_channel, features*32),
            self._shared_mlp(features*32, features*16),
            self._shared_mlp(features*16, features),
            nn.Linear(features, out_features)
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
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def _stroke_block(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )

    @staticmethod
    def _shared_mlp(in_channels, out_channels):
        
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )

    def forward(self, x):

        image, strokes = x

        batch_size,_,height,width = image.shape
        image = self.conv_layer(image)

        image = image.view(batch_size, self.features*32, -1)

        strokes = self.strokes_mpls(strokes)
        strokes_max = torch.max(strokes, 1)[0]
        strokes_avg = torch.mean(strokes, 1)
        strokes = torch.concat([strokes_max, strokes_avg], dim=1)

        x_max = torch.max(image, 2)[0]
        x_avg = torch.mean(image, 2)
        image = torch.concat([x_max, x_avg], dim=1)

        combined_input = torch.concat([image, strokes], dim=1)

        out = self.shared_mpls(combined_input)
        out = torch.sigmoid(out)

        distance_from_boarder = 0.25
        format_vector = out.new_tensor([
            width - width * distance_from_boarder,
            height - height * distance_from_boarder, 
            0.05, 
            0.05, 
        ])
        
        out = out * format_vector

        addition_boarder_vector = out.new_tensor([
            (width * distance_from_boarder) / 2, 
            (height * distance_from_boarder) / 2, 
            0.0, 
            0.0, 
        ])

        out = out + addition_boarder_vector

        return out


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.5) 


if __name__ == "__main__":

    random = torch.randn(1, 3, 64, 64)
    random_strokes = torch.randn(1, 10, 5)
    print(random.shape)
    model = Model()
    input_values = (random, random_strokes)
    out = model(input_values)
    print(out.shape)
    torch.save(model.state_dict(), "Model_Weights.pth")

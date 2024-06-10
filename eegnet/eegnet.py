import torch.nn as nn
import torch


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def __init__(
        self,
        F1=8,
        F2=16,
        D=2,
        K1=64,
        K2=16,
        n_timesteps=512,
        n_electrodes=96,
        n_classes=40,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, K1), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv2 = Conv2dWithConstraint(
            F1, F1 * D, (n_electrodes, 1), bias=False, groups=F1
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout()

        self.conv3 = nn.Conv2d(
            F1 * D, F1 * D, (1, K2), bias=False, groups=F1 * D, padding="same"
        )
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * (n_timesteps // 32), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act1(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.act2(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = EEGNet()
    x = torch.rand(2, 1, 96, 512)
    print(model(x).shape)

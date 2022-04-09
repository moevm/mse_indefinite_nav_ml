from torch import nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, activations, pool, kernel_size, padding=0, stride=1, ):
        super(Conv, self).__init__()
        layer = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding), activations, pool]
        self._model = nn.Sequential(*layer)

    def forward(self, x):
        return self._model(x)


class Linear(nn.Module):
    def __init__(self, in_size, out_size, activations):
        super(Linear, self).__init__()
        layer = [nn.Linear(in_features=in_size, out_features=out_size), activations]
        self._model = nn.Sequential(*layer)

    def forward(self, x):
        return self._model(x)

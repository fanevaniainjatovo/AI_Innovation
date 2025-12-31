from mindspore import nn

class AlcoholMLP(nn.Cell):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(input_dim, 128),
            nn.ReLU(),
            nn.Dense(128, 64),
            nn.ReLU(),
            nn.Dense(64, 1)
        )

    def construct(self, x):
        return self.net(x).view(-1)

"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        # x: (batch_size, 2)
        h1 = self.layer1(x).relu()
        h2 = self.layer2(h1).relu()
        return self.layer3(h2).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        # Веса размера (in_size, out_size)
        self.weights = minitorch.Parameter(
            minitorch.rand((in_size, out_size), requires_grad=True) * 2.0 - 1.0
        )
        # Смещение размера (out_size,)
        self.bias = minitorch.Parameter(
            minitorch.rand((out_size,), requires_grad=True) * 2.0 - 1.0
        )

    def forward(self, x):
        # x: (batch_size, in_size)
        # w: (in_size, out_size)
        # Бродкастим: (batch_size, in_size, 1) * (1, in_size, out_size)
        batch_size = x.shape[0]
        in_size = x.shape[1]
        out_size = self.weights.value.shape[1]

        x_expanded = x.view(batch_size, in_size, 1)
        w_expanded = self.weights.value.view(1, in_size, out_size)

        # Перемножаем и суммируем по размерности 1 (in_size) -> получаем (batch_size, 1, out_size)
        out = (x_expanded * w_expanded).sum(1).view(batch_size, out_size)
        return out + self.bias.value.view(1, out_size)


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)

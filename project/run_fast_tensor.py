
import math
import random
import time

import numba

import minitorch

datasets = minitorch.datasets
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", round(float(total_loss), 4), "correct", correct)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size, backend):
        super().__init__()
        scale = 1.0 / math.sqrt(in_size)
        self.weights = minitorch.Parameter(
            (minitorch.rand((in_size, out_size), backend=backend, requires_grad=True) * 2.0 - 1.0) * scale
        )
        self.bias = minitorch.Parameter(
            (minitorch.rand((out_size,), backend=backend, requires_grad=True) * 2.0 - 1.0) * 0.1
        )

    def forward(self, x):
        # x: (batch_size, in_size)
        # weights: (in_size, out_size)
        return (x @ self.weights.value) + self.bias.value.view(1, self.weights.value.shape[1])


class Network(minitorch.Module):
    def __init__(self, hidden_layers, backend):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers, backend)
        self.layer2 = Linear(hidden_layers, hidden_layers, backend)
        self.layer3 = Linear(hidden_layers, 1, backend)

    def forward(self, x):
        h1 = self.layer1(x).relu()
        h2 = self.layer2(h1).relu()
        return self.layer3(h2).sigmoid()


class FastTrain:
    def __init__(self, hidden_layers, backend=FastTensorBackend):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers, backend)
        self.backend = backend

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=self.backend))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X, backend=self.backend))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.model = Network(self.hidden_layers, self.backend)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        BATCH = 10
        losses = []

        start_time = time.time()

        for epoch in range(max_epochs):
            total_loss = 0.0
            c = list(zip(data.X, data.y))
            random.shuffle(c)
            X_shuf, y_shuf = zip(*c)

            for i in range(0, len(X_shuf), BATCH):
                optim.zero_grad()
                batch_len = len(X_shuf[i : i + BATCH])
                X = minitorch.tensor(X_shuf[i : i + BATCH], backend=self.backend)
                y = minitorch.tensor(y_shuf[i : i + BATCH], backend=self.backend)

                # Forward
                out = self.model.forward(X).view(batch_len)
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                # Численная защита вероятности от строгого 0
                prob = prob * 0.9999 + 0.00005
                loss = -prob.log()

                # Умножение на скаляр вместо деления тензора
                (loss * (1.0 / batch_len)).sum().view(1).backward()

                total_loss += loss.sum().view(1)[0]

                # Update
                optim.step()

            losses.append(total_loss)

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs - 1:
                X = minitorch.tensor(data.X, backend=self.backend)
                y = minitorch.tensor(data.y, backend=self.backend)
                out = self.model.forward(X).view(len(data.y))
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s (avg {total_time / max_epochs:.4f}s per epoch)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--PTS", type=int, default=50, help="number of points")
    parser.add_argument("--HIDDEN", type=int, default=10, help="number of hiddens")
    parser.add_argument("--RATE", type=float, default=0.05, help="learning rate")
    parser.add_argument("--BACKEND", default="cpu", help="backend mode")
    parser.add_argument("--DATASET", default="simple", help="dataset")
    parser.add_argument("--PLOT", default=False, help="dataset")

    args = parser.parse_args()

    PTS = args.PTS

    if args.DATASET == "xor":
        data = minitorch.datasets["Xor"](PTS)
    elif args.DATASET == "simple":
        data = minitorch.datasets["Simple"].simple(PTS)
    elif args.DATASET == "split":
        data = minitorch.datasets["Split"](PTS)

    HIDDEN = int(args.HIDDEN)
    RATE = args.RATE

    backend = FastTensorBackend if args.BACKEND != "gpu" else GPUBackend

    FastTrain(HIDDEN, backend=backend).train(data, RATE)
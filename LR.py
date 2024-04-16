import torch
import matplotlib.pyplot as plt

W = 0.5
B = 0.2
epochs = 1350

RAND_SEED = 69

torch.manual_seed(RAND_SEED)
inputs_tensor = torch.arange(0, 40, 1).unsqueeze(1)
output_tensor = W * inputs_tensor + B
losses = []


class LRC(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.bias = torch.nn.Parameter(torch.rand(1, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weight * x + self.bias)


model_0 = LRC()
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.001e-1,
                            momentum=0.9)

for epoch in range(epochs):

    model_0.train()
    outputs = model_0(inputs_tensor)

    loss = loss_fn(outputs, output_tensor)

    losses.append(loss.item())

    print(f"Weight: {model_0.weight.data} Bias: {model_0.bias.data}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


fig, ax = plt.subplots()
ax.plot(torch.arange(len(losses)), losses)
plt.show()

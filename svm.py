import torch
import numpy as np
import matplotlib.pyplot as plt

RAND_SEED = 69
torch.manual_seed(RAND_SEED)

np.random.seed(0)
X1_large = np.random.uniform(low=0, high=6, size=100)
X2_large = np.random.uniform(low=0, high=6, size=100)
Y_large = np.where(X1_large + X2_large > 6, 1, -1)

X1_large = X1_large.tolist()
X2_large = X2_large.tolist()
Y_large = Y_large.tolist()


X_tensor = torch.tensor(list(zip(X1_large, X2_large)))
Y_tensor = torch.tensor(Y_large)


class SVM(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(2))
        self.b = torch.nn.Parameter(torch.rand(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w) - self.b


def hinge_loss(outputs, targets):
    margin = 1 - outputs * targets
    return torch.mean(torch.clamp(margin, min=0))


model = SVM()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

num_epochs = 10000

for epoch in range(num_epochs):
    outputs = model(X_tensor)

    loss = hinge_loss(outputs, Y_tensor)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


plt.figure(figsize=(8, 6))
plt.scatter(X1_large, X2_large, c=Y_large,
            cmap=plt.cm.Paired, s=50, edgecolors='k', label='Data')

w = model.w.detach().numpy()
b = model.b.item()

x_min, x_max = min(X1_large) - 1, max(X1_large) + 1
y_min, y_max = min(X2_large) - 1, max(X2_large) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) - b
Z = np.sign(Z).reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k',
            levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Decision Boundary')
plt.legend()
plt.show()

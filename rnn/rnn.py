import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28
hidden_size = 128
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 1e-3

num_layers = 2
sequence_length = 28


train_dataset = datasets.MNIST(root='../dataset/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_dataset = datasets.MNIST(root='../dataset/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)


class RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layer, num_classes):
        super(RNN, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size,
                                hidden_size,
                                num_layer,
                                batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layer,
                         x.size(0),
                         self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    print(f"EPOCH {epoch+1} of {num_epochs}")
    for i, (images, labels) in enumerate(train_loader):

        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 100 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss {loss.item():.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss {loss.item():.4f}')


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)

        a, predicted = torch.max(outputs.data, 1)
        print(a)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy {acc} %')

torch.save(model, './model')

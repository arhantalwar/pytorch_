import torch
from torchvision import transforms
from PIL import Image

image_path = "./cat.jpg"
image = Image.open(image_path)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

input_image = transform(image).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_image = input_image.to(device)

model = torch.load('./model2')
model.eval()

with torch.no_grad():
    output = model(input_image)

_, predicted_class = torch.max(output, 1)

print("Predicted Class:", predicted_class.item())

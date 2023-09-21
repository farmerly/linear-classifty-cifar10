import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision.transforms import ToTensor


class SoftmaxNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_stack(x)
        return x


def get_mnist_dataloader():
    train_set = torchvision.datasets.FashionMNIST(
        root="./datasets", train=True, download=True, transform=ToTensor())
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=100, shuffle=True, num_workers=0, drop_last=False)
    test_set = torchvision.datasets.FashionMNIST(
        root="./datasets", train=False, download=True, transform=ToTensor())
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=100, shuffle=True, num_workers=0, drop_last=False)
    return train_loader, test_loader


def select_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return torch.device(device)


if __name__ == "__main__":
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    device = select_device()
    train_loader, test_loader = get_mnist_dataloader()
    linear_softmax = SoftmaxNetwork().to(device)
    linear_softmax.load_state_dict(torch.load('weights/model.pth'))
    loss_fn = nn.CrossEntropyLoss()

    size = len(test_loader.dataset)
    num_batchs = len(test_loader)
    linear_softmax.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            pred = linear_softmax(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss = test_loss / num_batchs
    correct = correct / size
    print(f'corrent: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}\n')

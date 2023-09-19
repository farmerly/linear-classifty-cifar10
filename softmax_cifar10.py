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
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_stack(x)
        return x


def get_cifar10_dataloader():
    train_set = torchvision.datasets.CIFAR10(
        root="./datasets", train=True, download=True, transform=ToTensor())
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=100, shuffle=True, num_workers=0, drop_last=False)
    test_set = torchvision.datasets.CIFAR10(
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


def train(dataloader, model, device, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算损失
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch + 1) % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


def test(dataloader, model, device, loss_fn):
    size = len(dataloader.dataset)
    num_batchs = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss = test_loss / num_batchs
    correct = correct / size
    print(f'corrent: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}\n')


if __name__ == "__main__":
    epoch = 5
    device = select_device()
    train_loader, test_loader = get_cifar10_dataloader()
    linear_svm = SoftmaxNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(linear_svm.parameters(), lr=0.01)

    for i in range(epoch):
        print(f'Epoch: {i+1}\n---------------------------------')
        train(train_loader, linear_svm, device, loss_fn, optimizer)
        test(train_loader, linear_svm, device, loss_fn)
    print('Done')

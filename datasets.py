import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


if __name__ == '__main__':
    train_sets = torchvision.datasets.FashionMNIST(
        root="datasets", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(dataset=train_sets, batch_size=16)

    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    img = train_features[0].squeeze()
    label = train_labels[0]

    plt.imshow(img, cmap="gray")
    plt.show()

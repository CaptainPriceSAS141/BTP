from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

CINIC_MEAN = (0.4789, 0.4723, 0.4305)
CINIC_STD  = (0.2421, 0.2383, 0.2587)

def get_transforms(dataset_name):

    if dataset_name.lower() == "cifar10":
        mean = CIFAR_MEAN
        std = CIFAR_STD
    else:
        mean = CINIC_MEAN
        std = CINIC_STD

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, test_transform


def get_dataloaders(dataset_name, batch_size=32):

    train_transform, test_transform = get_transforms(dataset_name)

    if dataset_name.lower() == "cifar10":

        train_dataset = datasets.CIFAR10(
            root="./data/cifar10",
            train=True,
            download=True,
            transform=train_transform
        )

        test_dataset = datasets.CIFAR10(
            root="./data/cifar10",
            train=False,
            download=True,
            transform=test_transform
        )

    elif dataset_name.lower() == "cinic10":

        train_dataset = datasets.ImageFolder(
            "./data/cinic10/train",
            transform=train_transform
        )

        test_dataset = datasets.ImageFolder(
            "./data/cinic10/test",
            transform=test_transform
        )

    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, test_loader
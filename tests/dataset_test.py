from pathlib import Path

from torchvision import transforms

from dataset import ImagesSeq2KarteDataset
from my_transforms import Normalize, ToFloatTensorInZeroOne


if __name__ == "__main__":
    dataset_name = 'dataset2'
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / dataset_name
    dataset_file = str(dataset_path / 'image_label.joblib')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    transform_3d = transforms.Compose([
        ToFloatTensorInZeroOne(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImagesSeq2KarteDataset(dataset_file=dataset_file, transform=transform, transform_3d=transform_3d)
    print(len(dataset))
    print(dataset.max_image_num)
    print(dataset[0][0].size())
    dataset = ImagesSeq2KarteDataset(dataset_file=dataset_file, transform=transform, transform_3d=transform_3d,
                                     image_num_limit=60)
    print(len(dataset))
    print(dataset.max_image_num)
    print(dataset[0][0])
    print(dataset[0][0].size())

import joblib
from pathlib import Path

from torchvision import transforms
from tqdm import tqdm


if __name__ == "__main__":
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'image-labels-ver4'
    dataset_file = str(dataset_path / 'data.joblib')
    with open(dataset_file, mode="rb") as f:
        data_dict = joblib.load(f)
    import matplotlib.pyplot as plt
    import accimage

    transform0 = transforms.Compose([
        transforms.ToTensor()
    ])
    transform1 = transforms.Compose([
        transforms.Resize((480, 560)),
        transforms.CenterCrop((448, 512)),
        transforms.ToTensor()
    ])
    transform2 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    transform3 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image_path_list = data_dict["image_path"]
    labels_list = data_dict["labels"]
    count = 0
    width = 0
    height = 0
    size_set = set()
    for image_path in tqdm(image_path_list):
        image = accimage.Image(image_path)
        if count == 0 or (image.width != width or image.height != height):
            width = image.width
            height = image.height
            size_set.add(','.join([str(width), str(height)]))
            plt.imshow(transform0(image).permute(1, 2, 0))
            plt.show()
            plt.imshow(transform1(image).permute(1, 2, 0))
            plt.show()
            plt.imshow(transform2(image).permute(1, 2, 0))
            plt.show()
            plt.imshow(transform3(image).permute(1, 2, 0))
            plt.show()
        else:
            width = image.width
            height = image.height
        if count > 10000:
            break
        count += 1
    print(size_set)


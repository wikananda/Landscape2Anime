import os
import argparse
import shutil
from sklearn.model_selection import train_test_split


def split_dataset(dataset_a_dir, dataset_b_dir, output_dir, test_size=0.2):
    for dataset, label in [(dataset_a_dir, 'A'), (dataset_b_dir, 'B')]:
        images = [f for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))]

        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)

        train_dir = os.path.join(output_dir, f'train{label}')
        test_dir = os.path.join(output_dir, f'test{label}')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for image in train_images:
            shutil.copy(os.path.join(dataset, image), os.path.join(train_dir, image))

        for image in test_images:
            shutil.copy(os.path.join(dataset, image), os.path.join(test_dir, image))

        print(f'{label}: {len(train_images)} train images, {len(test_images)} test images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split dataset into training and testing sets for image-to-image translation.')
    parser.add_argument('dataset_a_dir', type=str, help='Directory containing dataset A images')
    parser.add_argument('dataset_b_dir', type=str, help='Directory containing dataset B images')
    parser.add_argument('output_dir', type=str, help='Directory to save the split dataset')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split')

    args = parser.parse_args()

    split_dataset(args.dataset_a_dir, args.dataset_b_dir, args.output_dir, args.test_size)

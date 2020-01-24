from shutil import copy2
from pathlib import Path
from random import shuffle


def partition_dataset(dataset_path, train_split=0.7, validation_split=0.15, test_split=0.15, overwrite_possible=False):
    print(f'Partitioning files from {dataset_path} ... ', end='')
    if train_split + validation_split + test_split != 1:
        print('Also rescaling split fractions to sum to 1 ...', end='')
        train_split, validation_split, test_split = (split/(train_split + validation_split + test_split) for split in
                                                     [train_split, validation_split, test_split])

    dataset_path = Path(dataset_path)
    dataset_path_all = dataset_path / "All"

    assert '/All' not in dataset_path.as_posix(), "Partition will not work when data is already in an 'All' folder"
    # Create new folders
    new_folders = ["All", "Test", "Train", "Val"] if validation_split else ["All", "Test", "Train"]
    [(dataset_path / new_dir).mkdir(exist_ok=overwrite_possible) for new_dir in new_folders]

    # Move original files to All folder
    entries_to_move = list([dp for dp in dataset_path.glob('*') if dp.stem not in new_folders])
    if not list(entries_to_move):
        return
    [dp.rename(dataset_path_all / dp.stem) for dp in entries_to_move]

    # Get all files, then shuffle while keeping stratified
    partition_paths = [[], [], []]
    category_folders = list(dataset_path_all.iterdir())
    for category_folder in category_folders:
        category_files = list(category_folder.rglob("*.*"))
        shuffle(category_files)
        split_idx = [int(train_split * len(category_files)), int((train_split + validation_split) * len(category_files))]
        partition_paths[0] += category_files[split_idx[1]:]
        partition_paths[1] += category_files[:split_idx[0]]
        partition_paths[2] += category_files[split_idx[0]:split_idx[1]]

    for paths, name in zip(partition_paths, new_folders[1:]):
        for file_path in paths:
            dst = Path(file_path.as_posix().replace('/All/', f'/{name}/'))
            dst.parent.mkdir(exist_ok=True, parents=True)
            copy2(str(file_path), str(dst))
    print('Done')


if __name__ == "__main__":
    partition_dataset(r"D:\SplitTest", train_split=0.5, validation_split=0.5, test_split=.5)
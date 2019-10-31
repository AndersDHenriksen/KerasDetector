from shutil import copy2
from pathlib import Path
from random import shuffle


def partition_dataset(dataset_path, train_split=0.7, validation_split=0.15, test_split=0.15, overwrite_possible=False):
    assert train_split + validation_split + test_split == 1, "Train + validation + test must equal 1"

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
    print(f'Partitioning files from {dataset_path} ... ', end='')
    [dp.rename(dataset_path_all / dp.stem) for dp in entries_to_move]

    # Get all files then shuffle
    all_files = list(dataset_path_all.rglob("*.*"))
    shuffle(all_files)

    # Copy to Train, Val, Test folders
    split_idx = [int(train_split * len(all_files)), int((train_split + validation_split) * len(all_files))]
    partition_paths = [all_files[split_idx[1]:], all_files[:split_idx[0]], all_files[split_idx[0]:split_idx[1]]]
    for paths, name in zip(partition_paths, new_folders[1:]):
        for file_path in paths:
            dst = Path(file_path.as_posix().replace('/All/', f'/{name}/'))
            dst.mkdir(exist_ok=True, parents=True)
            copy2(str(file_path), str(dst))
    print('Done')

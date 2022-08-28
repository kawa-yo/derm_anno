
import os


def make_parent_dir(file: str) -> None:
    dirname = os.path.dirname(file)
    if dirname.strip() != '':
        os.makedirs(dirname, exist_ok=True)

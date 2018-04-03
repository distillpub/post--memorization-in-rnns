import os
import os.path as path

from python.download.util.tqdm_download import download

dirname = path.dirname(path.realpath(__file__))
content_dir = path.join(dirname, '..', 'content')


class ContentDir:
    dirname: str
    dirpath: str

    def __init__(self) -> None:
        self.dirpath = content_dir

    def __enter__(self):
        try:
            os.mkdir(self.dirpath, mode=0o755)
        except FileExistsError:
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def download(self, name: str, url: str) -> None:
        if not self.exists(name):
            print(f'downloading: {name} from {url}')
            download(url, self.filepath(name), desc=name)

    def exists(self, name: str) -> bool:
        return path.exists(self.filepath(name))

    def filepath(self, name: str) -> str:
        return path.join(self.dirpath, name)

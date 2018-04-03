
import urllib
from tqdm import tqdm


def download(*args, desc=None, **kwargs):
    last_b = [0]

    def _download_hook(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    with tqdm(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(*args,
                                   reporthook=_download_hook, data=None,
                                   **kwargs)

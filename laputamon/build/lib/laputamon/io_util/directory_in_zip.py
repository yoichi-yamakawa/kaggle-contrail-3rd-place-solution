# this part from Kaizaburo Chubachi's code

import os
from typing import Any, List, Optional

import pfio


def _get_abs_path(path: str) -> str:
    if path.startswith("hdfs://"):
        return path
    # Resolve symlinks if Posix
    return os.path.abspath(path)


# class DirectoryInZip:
#     def __init__(self, path: str) -> None:
#         parts = path.split(os.sep)
#         num_dot_zip = sum([name.endswith(".zip") for name in parts])

#         if num_dot_zip >= 2:
#             # TODO: message
#             raise ValueError

#         self.str = path
#         self.zip_container: Optional[pfio.containers.zip.ZipContainer]
#         self.root: str

#         if num_dot_zip == 0:
#             self.zip_container = None
#             self.root = _get_abs_path(path)
#         else:
#             split_point = path.find(".zip") + 4
#             self.zip_container = pfio.open_as_container(_get_abs_path(path[:split_point]))
#             self.root = path[split_point:].lstrip(os.sep)

#     def open(self, file_path: str, *args: Any, **kwargs: Any) -> Any:
#         path = os.path.join(self.root, file_path)

#         if self.zip_container is not None:
#             return self.zip_container.open(path, *args, **kwargs)
#         else:
#             return pfio.open(path, *args, **kwargs)

#     def listdir(self, path: Optional[str] = None) -> List[str]:
#         path = self.root if path is None else os.path.join(self.root, path)

#         if self.zip_container is not None:
#             return self.zip_container.list(path)
#         else:
#             return pfio.list(path)

#     def __str__(self) -> str:
#         return self.str


import os
from typing import Any, List, Optional

import pfio
from pfio.v2 import Zip, ZipFileStat, from_url, local

# from pfio.v2 import local as pfio


def _get_abs_path(path: str) -> str:
    if path.startswith("hdfs://"):
        return path
    # Resolve symlinks if Posix
    return os.path.abspath(path)


class DirectoryInZip:
    def __init__(self, path: str) -> None:
        parts = path.split(os.sep)
        num_dot_zip = sum([name.endswith(".zip") for name in parts])

        if num_dot_zip >= 2:
            # TODO: message
            raise ValueError

        self.str = path
        self.zip_container: Optional[pfio.containers.zip.ZipContainer]
        self.root: str

        if num_dot_zip == 0:
            self.root = _get_abs_path(path)
            self.is_zip = False
            self.fs = local
        else:
            split_point = path.find(".zip") + 4
            self.is_zip = True
            self.zip_file_path = _get_abs_path(path[:split_point])
            self.fs = Zip(local, self.zip_file_path)
            self.root = path[split_point:].lstrip(os.sep)

    def open(self, file_path: str, *args: Any, **kwargs: Any) -> Any:
        path = os.path.join(self.root, file_path)

        if self.is_zip:
            return self.fs.open(path, mode="rb")
        else:
            return self.fs.open(path, *args, **kwargs)

    def listdir(self, path: Optional[str] = None) -> List[str]:
        path = self.root if path is None else os.path.join(self.root, path)

        return self.fs.list(path)

    def __str__(self) -> str:
        return self.str

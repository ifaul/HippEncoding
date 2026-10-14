from copy import deepcopy
from os import makedirs, path
from typing import Any, Iterable, Optional


class Path(object):
    # bids: root / sub / <ses> / <dtype> / <entities>_suffix.ext
    # inspired by
    # https://bids-specification.readthedocs.io/en/stable/02-common-principles.html#filenames

    def __init__(
        self,
        root: Optional[str] = None,
        datatype: Optional[str] = None,
        suffix: Optional[str] = None,
        ext: Optional[str] = None,
        subdirkeys: Iterable[str] = ("sub", "conv", "ses", "datatype"),
        **kwargs,
    ) -> None:
        self.ext = ext
        self.root = root
        self.suffix = suffix
        self.datatype = datatype
        self.subdirkeys = subdirkeys
        self.metakeys = {"root", "datatype", "suffix", "ext"}
        self.entities = {k: v for k, v in kwargs.items() if v is not None}

    @property
    def basename(self) -> str:
        filename = self.stitch_(**self.entities)
        if (suffix := self.suffix) is not None:
            filename += "_" + suffix
        if (ext := self.ext) is not None:
            if not ext.startswith("."):
                ext = "." + ext
            filename += ext
        return filename

    @property
    def dirname(self) -> str:
        dirnames = []
        if self.root is not None:
            dirnames.append(self.root)
        for subdirkey in self.subdirkeys:
            if subdirkey in self.entities:
                dirnames.append(f"{subdirkey}-{self[subdirkey]}")
            elif hasattr(self, subdirkey):
                if (value := getattr(self, subdirkey)) is not None:
                    dirnames.append(value)
        return path.join(*dirnames)

    def __getitem__(self, item: str) -> Any:
        if item in self.entities:
            return self.entities[item]
        elif isinstance(item, str) and hasattr(self, item):
            return getattr(self, item)
        else:
            raise ValueError("item does not exist", item)

    def __delitem__(self, __name: str) -> None:
        if __name in self.entities:
            del self.entities[__name]
        elif hasattr(self, __name):
            setattr(self, __name, None)

    @property
    def fpath(self) -> str:
        return path.join(self.dirname, self.basename)

    def starstr(self, subdirkeys: Iterable[str]) -> str:
        dirnames = []

        if self.root is not None:
            dirnames.append(self.root)

        for subdirkey in subdirkeys:
            if subdirkey in self.entities:
                dirnames.append(f"{subdirkey}-{self[subdirkey]}")
            elif hasattr(self, subdirkey):
                if (value := getattr(self, subdirkey)) is not None:
                    dirnames.append(value)
            else:
                dirnames.append(f"{subdirkey}-*")

        filename = self.stitch_("*", **self.entities)
        filename += "*"
        if (suffix := self.suffix) is not None:
            filename += "_" + suffix
        if self.ext is not None:
            filename += self.ext
        dirnames.append(filename)

        return path.join(*dirnames)

    def update(
        self,
        **kwargs,
    ):
        for key, value in kwargs.items():
            if key in self.metakeys:
                setattr(self, key, kwargs[key])
        for key in self.metakeys:
            if key in kwargs:
                del kwargs[key]
        self.entities.update({k: v for k, v in kwargs.items() if v is not None})
        return self

    def mkdirs(self, exist_ok: bool = True) -> None:
        makedirs(self.dirname, exist_ok=exist_ok)

    def isfile(self) -> bool:
        return path.isfile(self)

    def __repr__(self) -> str:
        return self.fpath

    def __str__(self) -> str:
        return self.basename

    def __fspath__(self) -> str:
        return self.fpath

    def copy(self):
        return deepcopy(self)

    @staticmethod
    def frompath(filename: str, castint=True):
        dirname = path.dirname(filename)
        basename = path.basename(filename)
        string, ext = path.splitext(basename)
        parts = {}
        suffix = None
        for part in string.split("_"):
            kv = part.split("-", 1)
            value = None
            if len(kv) > 1:
                value = kv[1]
                if castint and value.isnumeric():
                    value = int(value)
            else:
                suffix = kv[0]
            parts[kv[0]] = value
        return Path(**parts, root=dirname, suffix=suffix, ext=ext)

    @staticmethod
    def stitch_(_delim="_", **kwargs: dict) -> str:
        string = _delim.join(
            f"{key}-{value}" if value else key for key, value in kwargs.items()
        )
        return string
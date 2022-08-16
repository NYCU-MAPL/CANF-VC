import logging
import os
import shutil
import time

isfile = os.path.isfile
isdir = os.path.isdir


def abspath(file):
    return os.path.dirname(os.path.abspath(file))+"/"


def parentpath(path):
    return "/".join(path.split("/")[:-2])+"/"


def ls(path="./", param=""):
    files = []
    for tmp in os.listdir(path):
        item = tmp
        if "a" not in param and tmp[0] == ".":
            continue
        if os.path.isdir(path+tmp):
            item += "/"
        files.append(item)
    return sorted(files)


def mkdir(path):
    path += "/" if path[-1] != "/" else ""
    os.makedirs(path, exist_ok=True)
    return path


def cleandir(root, rmroot=False):
    for f in ls(root, '-a'):
        if isdir(root+f):
            cleandir(root+f, rmroot=True)
        else:
            os.remove(root+f)
    if rmroot and os.path.isdir(root):
        os.removedirs(root)


class ffilter:
    def __init__(self, keep, ignore=None):
        """Factory function that can be used with copytree() ignore parameter.

        Arguments define a sequence of glob-style patterns
        that are used to specify what files to NOT ignore.
        Creates and returns a function that determines this for each directory
        in the file hierarchy rooted at the source directory when used with
        lib.cptree().
        """
        if ignore is None:
            ignore = []
        ignore += [".DS_Store", "__pycache__/"]

        self.keep = set(keep)
        self.ignore = set(ignore) - self.keep

        # print(self.keep, self.ignore)

    def __call__(self, lnames):
        names = set(lnames)
        import fnmatch
        keep = set(name for pattern in self.keep for name in fnmatch.filter(
            names, pattern))

        dirs = set(name for name in names -
                   keep if name[-1] == "/" and name not in self.ignore)
        ignore = set(name for pattern in self.ignore for name in fnmatch.filter(
            names, pattern))

        return keep - ignore | dirs - ignore


def cptree(src, dst, ffilter=None, param="-a", layer=0):
    names = ls(src, param)
    if ffilter is not None:
        names = ffilter(names)
    for f in names:
        if "v" in param:
            print(" "*layer*4 + "-"*3 + f)
        if os.path.isdir(src+f):
            subdst = mkdir(dst+f)
            cptree(src+f, subdst, ffilter, param, layer+1)
        else:
            shutil.copyfile(src+f, dst+f)

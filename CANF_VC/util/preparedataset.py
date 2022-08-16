import os
import numpy as np
import fnmatch


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(root)
    return result

Ds = os.getenv('DATASET')
folder = find('1.png', Ds+'vimeo_septuplet/sequences/')
np.save('folder.npy', folder)
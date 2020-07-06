import os


def mkdir(ph):
    if not os.path.exists(ph):
        os.makedirs(ph)

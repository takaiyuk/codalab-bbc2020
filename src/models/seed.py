import os
import random

import numpy as np
import tensorflow as tf


def fix_seeds():
    os.environ["PYTHONHASHSEED"] = "42"
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

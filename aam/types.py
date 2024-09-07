from typing import NewType

import tensorflow as tf

IntTensor = NewType("IntTensor", tf.Tensor)
FloatTensor = NewType("FloatTensor", tf.Tensor)

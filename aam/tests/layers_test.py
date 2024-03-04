"""NucleotideEinsum
>>> embeddings = tf.reshape(tf.range(0,2*6, dtype=tf.float32),(1,2,3,2))
>>> embedding
<tf.Tensor: shape=(1, 2, 3, 2), dtype=float32, numpy=
array([[[[ 0.,  1.],
         [ 2.,  3.],
         [ 4.,  5.]],

        [[ 6.,  7.],
         [ 8.,  9.],
         [10., 11.]]]], dtype=float32)>

>>> einsum_dense = NucleotideEinsum(dff=8, kernel_initializer="ones")
>>> einsum_dense(embeddings)
<tf.Tensor: shape=(1, 2, 2), dtype=float32, numpy=
array([[[ 48.,  72.],
        [192., 216.]]], dtype=float32)>
"""

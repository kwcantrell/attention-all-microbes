import unittest
import numpy as np
from biom.table import Table
from aam.data_utils import get_sequencing_dataset


class TestSequencingData(unittest.TestCase):

    def test_get_sequencing_data(self):
        rng = np.random.default_rng(12345)
        rints = rng.integers(low=0, high=4, size=3)
        nuc_chars = 'ACGT'
        data = np.arange(100).reshape(20, 5)
        s_ids = ['S%d' % i for i in range(5)]
        o_ids = [''.join([nuc_chars[i] for i in rng.integers(low=0, high=4,
                 size=5)]) for _ in range(20)]
        table = Table(data, o_ids, s_ids)
        print(table)
        print(rints)
        dataset = get_sequencing_dataset(table)
        for item in dataset.take(1):
            print(item)

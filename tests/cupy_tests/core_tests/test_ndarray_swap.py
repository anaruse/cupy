import unittest

import cupy
from cupy import cuda
from cupy import testing


@testing.gpu
class TestArraySwap(unittest.TestCase):

    _multiprocess_can_split_ = True
    shape = (2, 3, 4)

    def setUp(self):
        self.stream = cuda.Stream.null

    @testing.for_all_dtypes()
    def test_swapout(self, dtype):
        a = core.ndarray(self.shape)
        a.fill(1)
        a.swapout()
        testing.assertTrue(a.is_swapout)
        testing.assertEqual(a.data, None)


    @testing.for_all_dtypes()
    def test_swapin(self, dtype):
        a = core.ndarray(self.shape)
        a.fill(1)
        a.swapout()
        a.swapin()
        testing.assertFalse(a.is_swapout)
        testing.assertEqual(a.data_swapout, None)


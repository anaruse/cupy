import unittest

import cupy
from cupy import cuda
from cupy import testing

import copy


@testing.gpu
class TestArraySwap(unittest.TestCase):

    shape = (2, 3, 4)

    def setUp(self):
        self.stream = cuda.Stream()

    @testing.for_all_dtypes()
    def test_swapout(self, dtype):
        a = cupy.random.uniform(0, 2, self.shape).astype(dtype)
        a.swapout()
        self.assertTrue(a.is_swapout)
        self.assertEqual(a.data, None)
        self.assertNotEqual(a.data_swapout, None)

    @testing.for_all_dtypes()
    def test_swapout_async(self, dtype):
        a = cupy.random.uniform(0, 2, self.shape).astype(dtype)
        a.swapout(stream=self.stream)
        self.stream.synchronize()
        self.assertTrue(a.is_swapout)
        self.assertEqual(a.data, None)
        self.assertNotEqual(a.data_swapout, None)

    @testing.for_all_dtypes()
    def test_swapin(self, dtype):
        a = cupy.random.uniform(0, 2, self.shape).astype(dtype)
        b = copy.deepcopy(a)
        a.swapout()
        a.swapin()
        self.assertFalse(a.is_swapout)
        self.assertNotEqual(a.data, None)
        self.assertEqual(a.data_swapout, None)
        testing.assert_array_equal(a, b)

    @testing.for_all_dtypes()
    def test_swapin_async(self, dtype):
        a = cupy.random.uniform(0, 2, self.shape).astype(dtype)
        b = copy.deepcopy(a)
        a.swapout(stream=self.stream)
        a.swapin(stream=self.stream)
        self.stream.synchronize()
        self.assertFalse(a.is_swapout)
        self.assertNotEqual(a.data, None)
        self.assertEqual(a.data_swapout, None)
        testing.assert_array_equal(a, b)

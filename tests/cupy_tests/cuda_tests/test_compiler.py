import os
import unittest

import six
from cupy import testing
from cupy.cuda import compiler

from pynvrtc.compiler import ProgramException


@testing.gpu
class TestNvrtcErr(unittest.TestCase):

    def test(self):
        with self.assertRaises(ProgramException):
            compiler.nvrtc('a')

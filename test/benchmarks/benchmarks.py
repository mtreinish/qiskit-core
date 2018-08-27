# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import unittest


class TestSuite(unittest.TestCase):
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setUp(self):
        super(TimeSuite, self).setUp()
        self.d = {}
        for x in range(500):
            self.d[x] = None

    def test_keys(self):
        for key in self.d.keys():
            pass

    def test_iterkeys(self):
        for key in self.d.items():
            pass

    def test_range(self):
        d = self.d
        for key in range(500):
            x = d[key]

#    def test_xrange(self):
#        d = self.d
#        for key in range(500):
#            x = d[key]


class MemSuite:
    def mem_list(self):
        return [0] * 256

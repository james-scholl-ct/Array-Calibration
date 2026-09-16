import unittest
from parameterized import parameterized
from lcm_voltage_patterns.delta.load_patterns import load_standard_voltage_patterns as lp


class TestLCMLoadPatterns(unittest.TestCase):

    @parameterized.expand(
        (
                ((91, 4), ValueError()),
                ((-6, 10), ValueError()),
                ((0, 7), 9.0)
        )
    )
    def test_out_of_bounds_inputs(self, inp, exp):
        if isinstance(exp, Exception):
            with self.assertRaises(type(exp)):
                lp(*inp)
        else:
            self.assertEqual(exp, lp(*inp)[0][0])


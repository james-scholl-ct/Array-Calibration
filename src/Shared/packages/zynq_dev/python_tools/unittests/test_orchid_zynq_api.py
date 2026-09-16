import os
from parameterized import parameterized
import random
import testconfig as tc
import unittest
import sys

from python_tools.zynq_api import OrchidZynqAPI


class OrchidZynqAPITestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(OrchidZynqAPITestClass, cls).setUpClass()
        cls.dut = OrchidZynqAPI(tc.config['host'])
        cls.N_TABLES = 4
        cls.N_CHANNELS = 64
        cls.ADD_A_ADDR = 64
        cls.ADD_B_ADDR = 65
        cls.ADD_SUM_ADDR = 66

    def setUp(self):
        super(OrchidZynqAPITestClass, self).setUp()
        self.dut.start_remote_app()

        # setup PRNG
        def_seed = random.randrange(sys.maxsize)
        self.random_seed = int(tc.config.get('random_seed', def_seed))
        random.seed(self.random_seed)
        seed_file = os.path.abspath(os.path.dirname(__file__))
        seed_file = os.path.join(seed_file, 'random_seeds.txt')
        with open(seed_file, 'a') as f:
            f.write("{}: {}\n".format(self.id(), self.random_seed))

    def tearDown(self):
        super(OrchidZynqAPITestClass, self).tearDown()
        self.dut.stop_remote_app()

    @classmethod
    def tearDownClass(cls):
        super(OrchidZynqAPITestClass, cls).tearDownClass()
        cls.dut.close_connection()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def get_rand_bytes(self, n_bytes):
        return [random.randint(0, 0xff) for i in range(n_bytes)]

    def get_rand_words(self, n_words):
        return [random.randint(0, 0xffff_ffff) for i in range(n_words)]

    # -------------------------------------------------------------------------
    # Unit Tests
    # -------------------------------------------------------------------------
    @parameterized.expand([
        ("", 1, 2, 3),
        ("", 0xffff_ffff, 0x1, 0x0),
    ])
    def test_adder(self, name, a, b, gold):
        self.dut._send_write(self.ADD_A_ADDR, a)
        self.dut._send_write(self.ADD_B_ADDR, b)
        rdata = self.dut._send_read(self.ADD_SUM_ADDR)
        self.assertEqual(rdata, gold)


    @parameterized.expand([
        ("", 0),
        ("", 1),
        ("", 2),
        ("", 3),
    ])
    def test_set_table(self, name, table_idx):
        coeffs = self.get_rand_bytes(self.N_CHANNELS)
        self.dut.set_table(table_idx, coeffs)
        rdata = self.dut.get_table(table_idx)
        self.assertEqual(coeffs, rdata)

    @parameterized.expand([
        ("", 0, [0, 1, 63]),
        ("", 1, [0, 7, 63]),
        ("", 2, [0, 12, 63]),
        ("", 3, [0, 22, 63]),
    ])
    def test_set_channel(self, name, table_idx, channel_idxs):
        # set memory to deterministic values
        coeffs = [self.get_rand_bytes(self.N_CHANNELS) for i in range(self.N_TABLES)]
        for i in range(self.N_TABLES):
            self.dut.set_table(i, coeffs[i])
            rdata = self.dut.get_table(i)
            self.assertEqual(coeffs[i], rdata)

        for channel_idx in channel_idxs:
            wdata = self.get_rand_bytes(1)[0]
            self.dut.set_channel(table_idx, channel_idx, wdata)
            rdata = self.dut.get_channel(table_idx, channel_idx)
            self.assertEqual(wdata, rdata)
            coeffs[table_idx][channel_idx] = wdata

        coeffs_read = [self.dut.get_table(i) for i in range(self.N_TABLES)]
        self.assertEqual(coeffs, coeffs_read)

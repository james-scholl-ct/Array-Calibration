import os
from parameterized import parameterized
import random
import testconfig as tc
import unittest
import sys

from python_tools.zynq_api import LotusZynqAPI


class LotusZynqAPITestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(LotusZynqAPITestClass, cls).setUpClass()
        cls.dut = LotusZynqAPI(tc.config['host'])
        cls.N_CHANNELS = 204

    def setUp(self):
        super(LotusZynqAPITestClass, self).setUp()
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
        super(LotusZynqAPITestClass, self).tearDown()
        self.dut.stop_remote_app()

    @classmethod
    def tearDownClass(cls):
        super(LotusZynqAPITestClass, cls).tearDownClass()
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
    def test_set_table(self):
        coeffs = self.get_rand_bytes(self.N_CHANNELS)
        self.dut.set_table(coeffs)
        rdata = self.dut.get_table()
        self.assertEqual(coeffs, rdata)

    def test_axi_partitions(self):
        n_words = 20
        spi_wdata = self.get_rand_words(n_words)
        bsc_wdata = self.get_rand_words(n_words)

        self.dut._send_write_range(0, spi_wdata, periph='spi')
        self.dut._send_write_range(0, bsc_wdata, periph='bsc')

        spi_rdata = self.dut._send_read_range(0, n_words, periph='spi')
        bsc_rdata = self.dut._send_read_range(0, n_words, periph='bsc')

        self.assertEqual(spi_rdata, spi_wdata)
        self.assertEqual(bsc_rdata, bsc_wdata)
        self.assertNotEqual(bsc_rdata, spi_rdata)

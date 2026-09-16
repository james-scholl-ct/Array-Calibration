import os
from parameterized import parameterized
import random
import sys
from testconfig import config
import unittest

from python_tools.bcuda_zynq_api import BcudaZynqAPI


class BcudaZynqAPITestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(BcudaZynqAPITestClass, cls).setUpClass()
        cls.dut = BcudaZynqAPI(config['host'])

    def setUp(self):
        super(BcudaZynqAPITestClass, self).setUp()
        self.dut.start_remote_app()

        # setup PRNG
        def_seed = random.randrange(sys.maxsize)
        self.random_seed = int(config.get('random_seed', def_seed))
        random.seed(self.random_seed)
        seed_file = os.path.abspath(os.path.dirname(__file__))
        seed_file = os.path.join(seed_file, 'random_seeds.txt')
        with open(seed_file, 'a') as f:
            f.write("{}: {}\n".format(self.id(), self.random_seed))

    def tearDown(self):
        super(BcudaZynqAPITestClass, self).tearDown()
        self.dut.stop_remote_app()

    @classmethod
    def tearDownClass(cls):
        super(BcudaZynqAPITestClass, cls).tearDownClass()
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
    def test_axi_partitions(self):
        n_words = 56
        ip1_wdata = self.get_rand_words(n_words)
        ip0_wdata = self.get_rand_words(n_words)

        self.dut._send_write_range(0, ip1_wdata, periph='ip1')
        self.dut._send_write_range(0, ip0_wdata, periph='ip0')

        ip1_rdata = self.dut._send_read_range(0, n_words, periph='ip1')
        ip0_rdata = self.dut._send_read_range(0, n_words, periph='ip0')

        self.assertEqual(ip1_rdata, ip1_wdata)
        self.assertEqual(ip0_rdata, ip0_wdata)
        self.assertNotEqual(ip0_rdata, ip1_rdata)

import os
from parameterized import parameterized
import random
import testconfig as tc
import unittest
import sys

from python_tools.zynq_api import ZynqAPI


class ZynqAPITestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(ZynqAPITestClass, cls).setUpClass()
        cls.N_RW_WORDS = 66
        cls.dut = ZynqAPI(tc.config['host'],
                          'orchid', # no generic app, for now
                          {'bsc': cls.N_RW_WORDS})

    def setUp(self):
        super(ZynqAPITestClass, self).setUp()
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
        super(ZynqAPITestClass, self).tearDown()
        self.dut.stop_remote_app()

    @classmethod
    def tearDownClass(cls):
        super(ZynqAPITestClass, cls).tearDownClass()
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
    @unittest.skip("This test is disabled due to length (~20 seconds).")
    def test_read_write(self):
        for addr in range(self.N_RW_WORDS):
            self.dut._send_write(addr, addr**2)
        for addr in range(self.N_RW_WORDS):
            rdata = self.dut._send_read(addr)
            self.assertEqual(rdata, addr**2)

    def test_read_write_queued(self):
        for addr in range(self.N_RW_WORDS):
            self.dut._send_write(addr, (addr+1)**2, queue=True)
        self.dut._send_cmd_queue()
        for addr in range(self.N_RW_WORDS):
            self.dut._send_read(addr, queue=True)
        rdata = self.dut._send_cmd_queue()
        for addr, data in enumerate(rdata):
            self.assertEqual(data, (addr+1)**2)

    @parameterized.expand([
        ("", 0, 1, False),
        ("", 0, 64, False),
        ("", 3, 13, False),
        ("", 65, 1, False),
        ("", -1, 16, True),
        ("", 66, 1, True),
    ])
    def test_read_write_range(self, name, start_addr, n_words, raises):
        wdata = self.get_rand_words(n_words)
        if raises:
            with self.assertRaises(RuntimeError) as cm:
                self.dut._send_write_range(start_addr, wdata)
            with self.assertRaises(RuntimeError) as cm:
                rdata = self.dut._send_read_range(start_addr, n_words)
        else:
            self.dut._send_write_range(start_addr, wdata)
            rdata = self.dut._send_read_range(start_addr, n_words)
            self.assertEqual(rdata, wdata)

    @parameterized.expand([
        ("", [], []),
        ("", list(range(1)), [[0]]),
        ("", list(range(7)), [[0, 1, 2, 3], [4, 5, 6]]),
        ("", list(range(8)), [[0, 1, 2, 3], [4, 5, 6, 7]]),
        ("", list(range(9)), [[0, 1, 2, 3], [4, 5, 6, 7], [8]]),
    ])
    def test_fold_array(self, name, arr, gold):
        folded = self.dut.fold_array(arr, 4)
        self.assertEqual(folded, gold)
        unfolded = self.dut.unfold_array(folded)
        self.assertEqual(unfolded, arr)

    @parameterized.expand([
        ("", list(range(8)), 1, [[0], [1], [2], [3], [4], [5], [6], [7]]),
        ("", list(range(8)), 2, [[0, 1], [2, 3], [4, 5], [6, 7]]),
        ("", list(range(8)), 3, [[0, 1, 2], [3, 4, 5], [6, 7]]),
        ("", list(range(8)), 7, [[0, 1, 2, 3, 4, 5, 6], [7]]),
        ("", list(range(8)), 8, [[0, 1, 2, 3, 4, 5, 6, 7]]),
        ("", list(range(8)), 9, [[0, 1, 2, 3, 4, 5, 6, 7]]),
    ])
    def test_fold_array_width(self, name, arr, width, gold):
        folded = self.dut.fold_array(arr, width)
        self.assertEqual(folded, gold)
        unfolded = self.dut.unfold_array(folded)
        self.assertEqual(unfolded, arr)

    @parameterized.expand([
        ("", [0],             0x00),
        ("", [0, 1],          0x0100),
        ("", [0, 1, 2],       0x020100),
        ("", [0, 1, 2, 3],    0x03020100),
        ("", [3, 2, 1, 0],    0x010203),
        ("", [3, 2, 1],       0x010203),
        ("", [0, 0, 0],       0),
        ("", [1, 0, 0, 0, 0], 1),
    ])
    def test_byte_array_to_int(self, name, arr, gold):
        v = self.dut.byte_array_to_int(arr)
        self.assertEqual(v, gold)

    @parameterized.expand([
        ("",  0, 0x00,       [0]),
        ("",  1, 0x00,       [0]),
        ("",  7, 0x00,       [0]),
        ("",  8, 0x00,       [0]),
        ("",  9, 0x00,       [0, 0]),
        ("",  8, 0x01,       [1]),
        ("",  9, 0x01,       [1, 0]),
        ("",  9, 0x0100,     [0, 1]),
        ("", 16, 0x0100,     [0, 1]),
        ("", 24, 0x020100,   [0, 1, 2]),
        ("", 32, 0x03020100, [0, 1, 2, 3]),
        ("", 32, 0x010203,   [3, 2, 1, 0]),
        ("", 64, 0x010203,   [3, 2, 1, 0, 0, 0, 0, 0]),
        ("", 24, 0x010203,   [3, 2, 1]),
        ("", 17, 0x010203,   [3, 2, 1]),
    ])
    def test_int_to_byte_array(self, name, n_bits, v, gold):
        arr = self.dut.int_to_byte_array(v, n_bits)
        self.assertEqual(arr, gold)

    @parameterized.expand([
        ("", [], []),
        ("", [165, 0, 0, 0], [0xa5]),
        ("", [165], [0xa5]),
        ("", [3, 2, 1, 0], [0x010203]),
        ("", [3, 2, 1], [0x010203]),
        ("", list(range(8)), [0x03020100, 0x07060504]),
        ("", list(range(9)), [0x03020100, 0x07060504, 0x08]),
        ("", list(range(7)), [0x03020100, 0x060504]),
    ])
    def test_byte_array_to_word_array(self, name, arr, gold):
        wa = self.dut.byte_array_to_word_array(arr)
        self.assertEqual(wa, gold)

    @parameterized.expand([
        ("", [], []),
        ("", [0xa5], [165, 0, 0, 0]),
        ("", [0x010203], [3, 2, 1, 0]),
        ("", [0x03020100, 0x07060504], [0, 1, 2, 3, 4, 5, 6, 7]),
        ("", [0x03020100, 0x07060504, 0x08], [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0]),
        ("", [0x03020100, 0x060504], [0, 1, 2, 3, 4, 5, 6, 0]),
    ])
    def test_word_array_to_byte_array(self, name, arr, gold):
        ba = self.dut.word_array_to_byte_array(arr)
        self.assertEqual(ba, gold)

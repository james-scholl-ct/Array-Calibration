import os
import unittest
from parameterized import parameterized
from testconfig import config
from functools import reduce
import math

import python_tools.sim_source_builder as builder


class LotusSPITestClass(unittest.TestCase):
    _MODULE= 'lotus'

    def setUp(self):
        super(LotusSPITestClass, self).setUp()
        self.ops = []

    def tearDown(self):
        super(LotusSPITestClass, self).tearDown()

    @classmethod
    def setUpClass(cls):
        super(LotusSPITestClass, cls).setUpClass()
        if int(config.get('build_clean', '0')):
            cls.cwd = builder.build(cls._MODULE, cls._MODULE)
        else:
            cls.cwd = os.getcwd()
            os.chdir(cls._MODULE)

    @classmethod
    def tearDownClass(cls):
        super(LotusSPITestClass, cls).tearDownClass()
        os.chdir(cls.cwd)

    # -------------------------------------------------------------------------
    # Sim helpers
    # -------------------------------------------------------------------------
    def read_cmd(self, addr, periph='bsc'):
        cmd = 'read_{} {:x}'.format(periph, addr)
        self.ops.append(cmd)

    def write_cmd(self, addr, data, periph='bsc'):
        cmd = 'write_{} {:x} {:x}'.format(periph, addr, data)
        self.ops.append(cmd)

    def wait_cmd(self, delay_ns):
        cmd = 'wait {:d}'.format(delay_ns)
        self.ops.append(cmd)

    def write_iv_file(self):
        with open("input_vector.txt", "w") as f:
            for op in self.ops:
                f.write(op + "\n")

    def parse_ov_file(self):
        with open("output_vector.txt", "r") as f:
            self.rsp = f.readlines()

    # -------------------------------------------------------------------------
    # Lotus commands
    # -------------------------------------------------------------------------
    def send_config_clkdivs(self, arr):
        data = reduce((lambda x, y: x | y[1] << 4*y[0]), enumerate(arr), 0)
        self.write_cmd(24, data, periph='spi')

    def send_fill_cmd_fifo(self, val):
        self.write_cmd(25, (val & 0x1) << 0, periph='spi')

    def send_cmd_to_fifo(self, cmd, slave, payload):
        if not 0 <= cmd <= 2:
            raise Exception("Value {} for cmd is not supported".format(cmd))
        if not 0 <= slave < 8:
            raise Exception("Value {} for slave is not supported".format(slave))
        if not 0 <= payload < 2**24:
            raise Exception("Value {} for payload cannot exceed 24 bits".format(payload))
        data = cmd << 28 | slave << 24 | payload
        self.write_cmd(26, data, periph='spi')

    def send_read_rsp_fifo(self):
        self.read_cmd(30, periph='spi')

    def send_read_status(self):
        self.read_cmd(31, periph='spi')

    # -------------------------------------------------------------------------
    # Unit Tests
    # -------------------------------------------------------------------------
    def test_axi_partitions(self):
        for i in range(20):
            self.write_cmd(i, i | 1<<31, periph='spi')
            self.write_cmd(i, i, periph='bsc')

        for i in range(20):
            self.read_cmd(i, periph='bsc')

        for i in range(20):
            self.read_cmd(i, periph='spi')

        self.write_iv_file()
        builder.xsim(self._MODULE)
        self.parse_ov_file()
        bsc_reads = [line for line in self.rsp if line[0:8] == 'read_bsc']
        spi_reads = [line for line in self.rsp if line[0:8] == 'read_spi']
        for line in bsc_reads:
            cmd, addr, rdata, rresp = line.strip().split()
            self.assertTrue(int(rdata, 16) == int(addr, 16))
            self.assertTrue(int(rresp) == 0)
        for line in spi_reads:
            cmd, addr, rdata, rresp = line.strip().split()
            self.assertTrue(int(rdata, 16) == int(addr, 16) | 1<<31)
            self.assertTrue(int(rresp) == 0)


    def test_clk_config(self):
        self.send_config_clkdivs(range(8))
        for slave in range(8):
            self.send_cmd_to_fifo(0, slave, 0xaaaaaa)
        self.wait_cmd(10000)

        self.write_iv_file()
        builder.xsim(self._MODULE)


    def test_cmd_fifo_thru(self):
        clk_div = 2
        self.send_config_clkdivs([clk_div]*8)
        for slave in range(8):
            self.send_cmd_to_fifo(0, slave, slave + 0x41)
        self.send_read_status()
        for slave in range(8):
            self.send_cmd_to_fifo(0, 1, 0xf0f0f0)
        self.send_read_status()
        for slave in range(8):
            self.send_cmd_to_fifo(0, 1, 0xa5a5a5)
        self.send_read_status()
        for slave in range(8):
            self.send_cmd_to_fifo(0, 1, 0xc3c3c3)
        self.send_read_status()
        delay_ns = int(10 * (clk_div + 1) * 16 * 16*1.25)
        self.wait_cmd(delay_ns)

        self.write_iv_file()
        builder.xsim(self._MODULE)


    def test_cmd_fifo_fill(self):
        clk_div = 2
        self.send_config_clkdivs([clk_div]*8)
        self.send_fill_cmd_fifo(1)
        for i in range(16):
            self.send_cmd_to_fifo(0, 3, i)
        self.send_read_status()

        self.send_fill_cmd_fifo(0)
        delay_ns = int(10 * (clk_div + 1) * 16 * 16*1.25)
        self.wait_cmd(delay_ns)
        self.send_read_status()

        self.write_iv_file()
        builder.xsim(self._MODULE)


    def test_config(self):
        for i in range(8):
            self.write_cmd(25, i, periph='spi')
        self.write_iv_file()
        builder.xsim(self._MODULE)


    @parameterized.expand([
        ("", 3),
        ("", math.ceil(204/16)),
        ("", 16),
    ])
    def test_jumbo_frame(self, name, n_frames):
        clk_div = 2
        self.send_config_clkdivs([clk_div]*8)
        frames = [1<<15 | i for i in range(n_frames + n_frames%2)]
        words = [frames[i+1]<<16 | frames[i] for i in range(0, n_frames, 2)]
        for addr, word in enumerate(words):
            self.write_cmd(addr, word, periph='spi')
            self.write_cmd(addr+8, ~word & 0xffffffff, periph='spi')
        self.send_cmd_to_fifo(1, 3, n_frames-1)
        delay_ns = int(10 * (clk_div + 1) * 16 * n_frames*1.25)
        self.wait_cmd(delay_ns)

        self.write_iv_file()
        builder.xsim(self._MODULE)


    @parameterized.expand([
        ("", 1, 0),
        ("", 1, 13),
        ("", 7, 0),
        ("", 7, 27),
        ("", 64, 0),
        ("", 64, 3),
    ])
    def test_stream(self, name, n_frames, n_wait_cycles):
        clk_div = 2
        self.send_config_clkdivs([clk_div]*8)
        payload = n_wait_cycles << 6 | (n_frames-1)
        self.send_cmd_to_fifo(2, 3, payload)
        for _ in range(4 * n_frames):
            self.send_read_rsp_fifo()
            delay_ns = int(10 * (n_wait_cycles + (clk_div + 1) * 18)) // 4
            self.wait_cmd(delay_ns)

        self.write_iv_file()
        builder.xsim(self._MODULE)

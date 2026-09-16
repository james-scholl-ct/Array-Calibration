import os
import unittest
from parameterized import parameterized
from testconfig import config

import python_tools.sim_source_builder as builder


class LotusSPITestClass(unittest.TestCase):
    MODULE= 'genesis'
    word_addr_offsets = {'ip0': 0,
                         'ip1': 1024}

    def setUp(self):
        super(LotusSPITestClass, self).setUp()
        self.ops = []

    def tearDown(self):
        super(LotusSPITestClass, self).tearDown()

    @classmethod
    def setUpClass(cls):
        super(LotusSPITestClass, cls).setUpClass()
        if int(config.get('build_clean', '0')):
            cls.cwd = builder.build(cls.MODULE, cls.MODULE)
        else:
            cls.cwd = os.getcwd()
            os.chdir(cls.MODULE)

    @classmethod
    def tearDownClass(cls):
        super(LotusSPITestClass, cls).tearDownClass()
        os.chdir(cls.cwd)

    # -------------------------------------------------------------------------
    # Sim helpers
    # -------------------------------------------------------------------------
    def read_cmd(self, addr, periph):
        offset = self.word_addr_offsets[periph]
        cmd = 'read {:x}'.format(addr + offset)
        self.ops.append(cmd)

    def write_cmd(self, addr, data, periph):
        offset = self.word_addr_offsets[periph]
        cmd = 'write {:x} {:x}'.format(addr + offset, data)
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
    # Memory map helpers
    # -------------------------------------------------------------------------
    def write_gpo(self, data, periph):
        self.write_cmd(54, data, periph)

    def write_cmd_fifo(self, data, periph):
        self.write_cmd(55, data, periph)

    def pop_rsp_fifo(self, periph):
        self.read_cmd(56, periph)

    def peek_rsp_fifo(self, periph):
        self.read_cmd(57, periph)

    def read_cmd_fifo_status(self, periph):
        self.read_cmd(58, periph)

    def read_gpi(self, periph):
        self.read_cmd(59, periph)

    # -------------------------------------------------------------------------
    # Unit Tests
    # -------------------------------------------------------------------------
    def test_axi_partitions(self):
        data = [i for i in range(56)]

        # write one, then the other
        for a, d in enumerate(data):
            self.write_cmd(a, d | 1<<31, 'ip1')
            self.write_cmd(a, d, 'ip0')

        # read the first
        for a, d in enumerate(data):
            self.read_cmd(a, 'ip1')

        # read the second
        for a, d in enumerate(data):
            self.read_cmd(a, 'ip0')

        # run the sim and parse results
        self.write_iv_file()
        builder.xsim(self.MODULE)
        self.parse_ov_file()

        # get the reads for each
        ip0_reads = []
        ip1_reads = []
        for line in self.rsp:
            tokens = line.strip().split()
            if tokens[0] == 'read':
                addr = int(tokens[1], 16)
                if addr < self.word_addr_offsets['ip1']:
                    ip0_reads.append(line)
                else:
                    ip1_reads.append(line)

        # compare expected vs sim
        for i, line in enumerate(ip0_reads):
            cmd, addr, rdata, rresp = line.strip().split()
            self.assertTrue(int(rdata, 16) == data[i])
            self.assertTrue(int(rresp) == 0)
        for i, line in enumerate(ip1_reads):
            cmd, addr, rdata, rresp = line.strip().split()
            self.assertTrue(int(rdata, 16) == data[i] | 1<<31)
            self.assertTrue(int(rresp) == 0)

    @parameterized.expand([
        ("", 'ip0'),
        ("", 'ip1'),
    ])
    def test_fifos(self, name, periph):
        # nearly fill the rsp fifo
        for i in range(30):
            data = ~(1 << i) & 0xffffffff
            self.write_cmd_fifo(data, periph)
            self.peek_rsp_fifo(periph)
        self.read_cmd_fifo_status(periph)

        # fill rsp fifo
        self.write_cmd_fifo(0xdeadbeef, periph)
        self.peek_rsp_fifo(periph)

        # overfill rsp fifo
        self.write_cmd_fifo(0xcafef00d, periph)
        self.peek_rsp_fifo(periph)

        # drain the rsp fifo
        for i in range(31 + 3):
            self.pop_rsp_fifo(periph)

        # run the sim
        self.write_iv_file()
        builder.xsim(self.MODULE)

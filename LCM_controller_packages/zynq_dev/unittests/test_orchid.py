import os
import unittest
from parameterized import parameterized
from testconfig import config

import python_tools.sim_source_builder as builder


class OrchidTestClass(unittest.TestCase):
    _MODULE= 'orchid'

    def setUp(self):
        super(OrchidTestClass, self).setUp()
        self.ops = []

    def tearDown(self):
        super(OrchidTestClass, self).tearDown()

    @classmethod
    def setUpClass(cls):
        super(OrchidTestClass, cls).setUpClass()
        if int(config.get('build_clean', '0')):
            cls.cwd = builder.build(cls._MODULE, cls._MODULE)
        else:
            cls.cwd = os.getcwd()
            os.chdir(cls._MODULE)

    @classmethod
    def tearDownClass(cls):
        super(OrchidTestClass, cls).tearDownClass()
        os.chdir(cls.cwd)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def read_cmd(self, addr):
        cmd = 'read {:x}'.format(addr)
        self.ops.append(cmd)

    def write_cmd(self, addr, data):
        cmd = 'write {:x} {:x}'.format(addr, data)
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

    def send_reset(self):
        self.write_cmd(71, 1)

    def send_start(self, mode=0):
        if not 0 <= mode <= 3:
            raise Exception("Invalid mode value {}.".format(mode))
        self.write_cmd(71, mode << 2 | 1 << 1)

    def send_stop(self):
        self.write_cmd(71, 0)

    def set_dwell_normal(self, val):
        if not 0 <= val < 2**20:
            raise Exception("Value {} must be < 2**20".format(val))
        self.write_cmd(69, val)

    def set_dwell_mode00(self, val):
        if not 0 <= val < 2**20:
            raise Exception("Value {} must be < 2**20".format(val))
        self.write_cmd(68, val)

    # -------------------------------------------------------------------------
    # Unit Tests
    # -------------------------------------------------------------------------
    #@unittest.skip("This test is disabled.")
    def test_axi_loopback(self):
        for i in range(72):
            self.write_cmd(i, i)

        for i in range(72):
            self.read_cmd(i)

        self.write_iv_file()
        builder.xsim(self._MODULE)
        self.parse_ov_file()
        reads = [line for line in self.rsp if line[0:4] == 'read']
        match = True
        for line in reads:
            cmd, addr, rdata, rresp = line.strip().split()
            if int(addr, 16) == 66:
                self.assertTrue(int(rdata, 16) == 64+65)
            else:
                self.assertTrue(int(rdata, 16) == int(addr, 16))
            self.assertTrue(int(rresp) == 0)


    @parameterized.expand([
        ("", 0),
        ("", 1),
        ("", 2),
        ("", 3),
    ])
    def test_mode(self, name, mode):
        self.send_reset()

        for addr in range(64):
            if addr % 3 == 0:
                self.write_cmd(addr, 0xefbeadde)
            elif addr % 3 == 1:
                self.write_cmd(addr, 0xcadecefa)
            else:
                self.write_cmd(addr, 0xeeffc0f1)

        if mode == 0:
            self.set_dwell_mode00(17)
            delay_ns = 20e3
        else:
            # 1500 cycles at 120MHz = 12.5 us. min time between TP1 posedges
            # when reprogramming is 12.167 us = 1460 cycles at 120MHz.
            self.set_dwell_normal(1500)
            delay_ns = 80e3

        self.send_start(mode)
        self.wait_cmd(int(delay_ns))
        self.send_stop()
        self.wait_cmd(int(25e3))

        self.write_iv_file()
        builder.xsim(self._MODULE)


    @parameterized.expand([
        ("", 0),
        ("", 1),
        ("", 2),
        ("", 3),
    ])
    def test_restart(self, name, mode):
        self.send_reset()

        for addr in range(64):
            if addr % 3 == 0:
                self.write_cmd(addr, 0xefbeadde)
            elif addr % 3 == 1:
                self.write_cmd(addr, 0xcadecefa)
            else:
                self.write_cmd(addr, 0xeeffc0f1)

        if mode == 0:
            self.set_dwell_mode00(17)
            delay_ns = 20e3
        else:
            # 1500 cycles at 120MHz = 12.5 us. min time between TP1 posedges
            # when reprogramming is 12.167 us = 1460 cycles at 120MHz.
            self.set_dwell_normal(1500)
            delay_ns = 80e3

        self.send_start(mode)
        self.wait_cmd(int(25e3))
        self.send_stop()
        self.wait_cmd(int(25e3))
        self.send_start(mode)
        self.wait_cmd(int(delay_ns))

        self.write_iv_file()
        builder.xsim(self._MODULE)

import os
import unittest
from parameterized import parameterized
from testconfig import config

import python_tools.sim_source_builder as builder


class LotusTestClass(unittest.TestCase):
    _MODULE= 'lotus'

    def setUp(self):
        super(LotusTestClass, self).setUp()
        self.ops = []

    def tearDown(self):
        super(LotusTestClass, self).tearDown()

    @classmethod
    def setUpClass(cls):
        super(LotusTestClass, cls).setUpClass()
        if int(config.get('build_clean', '0')):
            cls.cwd = builder.build(cls._MODULE, cls._MODULE)
        else:
            cls.cwd = os.getcwd()
            os.chdir(cls._MODULE)

    @classmethod
    def tearDownClass(cls):
        super(LotusTestClass, cls).tearDownClass()
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
    def send_reset(self):
        self.write_cmd(62, 1<<0)

    def send_init(self):
        self.write_cmd(62, 1<<1)

    def send_enable(self, apply=1):
        self.write_cmd(62, 1<<2 | apply<<3)

    def send_apply_table(self):
        self.write_cmd(62, 1<<2 | 1<<3)

    def send_stop(self):
        self.write_cmd(62, 0)

    def send_start_txfer(self, angle_index):
        self.write_cmd(60, angle_index)

    def send_fire_laser(self):
        # for both enabling continuous mode and triggering single-shot mode
        self.write_cmd(59, 1)

    def send_stop_laser(self):
        self.write_cmd(59, 0)

    def send_config_laser(self,
                          clks_per_interval,
                          pulses_per_frame,
                          intervals_per_frame):
        if not 0 <= clks_per_interval < 2**8:
            raise Exception("Value {} must be < 2**8".format(clks_per_interval))
        if not 0 <= pulses_per_frame < 2**8:
            raise Exception("Value {} must be < 2**8".format(pulses_per_frame))
        if not 0 <= intervals_per_frame < 2**16:
            raise Exception("Value {} must be < 2**16".format(intervals_per_frame))

        data = (intervals_per_frame << 16 |
                pulses_per_frame << 8 |
                clks_per_interval)
        self.write_cmd(58, data)

    def set_dwell(self, val):
        if not 0 <= val < 2**20:
            raise Exception("Value {} must be < 2**20".format(val))
        self.write_cmd(61, val)

    def set_bsc_config(self, **kwargs):
        dwell_cnt = kwargs['dwell_cnt']
        ito_tc = kwargs['ito_tc']
        ito_invert = kwargs['ito_invert']

        if not 0 <= dwell_cnt < 2**20:
            msg = "Dwell value ({}) must be on [0, 2**20).".format(dwell_cnt)
            raise RuntimeError(msg)
        if not 0 <= ito_tc < 2**8:
            msg = "ITO TC value ({}) must be on [0, 2**8).".format(ito_tc)
            raise RuntimeError(msg)
        if ito_invert not in (0, 1):
            msg = "ITO invert ({}) must be zero or one.".format(ito_invert)
            raise RuntimeError(msg)

        val = (ito_invert << 28) | (ito_tc << 20) | dwell_cnt
        self.write_cmd(61, val)

    def send_read_status(self):
        self.read_cmd(63)

    # -------------------------------------------------------------------------
    # Unit Tests
    # -------------------------------------------------------------------------
    #@unittest.skip("This test is disabled.")
    def test_axi_loopback(self):
        for i in range(60):
            self.write_cmd(i, i)

        for i in range(60):
            self.read_cmd(i)

        self.write_iv_file()
        builder.xsim(self._MODULE)
        self.parse_ov_file()
        reads = [line for line in self.rsp if line[0:4] == 'read']
        for line in reads:
            cmd, addr, rdata, rresp = line.strip().split()
            if int(addr, 16) == 63:
                self.assertTrue(int(rdata, 16) == 0)
            else:
                self.assertTrue(int(rdata, 16) == int(addr, 16))
            self.assertTrue(int(rresp) == 0)


    @parameterized.expand([
        ("", 0),
        ("", 1),
        ("", 126),
        ("", 127),
    ])
    def test_transfer(self, name, angle_idx):
        self.send_start_txfer(angle_idx)
        delay_ns = 2000
        self.send_read_status()
        self.wait_cmd(int(delay_ns))
        self.send_read_status()

        self.write_iv_file()
        builder.xsim(self._MODULE)


    @parameterized.expand([
        ("", 0, [0, 1, 7]),
        ("", 1, [0, 1]),
        ("", 17, [0, 1, 16, 17]),
    ])
    def test_laser(self, name, intervals_per_frame, pulses_per_frame_list):
        for pulses_per_frame in pulses_per_frame_list:
            for clks_per_interval in [0, 1, 4, 5, 24]:
                self.send_config_laser(clks_per_interval,
                                       pulses_per_frame,
                                       intervals_per_frame)
                self.send_fire_laser()
                clks_per_frame = (clks_per_interval + 1) * (intervals_per_frame + 1)
                delay_ns = 4.5 * clks_per_frame / 0.1
                self.wait_cmd(int(delay_ns))
                if intervals_per_frame != 0:
                    self.send_stop_laser()
                    delay_ns = clks_per_frame / 0.1
                    self.wait_cmd(int(delay_ns))

        self.write_iv_file()
        builder.xsim(self._MODULE)


    def test_bsc(self):
        self.send_reset()
        # 1500 cycles at 120MHz = 12.5 us.
        self.set_dwell(1500)
        delay_ns_prog = 16e3
        delay_ns_steer = 6 * delay_ns_prog

        self.send_init()
        self.wait_cmd(int(delay_ns_prog))
        self.send_enable()
        self.wait_cmd(int(delay_ns_steer))

        for addr in range(204//4):
            if addr % 3 == 0:
                self.write_cmd(addr, 0xefbeadde)
            elif addr % 3 == 1:
                self.write_cmd(addr, 0xcadecefa)
            else:
                self.write_cmd(addr, 0xeeffc0f1)

        self.send_apply_table()
        self.wait_cmd(int(delay_ns_steer))
        self.send_stop()
        self.wait_cmd(int(delay_ns_steer))

        self.write_iv_file()
        builder.xsim(self._MODULE)


    @parameterized.expand([
        ("", 0, 0),
        ("", 0, 1),
        ("", 1, 0),
        ("", 3, 0),
    ])
    def test_ito(self, name, ito_tc, ito_invert):
        fields = {'ito_invert': ito_invert,
                  'ito_tc': ito_tc,
                  'dwell_cnt': 1500}
        self.send_reset()
        self.set_bsc_config(**fields)
        delay_ns_prog = 16e3
        delay_ns_steer = 10 * delay_ns_prog

        self.send_init()
        self.wait_cmd(int(delay_ns_prog))
        self.send_enable()
        self.wait_cmd(int(delay_ns_steer))
        self.send_stop()
        self.wait_cmd(int(delay_ns_steer))

        self.write_iv_file()
        builder.xsim(self._MODULE)

import numpy as np
import os
from parameterized import parameterized
import random
import re
import scipy.signal as ss
import sys
import testconfig as tc
import unittest

import python_tools.sim_source_builder as builder


class BcudaTestClass(unittest.TestCase):
    MODULE= 'bcuda'
    word_addr_offsets = {'ip0': 0,
                         'ip1': 1024}

    def setUp(self):
        super(BcudaTestClass, self).setUp()
        self.ops = []

        # setup PRNG
        def_seed = random.randrange(sys.maxsize)
        self.random_seed = int(tc.config.get('random_seed', def_seed))
        random.seed(self.random_seed)
        np.random.seed(self.random_seed % 2**32)
        seed_file = os.path.abspath(os.path.dirname(__file__))
        seed_file = os.path.join(seed_file, 'random_seeds.txt')
        with open(seed_file, 'a') as f:
            f.write("{}: {}\n".format(self.id(), self.random_seed))

    def tearDown(self):
        super(BcudaTestClass, self).tearDown()

    @classmethod
    def setUpClass(cls):
        super(BcudaTestClass, cls).setUpClass()
        if int(tc.config.get('build_clean', '0')):
            cls.cwd = builder.build(cls.MODULE, cls.MODULE)
        else:
            cls.cwd = os.getcwd()
            os.chdir(cls.MODULE)

    @classmethod
    def tearDownClass(cls):
        super(BcudaTestClass, cls).tearDownClass()
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
    # Pattern helpers
    # -------------------------------------------------------------------------
    def get_rand_ints(self, n_bits, size):
        return np.random.randint(-2**(n_bits-1), 2**(n_bits-1), size)

    # -------------------------------------------------------------------------
    # Memory map helpers
    # -------------------------------------------------------------------------
    def write_debug_word(self, data, periph):
        if periph == 'ip0':
            self.write_cmd(21, data, periph)
        else:
            self.write_cmd(53, data, periph)

    def write_gpo(self, data, periph):
        if periph == 'ip0':
            self.write_cmd(22, data, periph)
        else:
            self.write_cmd(54, data, periph)

    def write_cmd_fifo(self, data, periph):
        if periph == 'ip0':
            self.write_cmd(23, data, periph)
        else:
            self.write_cmd(55, data, periph)

    def read_rsp_fifo(self, periph):
        if periph == 'ip0':
            self.read_cmd(24, periph)
        else:
            self.read_cmd(56, periph)

    def read_rsp_fifo_status(self, periph):
        if periph == 'ip0':
            self.read_cmd(25, periph)
        else:
            self.read_cmd(57, periph)

    def read_cmd_fifo_status(self, periph):
        if periph == 'ip0':
            self.read_cmd(26, periph)
        else:
            self.read_cmd(58, periph)

    def read_gpi(self, periph):
        if periph == 'ip0':
            self.read_cmd(27, periph)
        else:
            self.read_cmd(59, periph)

    # -------------------------------------------------------------------------
    # Unit Tests
    # -------------------------------------------------------------------------
    def test_axi_partitions(self):
        data = [i for i in range(24)]

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
    def test_fifos(self, name, periph, debug=False):
        golden = []
        depth = 512

        # loopback fifos, hold cmd_fifo
        self.write_debug_word(0<<5 | 0<<1 | 1, periph)

        # nearly fill the cmd fifo
        for i in range(depth-1):
            data = ~i & 0xffffffff
            self.write_cmd_fifo(data, periph)

        # check cmd fifo: 511 writes and almost full
        self.read_cmd_fifo_status(periph)
        golden.append(0 << 23 |
                      0 << 22 |
                      0 << 21 |
                      0 << 12 |
                      0 << 11 |
                      0 << 10 |
                      1 << 9  |
                      depth-1)

        # fill cmd fifo
        self.write_cmd_fifo(0x1eadbeef, periph)

        # check cmd fifo: 512 % 512 = 0 writes and full
        self.read_cmd_fifo_status(periph)
        golden.append(0 << 23 |
                      0 << 22 |
                      0 << 21 |
                      0 << 12 |
                      0 << 11 |
                      1 << 10 |
                      1 << 9  |
                      0)

        # overflow the cmd fifo
        self.write_cmd_fifo(0xcafef00d, periph)

        # check cmd fifo: no wcount update (werr pulsed but not caught)
        self.read_cmd_fifo_status(periph)
        golden.append(0 << 23 |
                      0 << 22 |
                      0 << 21 |
                      0 << 12 |
                      0 << 11 |
                      1 << 10 |
                      1 << 9  |
                      0)

        # check rsp fifo: empty
        self.read_rsp_fifo_status(periph)
        golden.append(0 << 23 |
                      1 << 22 |
                      1 << 21 |
                      0 << 12 |
                      0 << 11 |
                      0 << 10 |
                      0 << 9  |
                      0)

        # drain the cmd fifo and fill rsp fifo
        self.write_debug_word(0<<1 | 0, periph)
        self.wait_cmd(5*512 + 100)

        # check cmd fifo: 512 % 512 = 0 reads and empty
        self.read_cmd_fifo_status(periph)
        golden.append(0 << 23 |
                      1 << 22 |
                      1 << 21 |
                      0 << 12 |
                      0 << 11 |
                      0 << 10 |
                      0 << 9  |
                      0)

        # check rsp fifo: 512 % 512 = 0 writes and full
        self.read_rsp_fifo_status(periph)
        golden.append(0 << 23 |
                      0 << 22 |
                      0 << 21 |
                      0 << 12 |
                      0 << 11 |
                      1 << 10 |
                      1 << 9  |
                      0)

        # read all but one from rsp fifo
        for i in range(depth-1):
            self.read_rsp_fifo(periph)
            # check rsp fifo data: 30 LSBs looped back, MSB is valid flag
            golden.append(1<<31 | ~i & 0x7fffffff)

        # check rsp fifo: 511 reads and almost empty
        self.read_rsp_fifo_status(periph)
        golden.append(0 << 23 |
                      0 << 22 |
                      1 << 21 |
                      depth-1 << 12 |
                      0 << 11 |
                      0 << 10 |
                      0 << 9  |
                      0)

        # empty rsp fifo
        self.read_rsp_fifo(periph)
        # check rsp fifo data: valid and special last word
        golden.append(1<<31 | 0x1eadbeef)

        # check rsp fifo: 512 % 512 = 0 reads and empty
        self.read_rsp_fifo_status(periph)
        golden.append(0 << 23 |
                      1 << 22 |
                      1 << 21 |
                      0 << 12 |
                      0 << 11 |
                      0 << 10 |
                      0 << 9  |
                      0)

        # underflow rsp fifo
        self.read_rsp_fifo(periph)
        # check rsp fifo data: invalid but special last word remains
        golden.append(0 << 31 | 0x1eadbeef)

        # check rsp fifo: no rcount update (rerr pulsed but not caught)
        self.read_rsp_fifo_status(periph)
        golden.append(0 << 23 |
                      1 << 22 |
                      1 << 21 |
                      0 << 12 |
                      0 << 11 |
                      0 << 10 |
                      0 << 9  |
                      0)

        # run the sim
        self.write_iv_file()
        builder.xsim(self.MODULE)
        self.parse_ov_file()

        # parse the responses
        reads = []
        for line in self.rsp:
            cmd, addr, rdata, rresp = line.strip().split()
            if cmd == 'read':
                reads.append(int(rdata, 16))

        self.assertEqual(len(golden), len(reads))
        for i, (a, b) in enumerate(zip(golden, reads)):
            if debug:
                print(i, hex(a), hex(b))
            self.assertEqual(a, b)

    @unittest.skip("Limited utility with full interpolation tests working.")
    def test_interp(self, periph='ip0',
                          n_pts=50,
                          n_taps=15,
                          threads=2,
                          phase=0,
                          x_n_bits=16,
                          y_n_bits=16,
                          debug=False):
        # config
        interp = 2
        h_n_bits = 16
        if n_taps == 15:
            h = [-1779, 0, 1698, 0, -3104, 0, 9802, 15478,
                 9802, 0, -3104, 0, 1698, 0, -1779]
        elif n_taps == 31:
            h = [-291, 0, 305, 0, -465, 0, 697, 0, -1040, 0, 1628, 0, -2910, 0, 9047, 14272,
                 9047, 0, -2910, 0, 1628, 0, -1040, 0, 697, 0, -465, 0, 305, 0, -291]
        else:
            raise RuntimeError("invalid number of taps {}".format(n_taps))

        # input data
        data = self.get_rand_ints(x_n_bits, threads*n_pts)
        data = np.concatenate((data, np.zeros(threads*n_pts, dtype=int)))

        # golden model
        n_toss = (x_n_bits + h_n_bits - 1) - y_n_bits
        y_gold_fir = ss.upfirdn(h, data[phase::threads], up=interp, down=1)
        y_gold_round = np.round(y_gold_fir / 2**n_toss)
        y_gold_clip = np.clip(y_gold_round, -2**(y_n_bits-1), 2**(y_n_bits-1)-1)
        y_gold = y_gold_clip
        if debug:
            with open('gm.dat', 'w') as f:
                f.write('y_gold_fir{0}y_gold_round{0}y_gold_clip'.format(' '*4))
                for i, j, k in zip(y_gold_fir, y_gold_round, y_gold_clip):
                    f.write('{0}{fill}{1}{fill}{2}\n'.format(i, j, k, fill=' '*4))

        # rtl stimulus
        for d in data:
            # write value to input vector file as a hexadecimal bit string
            self.write_cmd_fifo(d % 2**x_n_bits, periph)
            self.pop_rsp_fifo(periph)

        # run the sim and parse results
        self.write_iv_file()
        builder.xsim(self.MODULE)
        self.parse_ov_file()

        # parse the responses
        y_parse = []
        for line in self.rsp:
            cmd, addr, rdata, rresp = line.strip().split()
            if cmd == 'read':
                word = int(rdata, 16)
                valid = word >> 31
                if valid != 0:
                    unsigned = word % 2**y_n_bits
                    if unsigned >= 2**(y_n_bits-1):
                        signed = unsigned - 2**y_n_bits
                    else:
                        signed = unsigned
                    y_parse.append(signed)
        y_fir = y_parse[phase%(threads//2)::threads//2]

        # check results
        idx_max = min(len(y_gold), len(y_fir))
        passing = np.all(y_gold[0:idx_max] == y_fir[0:idx_max])
        if not passing:
            for i, (gold, rtl) in enumerate(zip(y_gold[0:idx_max], y_fir[0:idx_max])):
                if gold != rtl:
                    print(i, gold, rtl)
        self.assertTrue(passing)

    @parameterized.expand([
        ("", 0),
        ("", 1),
        ("", 2),
        ("", 3),
        ("", 4),
        ("", 5),
        ("", 6),
        ("", 7)
    ])
    def test_interp_full(self, name,
                               phase,
                               periph='ip0',
                               n_pts=200,
                               debug=False):
        # config
        interp = 2
        threads = 8
        x_n_bits = 18
        h_n_bits = 16
        y_n_bits = 16
        h15 = [-44, 0, 834, 0, -4112, 0, 19722, 32767, 19722, 0, -4112, 0, 834, 0, -44]
        h31 = [-21, 0, 116, 0, -341, 0, 786, 0, -1589, 0, 3054, 0, -6226, 0, 20608, 32767,
               20608, 0, -6226, 0, 3054, 0, -1589, 0, 786, 0, -341, 0, 116, 0, -21]

        # configure rsp fifo for interpolation and right channel
        self.write_debug_word(phase<<5 | 1<<1 | 0, periph)

        # input data
        data = self.get_rand_ints(x_n_bits, (threads, n_pts))
        pad = np.zeros((threads, 50), dtype=int)
        data_gold = np.concatenate((data, pad), axis=1)
        data_rtl = np.reshape(data_gold.T, (-1,))

        # golden model
        #   The clip calls are not merely the overflow protection in the
        #   RTL rounding. The coefficients are loaded with 1 <= L1_norm < 2
        #   so an integer bit is actually clipped.
        n_toss_first = (x_n_bits + h_n_bits - 1) - y_n_bits
        n_toss = (y_n_bits + h_n_bits - 1) - y_n_bits
        y_min = -2**(y_n_bits-1)
        y_max = 2**(y_n_bits-1)-1
        y_gold = np.clip(data_gold, 0, 2**(x_n_bits-1)-1)
        y_gold = ss.upfirdn(h31, y_gold, up=interp, down=1)
        y_gold = np.round(y_gold / 2**n_toss_first)
        y_gold = np.clip(y_gold, y_min, y_max)
        y_gold = ss.upfirdn(h15, y_gold, up=interp, down=1)
        y_gold = np.round(y_gold / 2**n_toss)
        y_gold = np.clip(y_gold, y_min, y_max)
        y_gold = ss.upfirdn(h15, y_gold, up=interp, down=1)
        y_gold = np.round(y_gold / 2**n_toss)
        y_gold = np.clip(y_gold, y_min, y_max)
        if debug:
            for i in range(8):
                print('channel, sample index, golden value')
                for j in range(20):
                    print(i, j, y_gold[i][j])
                print("\n")

        # rtl stimulus
        for i, d in enumerate(data_rtl):
            # write value to input vector file as a hexadecimal bit string
            self.write_cmd_fifo(d % 2**x_n_bits, periph)
            if i > 50:
                self.read_rsp_fifo(periph)

        # run the sim and parse results
        self.write_iv_file()
        builder.xsim(self.MODULE)
        self.parse_ov_file()

        # parse the responses
        y_parse = []
        for line in self.rsp:
            cmd, addr, rdata, rresp = line.strip().split()
            if cmd == 'read':
                word = int(rdata, 16)
                valid = word >> 31
                if valid != 0:
                    unsigned = word % 2**y_n_bits
                    if unsigned >= 2**(y_n_bits-1):
                        signed = unsigned - 2**y_n_bits
                    else:
                        signed = unsigned
                    y_parse.append(signed)
        y_fir = y_parse

        # check results
        idx_max = min(len(y_gold[phase]), len(y_fir))
        passing = np.all(y_gold[phase][0:idx_max] == y_fir[0:idx_max])
        if not passing:
            for i, (gold, rtl) in enumerate(zip(y_gold[phase][0:idx_max], y_fir[0:idx_max])):
                if gold != rtl:
                    print(i, gold, rtl)
        self.assertTrue(passing)

    @parameterized.expand([
        ("", 0, "interp_0_neg_inputs_clip2zero"),
        ("", 0, "interp_1_neg_inputs_clip2zero"),
        ("", 0, "interp_2_neg_inputs_clip2zero"),
        ("", 0, "interp_3_neg_inputs_clip2zero"),
        ("", 0, "interp_4_neg_inputs_clip2zero")
    ])
    def test_interp_file(self, name,
                               phase,
                               vec_name,
                               periph='ip0',
                               debug=False):
        # file i/o
        config_path = os.path.join(os.path.dirname(__file__),
                                   'dat',
                                   vec_name,
                                   'TC.dat')
        x_vec_path = os.path.join(os.path.dirname(__file__),
                                  'dat',
                                  vec_name,
                                  'acc_mon0.dat')
        y_vec_path = os.path.join(os.path.dirname(__file__),
                                  'dat',
                                  vec_name,
                                  'hbf_mon0.dat')
        with open(config_path, 'r') as f:
            config_lines = f.readlines()
        with open(x_vec_path, 'r') as f:
            x_lines = f.readlines()
        with open(y_vec_path, 'r') as f:
            y_lines = f.readlines()
        config_pat = re.compile(r"(\w+) +(\w+)")
        pat = re.compile(r"([0-9a-fA-F]+ +){7}[0-9a-fA-F]+")

        # config
        threads = 8
        x_n_bits = 18
        y_n_bits = 16

        #   - configure rsp fifo for interpolation and right channel
        self.write_debug_word(phase<<5 | 1<<1 | 0, periph)

        #   - get dict from file
        config = {}
        for line in config_lines:
            m = config_pat.match(line)
            if m:
                k, v = m.group().split()
                config[k] = v

        #   - configure pk_det block
        #       - INT_SAMPLES is the num_samples-1 non-zero samples into the
        #         interpolator, after upsampling by 8 (e.g. 2095).
        #       - INT_PADDED_SAMPLES is the num_samples-1 samples into the
        #         interpolator, including padding and upsampling by 8
        #         (e.g. 2295). The ACC hardware will need to perform
        #         zero-stuffing on the readout side. My test does it to model
        #         the ACC.
        #       - PKDET_ON_IDX is the sample index at which to turn on the
        #         peak_detector (idle until this time).
        #       - PKDET_OFF_IDX is the sample index at which to turn off the
        #         peak detector. It must be less than or equal to
        #               (INT_PADDED_SAMPLES + 1) - 34 - 8
        #         The interpolator min latency is 34 samples and the 8 peak
        #         detectors are offset by 8 samples due to pipelining. It
        #         should be long enough to flush the meaningful samples from
        #         the interpolator hardware.
        #       - PKDET_GUARD corresponds to num_samples-1, which is what the
        #         hardware expects.
        mode = 0
        interp_samples = int(config['INT_SAMPLES'])
        interp_samples_padded = int(config['INT_PADDED_SAMPLES'])
        pk_det_on_idx = int(config['PKDET_ON_IDX'])
        pk_det_off_idx = int(config['PKDET_OFF_IDX'])
        guard_samples = int(config['PKDET_GUARD'], 16)
        num_windows = 13
        prog_fa_thresh = int(config['PKDET_THR'], 16)
        prog_fa_thresh_en = 1

        word_offset = 8
        self.write_cmd(word_offset + 0, guard_samples, periph)
        self.write_cmd(word_offset + 1, mode, periph)
        self.write_cmd(word_offset + 2, (pk_det_on_idx << 12) | (pk_det_off_idx), periph)
        self.write_cmd(word_offset + 3, num_windows - 1, periph)
        self.write_cmd(word_offset + 4, prog_fa_thresh, periph)
        self.write_cmd(word_offset + 5, prog_fa_thresh_en, periph)

        # input data
        data = np.zeros((threads, (interp_samples_padded+1)//8), dtype=int)
        i = 0
        for line in x_lines:
            m = pat.match(line)
            if m:
                unsigned = [int(s, 16) for s in m.group().split()]
                signed = [v if v < 2**(x_n_bits - 1) else v - 2**x_n_bits
                          for v in unsigned]
                data[:, i] = signed
                i += 1
        data_rtl = np.reshape(data.T, (-1,))

        # golden model
        #   The length of the interpolator output vector from the RX model is
        #   arbitrary but bounded above by 8*ACC_SAMPLES + 162 =
        #   2*(2*(2*ACC_SAMPLES + 31 - 1) + 15 - 1) + 15 - 1
        #   samples per output channel. No meaningful data after this many.
        y_gold = np.zeros((threads, len(y_lines)), dtype=int)
        i = 0
        for line in y_lines:
            m = pat.match(line)
            if m:
                unsigned = [int(s, 16) for s in m.group().split()]
                signed = [v if v < 2**(y_n_bits - 1) else v - 2**y_n_bits
                          for v in unsigned]
                y_gold[:, i] = signed
                i += 1
        y_gold = y_gold[:, 0:i]

        if debug:
            for i in range(8):
                print('channel, sample index, golden value')
                for j in range(20):
                    print(i, j, y_gold[i][j])
                print("\n")

        # rtl stimulus
        #   account for minimum latency, which is currently 34 samples
        for i, d in enumerate(data_rtl):
            # write value to input vector file as a hexadecimal bit string
            self.write_cmd_fifo(d % 2**x_n_bits, periph)
            if i >= 50:
                self.read_rsp_fifo(periph)
        for i in range(50):
            self.read_rsp_fifo(periph)

        # run the sim and parse results
        self.write_iv_file()
        #return
        builder.xsim(self.MODULE)
        self.parse_ov_file()

        # parse the responses
        y_parse = []
        for line in self.rsp:
            cmd, addr, rdata, rresp = line.strip().split()
            if cmd == 'read':
                word = int(rdata, 16)
                valid = word >> 31
                if valid != 0:
                    unsigned = word % 2**y_n_bits
                    if unsigned >= 2**(y_n_bits-1):
                        signed = unsigned - 2**y_n_bits
                    else:
                        signed = unsigned
                    y_parse.append(signed)
        y_fir = y_parse

        # check results
        idx_max = min(len(y_gold[phase]), len(y_fir))
        passing = np.all(y_gold[phase][0:idx_max] == y_fir[0:idx_max])
        if not passing:
            for i, (gold, rtl) in enumerate(zip(y_gold[phase][0:idx_max], y_fir[0:idx_max])):
                if gold != rtl:
                    print(i, gold, rtl)
        self.assertTrue(passing)

    @unittest.skip("Not using the matched filter.")
    def test_mf(self, channel=0,
                      periph='ip0',
                      n_pts=200,
                      debug=False):
        # config
        x_n_bits = 16
        h_n_bits = 16
        y_n_bits = 16
        h = [1106, 1468, 1812, 2114, 2358, 2532, 2633, 2662, 2623, 2524,
             2373, 2183, 1962, 1722, 1473, 1223]

        # configure rsp fifo for interpolation and right channel
        # configure cmd fifo for stimulus into matched filter
        self.write_debug_word(1<<8 | channel<<5 | 2<<1 | 0, periph)

        # input data
        data = self.get_rand_ints(x_n_bits, n_pts)
        pad = np.zeros(20, dtype=int)
        data = np.concatenate((data, pad))

        # golden model
        n_toss = (x_n_bits + h_n_bits - 1) - y_n_bits
        y_min = -2**(y_n_bits-1)
        y_max = 2**(y_n_bits-1)-1
        y_gold = data
        y_gold = ss.upfirdn(h, y_gold, up=1, down=1)
        y_gold = np.round(y_gold / 2**n_toss)
        y_gold = np.clip(y_gold, y_min, y_max)
        if debug:
            for i in range(20):
                print(i, y_gold[i])
            print("\n")

        # rtl stimulus
        for i, d in enumerate(data):
            # write value to input vector file as a hexadecimal bit string
            self.write_cmd_fifo(d % 2**x_n_bits, periph)
            if i > len(pad):
                self.read_rsp_fifo(periph)

        # run the sim and parse results
        self.write_iv_file()
        builder.xsim(self.MODULE)
        self.parse_ov_file()

        # parse the responses
        y_parse = []
        for line in self.rsp:
            cmd, addr, rdata, rresp = line.strip().split()
            if cmd == 'read':
                word = int(rdata, 16)
                valid = word >> 31
                if valid != 0:
                    unsigned = word % 2**y_n_bits
                    if unsigned >= 2**(y_n_bits-1):
                        signed = unsigned - 2**y_n_bits
                    else:
                        signed = unsigned
                    y_parse.append(signed)
        y_fir = y_parse

        # check results
        idx_max = min(len(y_gold), len(y_fir))
        passing = np.all(y_gold[0:idx_max] == y_fir[0:idx_max])
        if not passing:
            for i, (gold, rtl) in enumerate(zip(y_gold[0:idx_max], y_fir[0:idx_max])):
                if gold != rtl:
                    print(i, gold, rtl)
        self.assertTrue(passing)

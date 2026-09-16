import os
import random
import sys
import testconfig as tc
import unittest
import yaml

import python_tools.sim_source_builder as builder
import python_tools.delorean_mem_map as dmm

class DeloreanBaseTestClass(unittest.TestCase):
    _MODULE = 'delorean'

    @classmethod
    def setUpClass(cls):
        super(DeloreanBaseTestClass, cls).setUpClass()
        if int(tc.config.get('build_clean', '0')):
            xcompile_opts = {}
            xelab_opts = {}
            #if 't_step' in tc.config:
            #    xelab_opts['timescale'] = '1n/' + tc.config['t_step']
            #    xelab_opts['override_timeprecision'] = None
            cls.cwd = builder.build(cls._MODULE,
                                    where=cls._MODULE,
                                    xcompile_opts=xcompile_opts,
                                    xelab_opts=xelab_opts)
        else:
            cls.cwd = os.getcwd()
            os.chdir(cls._MODULE)
        cls.map = dmm.DeloreanMemMap()

    def setUp(self):
        super(DeloreanBaseTestClass, self).setUp()
        self.ops = []
        self.bram_init = [0] * 1024
        self.debug = bool(int(tc.config.get('debug', '0')))

        # setup PRNG
        def_seed = random.randrange(sys.maxsize)
        self.random_seed = int(tc.config.get('random_seed', def_seed))
        random.seed(self.random_seed)
        seed_file = os.path.abspath(os.path.dirname(__file__))
        seed_file = os.path.join(seed_file, 'random_seeds.txt')
        with open(seed_file, 'a') as f:
            f.write("{}: {}\n".format(self.id(), self.random_seed))

    def tearDown(self):
        super(DeloreanBaseTestClass, self).tearDown()

    @classmethod
    def tearDownClass(cls):
        super(DeloreanBaseTestClass, cls).tearDownClass()
        os.chdir(cls.cwd)

    def tb_read(self, addr):
        cmd = 'read {:x}'.format(addr)
        self.ops.append(cmd)

    def tb_write(self, addr, data):
        cmd = 'write {:x} {:x}'.format(addr, data)
        self.ops.append(cmd)

    def tb_wait(self, delay_ns):
        cmd = 'wait {:d}'.format(int(delay_ns))
        self.ops.append(cmd)

    def tb_comment(self, comment, max_length=200):
        fill_count = max_length - len(comment)
        if fill_count < 0:
            msg = "Comment too long ({}): {}.".format(fill_count, comment)
            raise RuntimeError(msg)
        fill = '_' * fill_count
        cmd = 'comment {}'.format(comment.replace(' ', '_') + fill)
        self.ops.append(cmd)

    def write_iv_file(self):
        with open("input_vector.txt", "w") as f:
            for op in self.ops:
                f.write(op + "\n")

    def parse_ov_file(self):
        with open("output_vector.txt", "r") as f:
            self.rsp = f.readlines()
        self.assertTrue(self.rsp != [])

    def write_bram_file(self):
        with open('bram_init.txt', 'w') as f:
            for data in self.bram_init:
                f.write("{:08x}\n".format(data))

    def run_sim(self):
        self.write_iv_file()
        self.write_bram_file()
        builder.xsim(self._MODULE)
        self.parse_ov_file()

    def get_rand_bytes(self, n_bytes):
        return [random.randint(0, 0xff) for i in range(n_bytes)]

    def get_rand_words(self, n_words):
        return [random.randint(0, 0xffff_ffff) for i in range(n_words)]

import os
import unittest
from parameterized import parameterized
from testconfig import config

from unittests.delorean_baseclass import DeloreanBaseTestClass


class DeloreanLcmTestClass(DeloreanBaseTestClass):
    @classmethod
    def setUpClass(cls):
        super(DeloreanLcmTestClass, cls).setUpClass()

    def setUp(self):
        super(DeloreanLcmTestClass, self).setUp()

    def tearDown(self):
        super(DeloreanLcmTestClass, self).tearDown()

    @classmethod
    def tearDownClass(cls):
        super(DeloreanLcmTestClass, cls).tearDownClass()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _set_fields(self, **kwargs):
        for k, v in kwargs.items():
            self.tb_write(self.map.get_field_addr('lcm', k),
                          self.map.get_field_mask('lcm', k, v))

    # -------------------------------------------------------------------------
    # Unit Tests
    # -------------------------------------------------------------------------
    def test_pass(self):
        self.assertTrue(True)

    def test_axi_rw(self):
        lcm_rw_words = range(24)
        spi_rw_words = range(26)
        cdma_rw_words = [6, 8]

        wdata = []
        for word in lcm_rw_words:
            wdata.append(word ^ 0xefff_ffff)
            addr = self.map['base_addrs']['lcm'] + (word << 2)
            self.tb_write(addr, wdata[-1])
            self.tb_read(addr)

        for word in spi_rw_words:
            wdata.append(word ^ 0xdfff_ffff)
            addr = self.map['base_addrs']['spi'] + (word << 2)
            self.tb_write(addr, wdata[-1])
            self.tb_read(addr)

        for word in cdma_rw_words:
            wdata.append(word ^ 0xbfff_ffff)
            addr = self.map['base_addrs']['cdma'] + (word << 2)
            self.tb_write(addr, wdata[-1])
            self.tb_read(addr)

        self.run_sim()

        reads = [line for line in self.rsp if line[0:4] == 'read']
        for i, line in enumerate(reads):
            cmd, addr, rdata, rresp = line.strip().split()
            rdata_int = int(rdata, 16)
            check = (rdata_int == wdata[i])
            if not check or self.debug:
                print("{} 0x{:08x} 0x{:08x}".format(i, rdata_int, wdata[i]))
            self.assertTrue(check)
            self.assertTrue(int(rresp) == 0)

    def test_lcm(self, tp1_period_ns=10000):
        gold = []
        self.bram_init = self.get_rand_words(len(self.bram_init))

        # timing epochs to help with simulation capture
        tp1_period = int(tp1_period_ns / 10) - 4
        tp1_pw = 49
        rst_pw = 5 - 1
        n_steps = 171 - 1
        tx_wait = 7
        prog_time_ns = 10 * (1 + (rst_pw + 1) + 1 +
                             4 * (n_steps + 1) + (tx_wait + 1) + 1)
        tp1_pw_ns = 10 * (tp1_pw + 1)
        init_time_ns = 6 * tp1_pw_ns + prog_time_ns
        if self.debug:
            print("PROG TIME (ns)", prog_time_ns)
            print("TP1 PW (ns)", tp1_pw_ns)
            print("INIT TIME (ns)", init_time_ns)

        self.tb_comment('config controller')
        # 
        self.tb_write(self.map.get_field_addr('lcm', 'tp1_period'),
                      self.map.get_field_mask('lcm', 'tp1_period', tp1_period))
        #
        word = self.map.get_field_mask('lcm', 'n_steps', n_steps)
        word |= self.map.get_field_mask('lcm', 'rst_pw', rst_pw)
        word |= self.map.get_field_mask('lcm', 'tx_wait', tx_wait)
        word |= self.map.get_field_mask('lcm', 'tp1_pw', tp1_pw)
        self.tb_write(self.map.get_field_addr('lcm', 'n_steps'), word)
        #
        word = self.map.get_field_mask('lcm', 'reset_code', 0xa5)
        word |= self.map.get_field_mask('lcm', 'pol_finish_ovr', 0)
        word |= self.map.get_field_mask('lcm', 'tp1_done_high', 0)
        self.tb_write(self.map.get_field_addr('lcm', 'reset_code'), word)
        #
        word = self.map.get_field_mask('lcm', 'aux_code_even', 0xca)
        word |= self.map.get_field_mask('lcm', 'aux_code_odd', 0xfe)
        self.tb_write(self.map.get_field_addr('lcm', 'aux_code_even'), word)
        #
        word = self.map.get_field_mask('lcm', 'ito_tc', 2)
        word |= self.map.get_field_mask('lcm', 'ito_invert', 0)
        word |= self.map.get_field_mask('lcm', 'ito_async', 0)
        self.tb_write(self.map.get_field_addr('lcm', 'ito_tc'), word)
        #
        word = self.map.get_field_mask('lcm', 'pol_ovr_en', 0)
        word |= self.map.get_field_mask('lcm', 'pol_ovr_val', 0)
        self.tb_write(self.map.get_field_addr('lcm', 'pol_ovr_en'), word)
        # 
        self.tb_write(self.map.get_field_addr('lcm', 'prog_trigger_mode'),
                      self.map.get_field_mask('lcm', 'prog_trigger_mode', 0))

        self.tb_comment('check for idle state')
        self.tb_read(self.map.get_field_addr('lcm', 'tcon_state_idle'))
        gold.append(self.map.get_field_mask('lcm', 'tcon_state_idle'))

        self.tb_comment('startup controller')
        self.tb_write(self.map.get_field_addr('lcm', 'lcd_en'),
                      self.map.get_field_mask('lcm', 'lcd_en'))
        self.tb_write(self.map.get_field_addr('lcm', 'tcon_reset'),
                      self.map.get_field_mask('lcm', 'tcon_reset'))
        self.tb_write(self.map.get_field_addr('lcm', 'tcon_enable'),
                      self.map.get_field_mask('lcm', 'tcon_enable'))

        self.tb_comment('check for done state after startup')
        self.tb_wait(init_time_ns)
        self.tb_read(self.map.get_field_addr('lcm', 'tcon_state_done'))
        gold.append(self.map.get_field_mask('lcm', 'tcon_state_done'))

        self.tb_comment('check not loading after startup')
        self.tb_read(self.map.get_field_addr('lcm', 'loading'))
        gold.append(self.map.get_field_mask('lcm', 'loading', 0))

        self.tb_comment('apply table 0 and check loading flags')
        self.tb_write(self.map.get_field_addr('lcm', 'apply0'),
                      self.map.get_field_mask('lcm', 'apply0'))
        self.tb_read(self.map.get_field_addr('lcm', 'loading'))
        gold.append(self.map.get_field_mask('lcm', 'loading', 0b01))

        self.tb_comment('check not loading after programming completes')
        self.tb_wait(prog_time_ns)
        self.tb_read(self.map.get_field_addr('lcm', 'loading'))
        gold.append(self.map.get_field_mask('lcm', 'loading', 0))

        self.tb_comment('apply table 1 and check not loading (queued)')
        self.tb_write(self.map.get_field_addr('lcm', 'apply1'),
                      self.map.get_field_mask('lcm', 'apply1'))
        self.tb_read(self.map.get_field_addr('lcm', 'loading'))
        gold.append(self.map.get_field_mask('lcm', 'loading', 0))

        self.tb_comment('check loading flags after apply is serviced')
        self.tb_wait(tp1_period_ns + (tp1_period_ns - prog_time_ns))
        self.tb_read(self.map.get_field_addr('lcm', 'loading'))
        gold.append(self.map.get_field_mask('lcm', 'loading', 0b10))

        self.tb_comment('check not loading after programming completes')
        self.tb_wait(prog_time_ns)
        self.tb_read(self.map.get_field_addr('lcm', 'loading'))
        gold.append(self.map.get_field_mask('lcm', 'loading', 0))

        self.tb_comment('run')
        self.tb_wait(3 * tp1_period_ns)

        self.tb_comment('stop')
        self.tb_write(self.map.get_field_addr('lcm', 'tcon_enable'),
                      self.map.get_field_mask('lcm', 'tcon_enable', 0))

        self.tb_comment('check for done state')
        self.tb_wait(4 * tp1_period_ns)
        self.tb_read(self.map.get_field_addr('lcm', 'tcon_state_done'))
        gold.append(self.map.get_field_mask('lcm', 'tcon_state_done'))

        self.run_sim()

        reads = [line for line in self.rsp if line[0:4] == 'read']
        for i, line in enumerate(reads):
            cmd, addr, rdata, rresp = line.strip().split()
            rdata_int = int(rdata, 16)
            check = (rdata_int == gold[i])
            if not check or self.debug:
                print("{} 0x{:08x} 0x{:08x}".format(i, rdata_int, gold[i]))
            self.assertTrue(check)
            self.assertTrue(int(rresp) == 0)

    @parameterized.expand([
        ("", 0, [0, 1, 7]),
        ("", 1, [0, 1]),
        ("", 17, [0, 1, 16, 17]),
    ])
    def test_laser(self, name, intervals_per_frame, pulses_per_frame_list):
        word = self.map.get_field_mask('lcm', 'tx_pwr_en', 1)
        word |= self.map.get_field_mask('lcm', 'tx_pwr_switch', 0)
        self.tb_write(self.map.get_field_addr('lcm', 'tx_pwr_en'), word)
        for pulses_per_frame in pulses_per_frame_list:
            for clks_per_interval in [0, 1, 4, 5, 24]:
                self._set_fields(clks_per_interval=clks_per_interval,
                                 pulses_per_frame=pulses_per_frame,
                                 intervals_per_frame=intervals_per_frame,
                                 laser_pw_sel=0x2000)
                self.tb_write(self.map.get_field_addr('lcm', 'laser_enable'),
                              self.map.get_field_mask('lcm', 'laser_enable', 1))
                clks_per_frame = ((clks_per_interval + 1) *
                                  (intervals_per_frame + 1))
                self.tb_wait(5 * clks_per_frame * 10)
                if intervals_per_frame != 0:
                    self.tb_write(self.map.get_field_addr('lcm', 'laser_enable'),
                                  self.map.get_field_mask('lcm', 'laser_enable', 0))
                    self.tb_wait(clks_per_frame * 10)
        word = self.map.get_field_mask('lcm', 'tx_pwr_en', 0)
        word |= self.map.get_field_mask('lcm', 'tx_pwr_switch', 1)
        self.tb_write(self.map.get_field_addr('lcm', 'tx_pwr_en'), word)
        self.run_sim()

    @parameterized.expand([
        ("", 0, 1),
    ])
    @unittest.skip("Issues with Zynq VIP need to be debugged.")
    def test_cdma(self, name, ddr_idx, bram_idx):
        ddr_addr = self.map['base_addrs']['hp0'] + 512*ddr_idx
        bram_addr = self.map['base_addrs']['bram_ctrl'] + 512*bram_idx
        cdma_offset = self.map['base_addrs']['cdma']
        cdma_sa  = 0x18
        cdma_da  = 0x20
        cdma_btt = 0x28
        #self.tb_write(0xF8008000 | 0x0000, 0x1) # 32-bit HP0 read
        #self.tb_write(0xF8008000 | 0x0014, 0x1) # 32-bit HP0 write
        self.tb_write(cdma_offset | cdma_sa, ddr_addr)
        self.tb_write(cdma_offset | cdma_da, bram_addr)
        self.tb_write(cdma_offset | cdma_btt, 2048)
        self.tb_wait(20*1000)

        self.run_sim()

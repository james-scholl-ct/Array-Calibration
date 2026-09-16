from functools import reduce
import os
import unittest
from parameterized import parameterized
from testconfig import config

from unittests.delorean_baseclass import DeloreanBaseTestClass


class DeloreanSpiTestClass(DeloreanBaseTestClass):
    @classmethod
    def setUpClass(cls):
        super(DeloreanSpiTestClass, cls).setUpClass()

    def setUp(self):
        super(DeloreanSpiTestClass, self).setUp()

    def tearDown(self):
        super(DeloreanSpiTestClass, self).tearDown()

    @classmethod
    def tearDownClass(cls):
        super(DeloreanSpiTestClass, cls).tearDownClass()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _set_fields(self, **kwargs):
        for k, v in kwargs.items():
            self.tb_write(self.map.get_field_addr('spi', k),
                          self.map.get_field_mask('spi', k, v))

    def _get_clkdivs_mask(self, arr):
        return reduce((lambda x, y: x | y[1] << 4*y[0]), enumerate(arr), 0)

    def _get_fifo_cmd(self, cmd, slave, payload):
        if not 0 <= cmd <= 2:
            raise Exception("Value {} for cmd is not supported".format(cmd))
        if not 0 <= slave < 8:
            raise Exception("Value {} for slave is not supported".format(slave))
        if not 0 <= payload < 2**24:
            raise Exception("Value {} for payload cannot exceed 24 bits".format(payload))
        return (cmd << 28 | slave << 24 | payload)

    #def _send_read_rsp_fifo(self, sel=0):
    #    field = 'rsp_fifo_{}'.format(sel)
    #    self.tb_read(self.map.get_field_addr('spi', field))

    #def _send_read_status(self):
    #    self.tb_read(self.map.get_field_addr('spi', 'cmd_fifo_count'))

    # -------------------------------------------------------------------------
    # Unit Tests
    # -------------------------------------------------------------------------
    def test_pass(self):
        self.assertTrue(True)

    def test_clk_config(self):
        data = self._get_clkdivs_mask(range(8))
        self.tb_write(self.map.get_field_addr('spi', 'clk_div_adc'), data)
        for slave in range(8):
            data = self._get_fifo_cmd(0, slave, 0xaaaaa0|slave)
            self.tb_write(self.map.get_field_addr('spi', 'cmd_fifo'),
                          self.map.get_field_mask('spi', 'cmd_fifo', data))
        self.tb_wait(10000)
        self.run_sim()

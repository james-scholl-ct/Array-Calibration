import copy
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lcm_board.voltage_translator_delta import VoltageTranslator
from python_tools import himax_model
from python_tools.delorean_client import DeloreanApi

VERBOSE = True
DEBUG = False


class InvalidVoltageError(Exception):
    pass


class DeloreanBoard:
    BYTES_PER_TABLE = 1024  # technically, per channel
    MAX_TABLES = 1024

    __instances = {}  # __instances is a dict of {'addr': object} in this class

    def __new__(cls,
                addr):
        """This function is how the class returns new instances"""

        # Put addresses to upper case to eliminate ambiguity
        addr = addr.upper()

        # If there is no instrument at this address, make one
        if not addr in cls.__instances:
            cls.__instances[addr] = super(DeloreanBoard, cls).__new__(cls)
            cls.__instances[addr].__initialized = False
        # Either way, there is now an instrument at addr. Return it.
        return cls.__instances[addr]

    def __init__(self,
                 addr):
        """Initiates Lotus Board

        Args:
            addr: address of the board, e.g. microzed-3a-a3-14
        """

        self.__initialized = True
        self.v_min = 0
        self.v_max = 18
        self.v_gnd = 9
        self.delv_max = 18
        self.channel_count = 1021
        self.outlier_list_tx = None
        self.outlier_list_rx = None
        self._ito_amplitude_vpp = None
        self._ito_slew = None
        self._tx_ci_a = None
        self._tx_ci_b = None
        self.scratch_table_idx = self.MAX_TABLES - 1
        self.z_score_threshold = 5
        self.cof = True

        # Initialization for Himax driver
        vgma = {'vgma1': 18,
                'vgma2': 18,
                'vgma9': 9,
                'vgma10': 9,
                'vgma11': 9,
                'vgma12': 9,
                'vgma19': 0,
                'vgma20': 0}
        self.h = himax_model.HimaxModel(vgma)
        self.vpt = VoltageTranslator(max_del_v=self.delv_max,
                                     error_factor=1,
                                     voltage_factor=0)

        # These four vectors are of length self.channel_count and are ordered
        # spatially by physical rail position--left to right--across the
        # surface of the LCM. Furthermore, by definition, rail #1 is the
        # left-most rail and rail numbers increment by one moving left to
        # right. Here are brief descriptions of the four vectors:
        #   * rails: the LCM rail (indexed from 1)
        #   * driver_pins: the Himax driver output pins (indexed from 1)
        #   * table_indices: the index into a pattern table for FPGA memories
        #   * high_vec: a list of 0 or 1 values corresponding to the parity of
        #     the driver pin. By convention, POL = 0 such that odd driver
        #     channels are mapped to [VSSA, HVDDA] (i.e. high_vec = 0) and
        #     even driver channels are mapped to [HVDDA, VDDA] (i.e.
        #     high_vec = 1).
        # Note: It is recommended to use iter_lcm_map() instead of accessing
        # these fields directly.
        self.rails = tuple(range(1, 1 + self.channel_count))
        self.driver_pins = tuple(self.map_rails_to_driver_pins(self.rails))
        self.table_indices = tuple([d - 1 for d in self.driver_pins])
        self.high_vec = tuple([1 - (d % 2) for d in self.driver_pins])

        # The Delorean silkscreen no longer matches the driver/lcm channels
        self.hv_tp_to_rail_map = {'OUT_1': 1,
                                  'OUT_2': 2,
                                  'OUT_3': 512,
                                  'OUT_4': 3,
                                  'OUT_1018': 510,
                                  'OUT_1019': 1020,
                                  'OUT_1020': 511,
                                  'OUT_1021': 1021}

        self.addr = addr
        self._conn = DeloreanApi(self.addr)
        self.connect()
        self.voltage_on()
        self.ito_amplitude_vpp = 0

    def connect(self):
        """Connects to the Zynq host"""
        self._conn.connect()
        self.clk_freq_mhz = self.get_fpga_clk_freq_mhz()
        self.config_delorean()
        self.init_delorean()
        self.init_patterns_v()
        if VERBOSE or DEBUG:
            print("Connected to " + self.addr)

    def disconnect(self):
        self._conn.exec_exit()
        self._conn.disconnect()

    def set_fields(self, periph, **kwargs):
        """Writes a set of fields to the indicated peripheral in the FPGA. NB:
        this function does read-modify-writes but the order of programming is
        not guaranteed due to kwargs being a dict. The user is responsible for
        ensuring proper ordering of register writes.
        """
        if periph == 'lcm':
            wr_func = self._conn.exec_write_lcm
            rd_func = self._conn.exec_read_lcm
        elif periph == 'spi':
            wr_func = self._conn.exec_write_spi
            rd_func = self._conn.exec_read_spi
        else:
            raise RuntimeError("Invalid periph: {}.".format(periph))

        for field, value in kwargs.items():
            addr = self._conn.map.get_field_word_addr(periph, field)
            mask = self._conn.map.get_field_mask(periph, field)
            vmask = self._conn.map.get_field_mask(periph, field, value)
            data = rd_func(addr)
            data = (data & ~mask) | vmask
            wr_func(addr, data)

    def get_fields(self, periph, *args):
        """Returns a dictionary of values for each of the given fields.
        """
        if periph == 'lcm':
            rd_func = self._conn.exec_read_lcm
        elif periph == 'spi':
            rd_func = self._conn.exec_read_spi
        else:
            raise RuntimeError("Invalid periph: {}.".format(periph))

        ret = {k: None for k in args}
        for field in args:
            addr = self._conn.map.get_field_word_addr(periph, field)
            data = rd_func(addr)
            ret[field] = self._conn.map.get_field_value(periph, field, data)
        return ret

    def return_all_lcm_fields(self, with_print=False):
        """Returns a dictionary of all lcm fields and their current values"""
        keys = self._conn.map['fields']['lcm'].keys()
        lcm_fields = self.get_fields('lcm', *keys)
        if with_print:
            for k, v in sorted(lcm_fields.items()):
                print(f'{k:<28s}  {v:08x}  {v}')
            print('')
        return lcm_fields

    def return_all_spi_fields(self, with_print=False, exclude_rsp_fifo=False):
        """Returns a dictionary of all spi fields and their current values.
        Warning: the two response FIFOs will pop data if you read them so
        use exclude_rsp_fifo=True if that is undesirable.
        """
        keys = self._conn.map['fields']['spi'].keys()
        if exclude_rsp_fifo:
            keys = set(keys) - {'rsp_fifo_0', 'rsp_fifo_1'}
        spi_fields = self.get_fields('spi', *keys)
        if with_print:
            for k, v in sorted(spi_fields.items()):
                print(f'{k:<28s}  {v:08x}  {v}')
            print('')
        return spi_fields

    def get_fpga_clk_freq_mhz(self):
        """Query the frequency of the FPGA's fabric clock. Returned value
        is in MHz.
        """
        addr = self._conn.map.get_field_word_addr('lcm', 'clk_freq')
        return self._conn.exec_read_lcm(addr)

    def tp1_period_from_us(self, time_us):
        """Convert TP1 period from real time, in us, to a value for the
        register tp1_period.
        """
        return int(time_us * self.clk_freq_mhz - 4)

    def tp1_period_from_ac_freq_hz(self, f_hz):
        """Convert LCM AC frequency, in Hz, to a value for the regster
        tp1_period.
        """
        time_us = 1 / (2 * f_hz) * 1e6
        return self.tp1_period_from_us(time_us)

    def ito_tc_async_from_khz(self, f_khz):
        """Convert ITO frequency, in kHz, to a value for the register ito_tc.
        This function is only valid if ITO is in async mode.
        """
        return int(500 / f_khz * self.clk_freq_mhz - 1)

    def get_laser_config(self,
                         prf_khz=20,
                         interval_us=1,
                         ppf=1,
                         pw_index=4):
        """Return all laser configuration per the provided settings. The
        return value is suitable for use with set_fields.
        """
        clks_per_interval = int(interval_us * self.clk_freq_mhz)
        prf_period_us = 1000 / prf_khz
        intervals_per_frame = math.ceil(prf_period_us / interval_us)
        pulses_per_frame = int(min(ppf, intervals_per_frame))

        if DEBUG:
            rep_rate_sec = (clks_per_interval *
                            intervals_per_frame /
                            (self.clk_freq_mhz * 1e6))
            msg = "Laser rep rate set to {:0.1f} kHz ({:0.1f} us)."
            msg = msg.format(1 / rep_rate_sec / 1e3, rep_rate_sec * 1e6)
            print(msg)

        return {'laser_pw_sel': (1 << pw_index),
                'clks_per_interval': (clks_per_interval - 1),
                'pulses_per_frame': (pulses_per_frame - 1),
                'intervals_per_frame': (intervals_per_frame - 1)}

    def config_delorean(self):
        """Configure some sensible non-zero (non-default) values with special
        mention of Bravo- and Delta-specific fields.
        """
        lcm_config = {
            'tp1_period': self.tp1_period_from_us(250),
            'reset_code': 0x00,  # 0xff for Bravo
            'pol_finish_ovr': 0,  # 1 for Bravo
            'tp1_done_high': 0,  # 1 for Bravo
            'ito_async': 1,
            'ito_invert': 0,
            'ito_tc': self.ito_tc_async_from_khz(1),
            'n_steps': 170,
            'rst_pw': 4,
            'tx_wait': 7,
            'tp1_pw': int(0.5 * self.clk_freq_mhz - 1),
            'prog_trigger_mode': 1}  # pulse mode
        laser_config = self.get_laser_config()
        lcm_config.update(laser_config)
        spi_config = {
            'clk_div_adc': 2,  # 33.3 MHz
            'clk_div_switch': 9,  # 10.0 MHz
            'clk_div_tmp': 9,  # 10.0 MHz
            'clk_div_pot_ito': 19,  # 5.0 MHz
            'clk_div_pot_tx': 19}  # 5.0 MHz
        self.set_fields('lcm', **lcm_config)
        self.set_fields('spi', **spi_config)

    def init_delorean(self):
        """Turns on power supplies for the LCD driver, the analog switches,
        and the Laser driver. Also, enables the TCON and configures the
        switches for daisy-chain mode. This function should be called after
        configuration of FPGA registers.
        """
        self.set_fields('lcm', lcd_en=1)
        self.set_fields('lcm', tx_pwr_en=0)
        self.set_fields('spi', daisy_en=1)
        time.sleep(1)
        self.set_fields('lcm', tcon_enable=1)
        self.spi_enable_daisy_chain_mode()

    def init_patterns_v(self):
        """Makes sure the Tx and Rx patterns on startup are set to all values at V_GND"""
        startup_table = [self.h.MIN_CODE] * self.BYTES_PER_TABLE
        self.set_table(idx=self.scratch_table_idx,
                       table_tx=startup_table,
                       table_rx=startup_table)
        for i in range(3):
            if self.lcm_prog_ready():
                self._conn.exec_cdma_down(self.scratch_table_idx, 0)
                return
            time.sleep(1)
        raise RuntimeError("Programing failed.")

    def enable_tcon(self):
        """Enables the TCON so that patterns can later be applied.
        """
        self.set_fields('lcm', tcon_enable=1)

    def disable_tcon(self):
        """Disables the TCON, causing the TCON to return to the DONE state.
        """
        self.set_fields('lcm', tcon_enable=0)

    def get_rx_table(self, idx):
        """Returns a table of pattern data from Delorean DDR RAM. The table
        corresponds to the RX channel.
        """
        return self.get_table(idx)[0]

    def get_tx_table(self, idx):
        """Returns a table of pattern data from Delorean DDR RAM. The table
        corresponds to the TX channel.
        """
        return self.get_table(idx)[1]

    def get_table(self, idx):
        """Returns a table of pattern data from Delorean DDR RAM. Two tables
        are returned as a tuple (table_rx, and table_tx).
        """
        if not 0 <= idx < self.MAX_TABLES:
            raise RuntimeError("Invalid index: {}.".format(idx))
        tables = self._conn.exec_get_table(idx)
        table_rx = []
        table_tx = []
        [table_rx.extend(tup) for tup in zip(tables[0::4], tables[1::4])]
        [table_tx.extend(tup) for tup in zip(tables[2::4], tables[3::4])]
        return (table_rx, table_tx)

    def set_rx_table(self, idx, table):
        """Sets a table of pattern data in Delorean DDR RAM. The table
        corresponds to the RX channel. The table for the TX channel is
        left unmodified.
        """
        self.set_table(idx, table, [])

    def set_tx_table(self, idx, table):
        """Sets a table of pattern data in Delorean DDR RAM. The table
        corresponds to the TX channel. The table for the RX channel is
        left unmodified.
        """
        self.set_table(idx, [], table)

    def set_table(self, idx, table_rx, table_tx):
        """Sets a table of pattern data in Delorean DDR RAM. Both the
        TX channel and RX channel tables are modified if provided. Only
        one channel's data needs to be provided.
        """
        if table_rx and len(table_rx) != self.BYTES_PER_TABLE:
            msg = "RX table has {} bytes but {} are expected."
            msg = msg.format(len(table_rx), self.BYTES_PER_TABLE)
            raise RuntimeError(msg)
        if table_tx and len(table_tx) != self.BYTES_PER_TABLE:
            msg = "TX table has {} bytes but {} are expected."
            msg = msg.format(len(table_tx), self.BYTES_PER_TABLE)
            raise RuntimeError(msg)
        if table_tx and table_rx:
            mask = 3
            lyst = []
            [lyst.extend(tup) for tup in zip(table_rx[0::2],
                                             table_rx[1::2],
                                             table_tx[0::2],
                                             table_tx[1::2])]
        elif table_tx:
            mask = 2
            lyst = table_tx
        elif table_rx:
            mask = 1
            lyst = table_rx
        else:
            raise RuntimeError("No tables given.")
        self._conn.exec_set_table(mask, idx, lyst)

    def lcm_prog_ready(self):
        """Returns a boolean indicating if it is safe to transfer a table
        from DDR into the LCM controller's buffer and/or apply the buffer
        to the LCD driver.
        """
        addr = self._conn.map.get_field_word_addr('lcm', 'apply_cache')
        apply_cache = self._conn.exec_read_lcm(addr)

        addr = self._conn.map.get_field_word_addr('lcm', 'loading')
        loading = self._conn.exec_read_lcm(addr)
        return (apply_cache == 0) and (loading == 0)

    def do_set_transfer_apply(self, idx, table_rx=[], table_tx=[]):
        """Sets a pattern in DDR, transfers the pattern to the LCM peripheral,
        and applies the pattern to the LCM. This function always uses the LCM
        controller's buffer 0.
        """
        self.set_table(idx, table_rx, table_tx)
        for i in range(3):
            if self.lcm_prog_ready():
                self._conn.exec_cdma_down(idx, 0)
                self.set_fields('lcm', apply0=1)
                return
            time.sleep(1)
        raise RuntimeError("Programing failed.")

    def iter_lcm_map(self, sort_pos=0):
        """Yields tuples of (rail, driver_pin, table_index, high_vec) for
        iteration. Iteration is done according to the order of rails (i.e.
        sort_pos = 0), by default. To sort the iteration by one of the other
        parameters, set sort_pos to the position of the parameter in the
        tuple.
        """
        for r, d, t, h in sorted(zip(self.rails,
                                     self.driver_pins,
                                     self.table_indices,
                                     self.high_vec),
                                 key=lambda tup: tup[sort_pos]):
            yield (r, d, t, h)

    def map_rails_to_driver_pins(self, rails):
        """Maps the given LCM rails (1-1021) to Himax driver pins, which are
        indexed from 1.
        """
        if self.cof:
            return [2 * r - 1 if r <= 511 else
                    2 * r - 1022
                    for r in rails]
        else:
            return [2 if r == 1 else
                    2 * r - 3 if r <= 511 else
                    2 * r - 1020
                    for r in rails]

    def map_driver_pins_to_rails(self, d_pins):
        """Maps the given Himax driver pins (1-1020 and 1022) to LCM rails,
        which are indexed from 1.
        """
        if self.cof:
            return [(d + 1) // 2 if d & 1 == 1 else
                    d // 2 + 511
                    for d in d_pins]
        else:
            return [1 if d == 2 else
                    (d + 3) // 2 if d & 1 == 1 else
                    d // 2 + 510
                    for d in d_pins]

    def spi_send_cmd(self, cmd, slave, payload):
        """Sends a command to a SPI slave device. Command and slave index maps
        are defined in the yaml file parsed by self._conn.map, an instance of
        DeloreanMemMap.
        """
        cmd_idx = self._conn.map['spi_cmd_map'][cmd]
        slave_idx = self._conn.map['spi_slave_map'][slave]
        if not 0 <= payload < 2 ** 24:
            msg = "Value {} for payload cannot exceed 24 bits".format(payload)
            raise RuntimeError(msg)
        data = cmd_idx << 28 | slave_idx << 24 | payload
        addr = self._conn.map.get_field_word_addr('spi', 'cmd_fifo')
        self._conn.exec_write_spi(addr, data)

    def spi_read_rsp(self, channel, n_rsp=1):
        """Reads a response from the corresponding SPI channel RSP FIFO,
        either 0 or 1. Refer to the command-response protocol of the SPI
        peripheral for information on when the RSP FIFO is written.
        """
        if channel != 0 and channel != 1:
            raise RuntimeError(f"Invalid channel: {channel}.")
        addr = self._conn.map.get_field_word_addr('spi', f'rsp_fifo_{channel}')
        data = [self._conn.exec_read_spi(addr) for _ in range(n_rsp)]
        if n_rsp == 1:
            return data[0]
        else:
            return data

    def spi_clear_rsp_fifo(self, channel):
        """Clears the RSP FIFO for a particular channel.
        """
        addr = self._conn.map.get_field_word_addr('spi', f'rsp_fifo_{channel}')
        self._conn.exec_write_spi(addr, 0)

    def spi_get_rsp_is_valid(self, rsp):
        """Evaluates whether a response returned from the RSP FIFO contains
        valid data.
        """
        return rsp >> 31 == 1

    def spi_get_rsp_fifo_count(self, rsp):
        """Parses a response returned from a RSP FIFO.
        """
        return rsp >> 19 & 0x3f

    def spi_get_rsp_slave_idx(self, rsp):
        """Parses a response returned from a RSP FIFO.
        """
        return rsp >> 16 & 0x7

    def spi_get_rsp_payload(self, rsp):
        """Parses a response returned from a RSP FIFO.
        """
        return rsp & 0xffff

    def spi_get_cmd_fifo_is_full(self):
        """Evaluates whether more commands can be pushed to the CMD FIFO.
        """
        addr = self._conn.map.get_field_word_addr('spi', 'cmd_fifo_full')
        data = self._conn.exec_read_spi(addr)
        return self._conn.map.get_field_value('spi', 'cmd_fifo_full', data)

    def spi_get_cmd_fifo_count(self):
        """Returns the number of commands presently in the CMD FIFO.
        """
        addr = self._conn.map.get_field_word_addr('spi', 'cmd_fifo_count')
        data = self._conn.exec_read_spi(addr)
        return self._conn.map.get_field_value('spi', 'cmd_fifo_count', data)

    def spi_get_cmd_fifo_headroom(self):
        """Returns the number of additional commands that the CMD FIFO can
        presetnly accept without overflowing.
        """
        return 15 - self.spi_get_cmd_fifo_count()

    def int_to_signed(self, val, n_bits, n_fractional):
        """Converts an unsigned integer to a signed, fixed-point value.
        """
        if val < 2 ** (n_bits - 1):
            tmp = val
        else:
            tmp = val - 2 ** n_bits
        return tmp * 2 ** (-n_fractional)

    def spi_read_temp_sensor(self, use_c=False):
        '''Read the SPI temperature sensor and return the temperature in the
        indicated units.
        '''
        self.spi_send_cmd('standard', 'temp', 0x0000)
        data = self.spi_read_rsp(0)
        self.spi_read_rsp(1)
        if DEBUG:
            print("spi_read_temp_sensor")
            print("    Word = 0x{:08x}".format(data))
            print("    RSP Valid = {}".format(self.spi_get_rsp_is_valid(data)))
            print("    RSP FIFO Count = {}".format(self.spi_get_rsp_fifo_count(data)))
            print("    RSP SLAVE_IDX = {}".format(self.spi_get_rsp_slave_idx(data)))
            print("    RSP Payload = 0x{:04x}".format(self.spi_get_rsp_payload(data)))
        if not self.spi_get_rsp_is_valid(data):
            return None
        deg_c = self.int_to_signed(data & 0xfff8, 16, 7)
        deg_f = 9 / 5 * deg_c + 32
        if DEBUG:
            print("    Temp is {:.4f}F ({:.4f}C)\n".format(deg_f, deg_c))
        if use_c:
            return deg_c
        else:
            return deg_f

    def spi_read_adcs_simple(self):
        """Read a single sample from each of the two ADCs, in parallel. The
        returned value is a tuple of voltages: (voltage_adc0, voltage_adc1).
        Each value can be a -1 if an error occurred.
        """
        # sample and convert
        self.spi_send_cmd('standard', 'adc', 0x0000)
        # readout
        self.spi_send_cmd('standard', 'adc', 0x0000)

        if DEBUG:
            print("spi_read_adcs_single()")
        data = [None, None]
        voltages = [-1, -1]
        for i in range(2):
            data[i] = self.spi_read_rsp(i)
            data[i] = self.spi_read_rsp(i)
            if DEBUG:
                print("  Word_{} = 0x{:08x}".format(i, data[i]))
                print("    RSP Valid = {}".format(self.spi_get_rsp_is_valid(data[i])))
                print("    RSP FIFO Count = {}".format(self.spi_get_rsp_fifo_count(data[i])))
                print("    RSP SLAVE_IDX = {}".format(self.spi_get_rsp_slave_idx(data[i])))
                print("    RSP Payload = 0x{:04x}".format(self.spi_get_rsp_payload(data[i])))
            if not self.spi_get_rsp_is_valid(data[i]):
                continue
            voltages[i] = 0.00025 * ((data[i] >> 2) & 0x3fff)
            if DEBUG:
                print("    Voltage_{} = {:.5f}V.".format(i, voltages[i]))
                print("")
        return voltages

    def spi_read_adcs_stream(self, sample_interval_us, n_frames=63):
        """Sends a STREAM SPI command to sample n_frames times with sample
        interval sample_interval_us microseconds. RSP FIFOs are not accessed.
        """
        n_wait_cycles, actual_sample_interval_ns = \
            self.get_n_wait_cycles(sample_interval_us * 1000)
        actual_sample_interval_us = actual_sample_interval_ns / 1000

        if not 0 <= n_frames < 64:
            raise RuntimeError(f'Invalid value for n_frames: {n_frames}.')
        sampling_error = (actual_sample_interval_us -
                          sample_interval_us) / sample_interval_us
        if sampling_error > 0.01:
            msg = "Sample interval is off. You requested {} us but got {} us."
            raise RuntimeError(msg.format(sample_interval_us,
                                          actual_sample_interval_us))

        payload = n_wait_cycles << 6 | n_frames
        self.spi_send_cmd('stream', 'adc', payload)

    def spi_set_adc_ch_sel(self, tx_or_rx):
        """Sets the adc_ch_sel register according to the given channel
        ('rx' or 'tx').
        """
        if tx_or_rx == 'rx':
            self.set_fields('spi', adc_ch_sel=0b11)
        elif tx_or_rx == 'tx':
            self.set_fields('spi', adc_ch_sel=0b00)
        else:
            raise RuntimeError(f'Invalid channel selection: {tx_or_rx}.')

    def spi_enable_daisy_chain_mode(self):
        """Enable daisy-chain mode for the analog switches. A reset via the
        daisy_en configuration bit is required to exit daisy-chain mode.
        """
        self.spi_send_cmd('standard', 'switch', 0x2500)
        self.spi_read_rsp(0)
        self.spi_read_rsp(1)

    def spi_get_all_switch_signals(self, incl_rx=True, incl_tx=True):
        """Returns a list of all signals that can be used for
        spi_set_switch_data().
        """
        rx_signals = self._conn.map['spi_switch_map']['rx']['signal_pos'].keys()
        tx_signals = self._conn.map['spi_switch_map']['tx']['signal_pos'].keys()
        lyst = []
        if incl_rx:
            lyst.extend(rx_signals)
        if incl_tx:
            lyst.extend(tx_signals)
        return lyst

    def spi_set_switch_data(self, signal):
        """For the given named signal, sets the appropriate register with
        the appropriate data. See spi_get_all_switch_signals() for the list of
        possible signals. The switch data isn't applied until the function
        spi_apply_switch_data() is called. This function returns the name of
        the written register.
        """
        try:
            pos = self._conn.map['spi_switch_map']['rx']['signal_pos'][signal]
            reg = self._conn.map['spi_switch_map']['rx']['spi_reg']
        except KeyError:
            pos = self._conn.map['spi_switch_map']['tx']['signal_pos'][signal]
            reg = self._conn.map['spi_switch_map']['tx']['spi_reg']
        mask = 1 << pos
        self.set_fields('spi', **{reg: mask})
        return reg

    def spi_apply_switch_data(self):
        """Applies the switch settings that were previously configured with
        spi_set_switch_data(). This command uses a JUMBO SPI command
        so that RSP FIFOs are not activiated and switch settings for RX
        and TX can be different. This function checks the switch settings
        before application and raises a RuntimeError if more than one switch
        is activated in each channel.
        """
        regs = self.get_fields('spi', *('rx_switch', 'tx_switch'))
        for name, value in regs.items():
            # Allow powers of two or zero
            if value is None or bin(value).count('1') > 1:
                msg = 'Attempted to apply invalid switch value 0x{:04x} to ' \
                      'register {}.'
                raise RuntimeError(msg.format(value, name))
        n_bits = 16
        self.spi_send_cmd('jumbo', 'switch', n_bits // 16 - 1)

    def spi_write_pot_ito_wiper_register(self,
                                         wiper: str,
                                         data: int):
        """
        Writes wiper register values [0, 255] to either wiper A or B of the ITO digipot

        Args:
            wiper: 'a' or 'b'
            data: integer value [0, 255] specifying digipot wiper position

        Returns:

        """
        self.spi_write_digipot('pot_ito', wiper, data)

    def spi_write_pot_tx_wiper_register(self,
                                        wiper: str,
                                        data: int):
        """
        Writes wiper register values [0, 255] to either wiper A or B of the TX Laser digipot

        Args:
            wiper: 'a' or 'b'
            data: integer value [0, 255] specifying digipot wiper position

        Returns:

        """
        self.spi_write_digipot('pot_tx', wiper, data)

    def spi_write_digipot(self, name, wiper, data):
        if name != 'pot_ito' and name != 'pot_tx':
            raise RuntimeError(f'Illegal name {name}.')

        if wiper.lower() == 'a':
            address = 1
        elif wiper.lower() == 'b':
            address = 2
        else:
            raise RuntimeError('Wiper value must be either "a" or "b".')

        if data < 0 or data > 255:
            raise RuntimeError(f'Data value {data} is outside limits.')

        code = 0  # writing to volatile memory
        payload = code << 12 | address << 8 | data

        self.spi_send_cmd('standard', name, payload)
        self.spi_read_rsp(0)
        self.spi_read_rsp(1)

    @property
    def number_of_outliers_tx(self):
        return int(sum(self.outlier_list_tx < 1))

    @property
    def number_of_outliers_rx(self):
        return int(sum(self.outlier_list_rx < 1))

    @property
    def ito_async(self):
        """
        Get / set whether the ITO is synchronous or asynchronous to TP1
        Args:
            mode: 0 is synchronous, 1 is asynchronous

        Returns:
            0 for synchronous, 1 for asynchronous
        """
        return self.get_fields('lcm', 'ito_async')['ito_async']

    @ito_async.setter
    def ito_async(self,
                  mode: int = 1):
        if self.get_fields('lcm', 'tcon_enable')['tcon_enable'] == 1:
            print('cannot change ITO mode while TCON is enabled')
        else:
            self.set_fields('lcm', ito_async=mode)

    @property
    def ito_frequency_hz(self):
        """
        Get / set the ITO frequency, multiple options depending on the ITO mode.

        In sync mode, the frequency must be less than or equal to that of the Himax (POL) frequency as well as an
        integer factor of the Himax (POL) frequency

        In async mode, the frequency can take on any value

        Args:
            frequency_hz: desired ITO frequency in hertz

        Returns:
            ITO frequency in hertz
        """
        ito_tc = self.get_fields('lcm', 'ito_tc')['ito_tc']
        if self.ito_async == 1:
            ito_frequency_hz = 500 * 1000 / (ito_tc + 1) * self.clk_freq_mhz
        else:
            ito_frequency_hz = self.himax_frequency_hz / (ito_tc + 1)
        return ito_frequency_hz

    @ito_frequency_hz.setter
    def ito_frequency_hz(self,
                         frequency_hz: float = 2000):
        if self.ito_async == 1:
            ito_tc = self.ito_tc_async_from_khz(f_khz=frequency_hz / 1000)
            self.set_fields('lcm', ito_tc=ito_tc)
        else:
            x = self.himax_frequency_hz / frequency_hz
            if x.is_integer():
                self.set_fields('lcm', ito_tc=int(x - 1))
            else:
                print('in sync mode, ito frequency must be integer factor of himax frequency')

    @property
    def ito_amplitude_vpp(self):
        """
        Get / set the amplitude of the ITO electrode
        Args:
            amplitude_vpp: amplitude of the ITO electrode voltage
        """
        if self._ito_amplitude_vpp is None:
            print('ITO amplitude has not yet been set')
        return self._ito_amplitude_vpp

    @ito_amplitude_vpp.setter
    def ito_amplitude_vpp(self,
                          amplitude_vpp: float = 50):

        if amplitude_vpp > 3.3 * 16.5 or amplitude_vpp < 0:
            raise RuntimeError('Requested ITO voltage exceeds limits')

        data = min(255, round(amplitude_vpp / (3.3 * 16.5) * 256))
        self.spi_write_pot_ito_wiper_register(wiper='a', data=data)
        self._ito_amplitude_vpp = round((3.3 * 16.5) * (data / 256), 1)

    @property
    def ito_slew(self):
        if self._ito_slew is None:
            print('ITO slew has not yet been set')
        return self._ito_slew

    @ito_slew.setter
    def ito_slew(self,
                 slew_code: int = 0xff):
        if not 0 <= slew_code <= 0xff:
            msg = f'Requested slew code {slew_code} exceeds limits.'
            raise RuntimeError(msg)
        self.spi_write_pot_ito_wiper_register(wiper='b', data=slew_code)
        self._ito_slew = slew_code

    @property
    def tx_ci_a(self):
        if self._tx_ci_a is None:
            print('TX current limit A has not yet been set')
        return self._tx_ci_a

    @tx_ci_a.setter
    def tx_ci_a(self, ci):
        if not 0 <= ci <= 5:
            msg = f'Requested current limit {ci} exceeds limits.'
            raise RuntimeError(msg)
        data = min(255, round(ci / 5 * 256))
        self.spi_write_pot_tx_wiper_register(wiper='a', data=data)
        self._tx_ci_a = data / 256 * 5

    @property
    def tx_ci_b(self):
        if self._tx_ci_b is None:
            print('TX current limit B has not yet been set')
        return self._tx_ci_b

    @tx_ci_b.setter
    def tx_ci_b(self, ci):
        if not 0 <= ci <= 5:
            msg = f'Requested current limit {ci} exceeds limits.'
            raise RuntimeError(msg)
        data = min(255, round(ci / 5 * 256))
        self.spi_write_pot_tx_wiper_register(wiper='b', data=data)
        self._tx_ci_b = data / 256 * 5

    @property
    def himax_frequency_hz(self):
        """
        Get / set the frequency of the Himax LCD driver
        Args:
            frequency_hz: frequency of the polarity flip
        """
        tp1_period = self.get_fields('lcm', 'tp1_period')['tp1_period']
        time_us = (tp1_period + 4) / self.clk_freq_mhz
        himax_frequency_hz = 1 / (2 * time_us) * 1e6
        return himax_frequency_hz

    @himax_frequency_hz.setter
    def himax_frequency_hz(self,
                           frequency_hz: float = 2000):
        tp1_period = self.tp1_period_from_ac_freq_hz(f_hz=frequency_hz)
        self.set_fields('lcm', tp1_period=tp1_period)

    @property
    def prog_trigger_mode(self):
        """
        Get / set reprogram pattern trigger signal mode
        Args:
            mode: 0 toggles between high and low when pattern is rewritten, 1 sends pulse every time pattern is
            rewritten
        """
        return self.get_fields('lcm', 'prog_trigger_mode')['prog_trigger_mode']

    @prog_trigger_mode.setter
    def prog_trigger_mode(self,
                          mode: int = 1):
        self.set_fields('lcm', prog_trigger_mode=mode)

    def voltage_on(self):
        """Turns on the Himax driver voltage output for both Tx and Rx as well as ITO"""
        self.enable_tcon()
        self.set_fields('lcm', apply0=1)

    def voltage_off(self):
        """Turns off the Himax driver voltage output for both Tx and Rx as well as ITO"""
        self.disable_tcon()

    def shutdown(self):
        """Shutdown function to match our call from Lotus board. Currently just calls disconnect function but other
        shutdown procedures could be added here"""
        self.disconnect()

    def lcm_standby_mode(self):
        """Puts the LCM(s) in a safer standby mode by writing the pattern of all v_gnd"""
        standby_pattern_v = np.ones(self.channel_count)*self.v_gnd
        self.write_pattern_v(v_pattern=standby_pattern_v, tx_or_rx='tx')
        self.write_pattern_v(v_pattern=standby_pattern_v, tx_or_rx='rx')

    def read_pattern_delv(self,
                          tx_or_rx: str) -> np.ndarray:
        """
        Reads the current delv pattern from the Himax driver

        Args:
            tx_or_rx: specifies whether to read the tx or rx delv pattern

        Returns:
            delv (numpy): array of delv values
        """
        v = self.read_pattern_v(tx_or_rx=tx_or_rx)
        delv = self.vpt.vtodelv(v)
        return delv

    def read_pattern_v(self,
                       tx_or_rx: str) -> np.ndarray:
        """
        Reads the current voltage pattern of the Himax driver

        Args:
            tx_or_rx: specifies whether to read the tx or rx v pattern

        Returns:
            v (numpy): array of v values

        """

        if tx_or_rx == 'tx':
            table = self.get_tx_table(idx=self.scratch_table_idx)
        elif tx_or_rx == 'rx':
            table = self.get_rx_table(idx=self.scratch_table_idx)
        else:
            raise RuntimeError(f'Invalid tx_or_rx selection: {tx_or_rx}')

        # codes is ordered according to LCM rail so use sort_pos=0
        # high_vec should have the same ordering as codes
        codes = [table[t] for r, d, t, h in self.iter_lcm_map(sort_pos=0)]
        high_vec = [h for r, d, t, h in self.iter_lcm_map(sort_pos=0)]
        v = self.h.get_voltages(codes=codes, high_vec=high_vec)
        v = np.asarray(v)
        return v

    def write_pattern_v(self,
                        v_pattern: np.ndarray,
                        tx_or_rx: str):
        """
        Writes a v pattern to the Himax driver

        Args:
            v_pattern (numpy): array of voltage values
            tx_or_rx: specifies whether to write the tx or rx v pattern
        """

        v = copy.deepcopy(v_pattern)
        if np.size(v) != self.channel_count:
            raise RuntimeError("Wrong number of voltages: {}.".format(np.size(v)))
        if np.any(np.greater(v, self.v_max)):
            raise InvalidVoltageError('An input voltage is greater than allowed range!')
        if np.any(np.less(v, self.v_min)):
            raise InvalidVoltageError('An input voltage is less than allowed range!')

        # get the voltage codes and the applied v and delv patterns
        codes = self.h.get_codes(v)
        high_vec = [h for r, d, t, h in self.iter_lcm_map(sort_pos=0)]
        applied_v = np.asarray(self.h.get_voltages(codes=codes,
                                                   high_vec=high_vec))
        print(applied_v)
        applied_delv = self.vpt.vtodelv(applied_v)

        table = [self.h.MIN_CODE] * self.BYTES_PER_TABLE
        for r, d, t, h in self.iter_lcm_map():
            table[t] = codes[r - 1]

        # Test that delv pattern is still within limits even after digitization
        if np.any(np.less(applied_delv, 0)):
            raise InvalidVoltageError('Attempted to apply a negative delV')
        if np.any(np.greater(applied_delv, self.delv_max)):
            raise InvalidVoltageError(f'Attempted to apply a delV greater than allowed, delv_max = {self.delv_max}')

        # Write pattern to correct Tx or Rx location
        if tx_or_rx == 'tx':
            self.do_set_transfer_apply(idx=self.scratch_table_idx, table_tx=table)
        elif tx_or_rx == 'rx':
            self.do_set_transfer_apply(idx=self.scratch_table_idx, table_rx=table)
        else:
            raise RuntimeError('Specify either "tx" or "rx" for "tx_or_rx"')

    def plot_current_patterns(self,
                              tx_or_rx: str):
        """Plots the current voltage and delv patterns stored in Himax driver

        Args:
            tx_or_rx: specifies whether to plot the tx or rx patterns
        """
        v = self.read_pattern_v(tx_or_rx=tx_or_rx)
        delv = self.read_pattern_delv(tx_or_rx=tx_or_rx)
        plt.close('all')
        plt.subplot(211)
        plt.plot(np.arange(1, self.channel_count + 1, 1), delv, 'ob', markersize=2)
        plt.xlabel('LCM Delta V Number [1-1021]')
        plt.ylabel('Delta V')
        plt.title(f'Voltage and DelV Patterns for {tx_or_rx} LCM')
        plt.xlim([-10, 1030])
        plt.subplot(212)
        plt.plot(np.arange(1, self.channel_count + 1, 1), v, '-or', markersize=2)
        plt.axhline(y=self.v_max / 2, color='k', linestyle='--')
        plt.xlabel('LCM Rail Number [1-1021]')
        plt.ylabel('V')
        plt.xlim([-10, 1030])
        plt.ylim([self.v_min, self.v_max])
        plt.tight_layout()

    def get_n_wait_cycles(self,
                          desired_sampling_interval_ns: float):
        """
        Returns the number of wait cycles needed to achieve a specified time
        interval between ADC sample points.

        Args:
            desired_sampling_interval_ns: desired time (in ns) between ADC
                sampling points

        Returns:
            n_wait_cycles: number of wait cycles to pass to the ADC to get
                desired sampling interval time
            actual_sampling_interval_ns: the actual time (in ns) between
                ADC sampling points
        """
        # returns the adc clk divisions, typically 2
        clk_div_adc = self.get_fields('spi', 'clk_div_adc')['clk_div_adc']
        frame_time_cycles = (16 + 1) * (clk_div_adc + 1)
        frame_time_us = frame_time_cycles / self.clk_freq_mhz
        frame_spacing_us = desired_sampling_interval_ns / 1000 - frame_time_us
        frame_spacing_cycles = frame_spacing_us * self.clk_freq_mhz
        n_wait_cycles = int(frame_spacing_cycles - 4 - clk_div_adc)
        n_wait_cycles = min(2 ** 18, max(0, n_wait_cycles))

        actual_frame_spacing_cycles = 4 + clk_div_adc + n_wait_cycles
        actual_frame_spacing_us = actual_frame_spacing_cycles / self.clk_freq_mhz
        actual_sampling_interval_ns = 1000 * (frame_time_us + frame_spacing_us)
        return n_wait_cycles, actual_sampling_interval_ns

    def single_pair_outlier_check(self,
                                  tx_or_rx: str,
                                  lcm_rail_number: int,
                                  delv: float = 4,
                                  plot: bool = True,
                                  repeated_measurement_mode: bool = False,
                                  averages: int = 1):
        """
        Checks for outlier current consumption between a single pair of voltage rails on the LCM by interpreting
        the output of the ADC monitoring the 18V current sense line (Himax LCD current consumption)

        Args:
            tx_or_rx: specifies whether to measure the tx or rx LCM
            lcm_rail_number: lcm rail [1,1021] to bias for testing the himax driver current consumption
            delv: the voltage differential between lcm_rail_number and V_GND
            plot: boolean indicating whether to plot
            repeated_measurement_mode: turn on if you are doing back to back measurements and do not
                want to repeat configuring the switches and other unnecessary operations
            averages: number of repeat ADC measurements to use. Note plotting will only do last reading

        Returns:
            consumption_voltage_ma: the average current consumption (mA) in the POL state leading to the largest value

        """

        # create the v pattern required for this measurement
        v = np.ones(self.channel_count) * self.v_gnd
        v[lcm_rail_number - 1] = self.v_gnd - delv
        self.write_pattern_v(v_pattern=v,
                             tx_or_rx=tx_or_rx)
        # sleep to ensure that the voltage pattern has been written
        time.sleep(0.001)
        n_frames = 63

        # turn on the correct switches
        if not repeated_measurement_mode:
            if tx_or_rx == 'tx':
                self.spi_set_adc_ch_sel('tx')
                self.spi_set_switch_data('tx_pol')
            else:
                self.spi_set_adc_ch_sel('rx')
                self.spi_set_switch_data('rx_pol')
            self.spi_apply_switch_data()
            self.spi_clear_rsp_fifo(0)
            self.spi_clear_rsp_fifo(1)

            # adjust sampling rate according to POL frequency
            himax_polarity_period_us = 1 / self.himax_frequency_hz * 1e6  # time for one full polarity flipping cycle
            self.desired_sampling_interval_us = himax_polarity_period_us / (n_frames + 1)

        temp = []
        for i in range(averages):
            # grab data from ADC
            self.spi_read_adcs_stream(sample_interval_us=self.desired_sampling_interval_us, n_frames=n_frames - 1)

            rsps_0 = self.spi_read_rsp(channel=0, n_rsp=n_frames)
            rsps_0_valid = [r for r in rsps_0 if self.spi_get_rsp_is_valid(r)]
            rsps_0_valid.pop(0)  # discard first sample
            current_sense_data_V = [0.00025 * (r >> 2 & 0x3fff) for r in rsps_0_valid]
            current_sense_data = [i*100 for i in current_sense_data_V]  # transform from V to mA

            rsps_1 = self.spi_read_rsp(channel=1, n_rsp=n_frames)
            rsps_1_valid = [r for r in rsps_1 if self.spi_get_rsp_is_valid(r)]
            rsps_1_valid.pop(0)  # discard first sample
            pol_data = [0.00025 * (r >> 2 & 0x3fff) for r in rsps_1_valid]
            pol_average = sum(pol_data) / len(pol_data)

            current_sense_data_regime1 = [current_sense_data[i] for i, value in enumerate(pol_data) if value > pol_average]
            current_sense_data_regime2 = [current_sense_data[i] for i, value in enumerate(pol_data) if value <= pol_average]

            median_regime1 = np.median(current_sense_data_regime1)
            median_regime2 = np.median(current_sense_data_regime2)
            if median_regime2 > median_regime1:
                temp.append(median_regime2)
            else:
                temp.append(median_regime1)

            if len(current_sense_data) and len(pol_data) != n_frames - 1:
                raise RuntimeError('Length of ADC responses do not match requested frame number')

        consumption_current_mean_ma = np.mean(temp)
        consumption_current_std_dev_ma = np.std(temp)


        if plot:
            time_data_us = 0 + np.arange(n_frames - 1) * self.desired_sampling_interval_us
            time_data_regime1 = [time_data_us[i] for i, value in enumerate(pol_data) if value > pol_average]
            time_data_regime2 = [time_data_us[i] for i, value in enumerate(pol_data) if value <= pol_average]
            plt.close('all')
            ax1 = plt.subplot(211)
            if median_regime2 > median_regime1:
                plt.plot(time_data_regime1, current_sense_data_regime1, 'ok')
                plt.plot(time_data_regime2, current_sense_data_regime2, 'or')
            else:
                plt.plot(time_data_regime1, current_sense_data_regime1, 'or')
                plt.plot(time_data_regime2, current_sense_data_regime2, 'ok')
            plt.axhline(y=median_regime1, color='k', linestyle='--')
            plt.axhline(y=median_regime2, color='k', linestyle='--')
            plt.xlabel('time (us)')
            plt.ylabel('Current Consumption (mA)')
            plt.title(f'ADC Readout for {tx_or_rx} LCM')
            ax2 = plt.subplot(212)
            plt.plot(time_data_us, pol_data, '-ok')
            plt.xlabel('time (us)')
            plt.ylabel('POL signal (V)')
            plt.tight_layout()

        return consumption_current_mean_ma, consumption_current_std_dev_ma

    def full_outlier_check(self,
                           tx_or_rx='tx',
                           plot: bool = False,
                           averages: int = 1):
        """
        Runs the outlier check for every rail in the LCM and updates self.outlier_list accordingly

        Args:
            tx_or_rx: specifies whether to measure the tx or rx LCM
            plot: boolean whether to plot the data
            averages: number of repeat single_pair_outlier_checks do to on each LCM rail

        Returns:
            outlier_dataframe: summary dataframe specifying the results of each rail measurement
            fig: plot of outliers

        """
        tstart = time.time()
        self.voltage_on()

        # Check himax frequency, turn down for duration of check
        prior_himax_freq = self.himax_frequency_hz
        self.himax_frequency_hz = 2000

        # Check ito voltage, turn down for duration of check
        prior_ito_voltage = self.ito_amplitude_vpp
        self.ito_amplitude_vpp = 0

        # Clear both Tx and Rx patterns
        self.write_pattern_v(v_pattern=np.ones(self.channel_count) * self.v_gnd, tx_or_rx='tx')
        self.write_pattern_v(v_pattern=np.ones(self.channel_count) * self.v_gnd, tx_or_rx='rx')

        # Throwaway measurement to properly set switch configuration
        self.single_pair_outlier_check(tx_or_rx=tx_or_rx,
                                       lcm_rail_number=1,
                                       delv=0,
                                       plot=False,
                                       repeated_measurement_mode=False,
                                       averages=1)

        # Stabilize
        time.sleep(5)

        # Grab background (9V on every rail) values
        standby_consumption_mean_ma, standby_consumption_std_ma = self.single_pair_outlier_check(tx_or_rx=tx_or_rx,
                                                                                                 lcm_rail_number=1,
                                                                                                 delv=0,
                                                                                                 plot=False,
                                                                                                 repeated_measurement_mode=True,
                                                                                                 averages=500)
        std_dev_percent = standby_consumption_std_ma / standby_consumption_mean_ma * 100

        # Cycle through all rails
        delv = 4
        consumption_current_list = []
        print_interval = int(0.10 * self.channel_count)
        for i, r in enumerate(self.rails):
            if (VERBOSE or DEBUG) and 0 == i % print_interval:
                print('.', end='', flush=True)
            consumption_current_ma, _ = self.single_pair_outlier_check(tx_or_rx=tx_or_rx,
                                                                       lcm_rail_number=r,
                                                                       delv=delv,
                                                                       plot=False,
                                                                       repeated_measurement_mode=True,
                                                                       averages=averages)
            consumption_current_list.append(consumption_current_ma)

        z_scores = (np.asarray(consumption_current_list) - standby_consumption_mean_ma) \
                   / standby_consumption_std_ma

        # Initialize outlier list with all ones (opens)
        temp_outlier_list = np.ones(self.channel_count)
        for i in self.rails:
            if z_scores[i - 1] > self.z_score_threshold:
                temp_outlier_list[i - 1] = 0  # This is an irregular current consumption

        temp_outlier_list = temp_outlier_list.astype(int)
        number_outliers = len([i for i in z_scores if i > self.z_score_threshold])

        if tx_or_rx == 'tx':
            self.outlier_list_tx = temp_outlier_list
        elif tx_or_rx == 'rx':
            self.outlier_list_rx = temp_outlier_list
        else:
            raise RuntimeError(f'Invalid tx_or_rx selection: {tx_or_rx}')

        if plot:
            fig = plt.figure(figsize=(12, 8))
            plt.subplot(211)
            plt.plot(self.rails, consumption_current_list, 'ob', markersize=2)
            plt.xlim([-1, self.channel_count + 1])
            plt.axhline(y=standby_consumption_mean_ma, linestyle='-', color='k')
            plt.axhline(y=standby_consumption_mean_ma + standby_consumption_std_ma * self.z_score_threshold,
                        linestyle='--', color='k')
            plt.axhline(y=standby_consumption_mean_ma - standby_consumption_std_ma * self.z_score_threshold,
                        linestyle='--', color='k')
            plt.ylabel('Consumption Current (mA)')
            plt.title(f'Number of Outlier Rails: {number_outliers}    Standby Current (mA): '
                      f'{standby_consumption_mean_ma:.2f} +/- {std_dev_percent:.2f}%     '
                      f'Averages per Rail: {averages}    '
                      f'Time (s): {int(time.time()-tstart)}')
            plt.subplot(212)
            plt.plot(self.rails, z_scores, 'og', markersize=2)
            plt.xlim([-1, self.channel_count + 1])
            plt.axhline(y=0, linestyle='-', color='k')
            plt.axhline(y=self.z_score_threshold, linestyle='--', color='k')
            plt.axhline(y=-1 * self.z_score_threshold, linestyle='--', color='k')
            plt.ylabel('Z-Score')
            plt.xlabel('LCM Rail Number [1-1021]')

        v = np.ones(self.channel_count) * self.v_gnd
        self.write_pattern_v(tx_or_rx=tx_or_rx, v_pattern=v)

        if VERBOSE or DEBUG:
            print('\n')
            print(f'Standby Current (mA): {standby_consumption_mean_ma:.2f} +/- {std_dev_percent:.2f}%')
            if standby_consumption_mean_ma > 11:
                print('WARNING: This Value Exceeds Datasheet Specification')
            if tx_or_rx == 'tx':
                print(f'Number of Outliers: {self.number_of_outliers_tx}')
            elif tx_or_rx == 'rx':
                print(f'Number of Outliers: {self.number_of_outliers_rx}')
            else:
                raise RuntimeError(f'Invalid tx_or_rx selection: {tx_or_rx}')

        # Restore to prior state
        self.himax_frequency_hz = prior_himax_freq
        self.ito_amplitude_vpp = prior_ito_voltage

        # Create a dataframe to organize the resulting data
        pin1 = self.rails
        pin2 = [self.rails[i - 1] for i in range(len(self.rails))]
        standby_mean_list = [standby_consumption_mean_ma for i in self.rails]
        standby_std_list = [standby_consumption_std_ma for i in self.rails]
        z_scores_list = z_scores.tolist()
        outlier_list = temp_outlier_list.tolist()
        outlier_dataframe = pd.DataFrame(data=list(zip(pin1, pin2, standby_mean_list, standby_std_list,
                                                       consumption_current_list, z_scores_list, outlier_list)),
                                         columns=['pin1', 'pin2', 'standby_current_ma',
                                                  'standby_current_std_ma', 'consumption_current_ma',
                                                  'z_score', 'outlier_list'])

        if plot:
            return outlier_dataframe, fig
        else:
            return outlier_dataframe

    def get_i_vdda(self,
                   tx_or_rx='tx'):
        """
        Measures the current draw to the Himax driver

        Args:
            tx_or_rx: specifies whether to measure the Tx or Rx LCM module

        Returns:
            i_vdda_a: average value of i_vdda over a full pol cycle, in amps
        """

        # set number of frames to collect
        n_frames = 63

        # turn on the correct switches
        if tx_or_rx == 'tx':
            self.spi_set_adc_ch_sel('tx')
            self.spi_set_switch_data('tx_pol')
        else:
            self.spi_set_adc_ch_sel('rx')
            self.spi_set_switch_data('rx_pol')
        self.spi_apply_switch_data()
        self.spi_clear_rsp_fifo(0)
        self.spi_clear_rsp_fifo(1)

        # adjust sampling rate according to POL frequency
        himax_polarity_period_us = 1 / self.himax_frequency_hz * 1e6  # time for one full polarity flipping cycle
        self.desired_sampling_interval_us = himax_polarity_period_us / (n_frames + 1)

        # grab data from ADC
        self.spi_read_adcs_stream(sample_interval_us=self.desired_sampling_interval_us, n_frames=n_frames - 1)
        rsps_0 = self.spi_read_rsp(channel=0, n_rsp=n_frames)
        rsps_0_valid = [r for r in rsps_0 if self.spi_get_rsp_is_valid(r)]
        rsps_0_valid.pop(0)  # discard first sample
        current_sense_data = [0.00025 * (r >> 2 & 0x3fff) for r in rsps_0_valid]

        # average and convert from volts to amps, relative to 0.1 ohm resistor
        i_vdda_a = np.asscalar(10 * np.average(current_sense_data) / 100)

        return i_vdda_a

    def measure_operation_current(self,
                                  tx_or_rx='tx',
                                  delv=2):
        """
        Applies a standard voltage pattern and measures the current consumption of the HX8175 driver.

        Returns:
            operation_current_ma: 100 averages of the current consumption measurement, in mA
        """
        # Check himax frequency, turn down for duration of check
        prior_himax_freq = self.himax_frequency_hz
        self.himax_frequency_hz = 2000

        # Check ito voltage, turn down for duration of check
        prior_ito_voltage = self.ito_amplitude_vpp
        self.ito_amplitude_vpp = 0

        # use standard delv = 4 everywhere pattern
        v = np.ones(self.channel_count) * self.v_gnd
        if self.cof:
            for k, value in enumerate(v):
                if k % 2 == 1:
                    v[k] = self.v_gnd - delv
        else:
            for k, value in enumerate(v):
                if k % 2 == 0:
                    v[k] = self.v_gnd - delv

        self.write_pattern_v(v_pattern=v, tx_or_rx=tx_or_rx)

        # Stabilize
        time.sleep(5)

        # Take 100 averages
        temp = []
        for i in range(100):
            temp.append(self.get_i_vdda(tx_or_rx=tx_or_rx))
        operation_current_ma = float(np.mean(temp)*1000)

        # Return to prior state
        self.himax_frequency_hz = prior_himax_freq
        self.ito_amplitude_vpp = prior_ito_voltage
        self.write_pattern_v(v_pattern=np.ones(self.channel_count) * self.v_gnd, tx_or_rx=tx_or_rx)

        print(f'Operation Current (mA): {operation_current_ma:.2f}')

        return operation_current_ma

    def measure_standby_current(self,
                                tx_or_rx='tx'):
        """
        Measures the current consumption of the HX8175 driver with no voltages applied (including ITO).

        Returns:
            standby_current_ma: 100 averages of the current consumption measurement, in mA
        """
        # Check himax frequency, turn down for duration of check
        prior_himax_freq = self.himax_frequency_hz
        self.himax_frequency_hz = 2000

        # Check ito voltage, turn down for duration of check
        prior_ito_voltage = self.ito_amplitude_vpp
        self.ito_amplitude_vpp = 0

        # Write all 9V pattern
        v = np.ones(self.channel_count) * self.v_gnd
        self.write_pattern_v(v_pattern=v, tx_or_rx=tx_or_rx)

        # Stabilize
        time.sleep(5)

        # Take 100 averages
        temp = []
        for i in range(100):
            temp.append(self.get_i_vdda(tx_or_rx=tx_or_rx))
        standby_current_ma = float(np.mean(temp)*1000)

        # Return to prior state
        self.himax_frequency_hz = prior_himax_freq
        self.ito_amplitude_vpp = prior_ito_voltage

        print(f'Standby Current (mA): {standby_current_ma:.2f}')
        if standby_current_ma > 11:
            print('WARNING: This Value Exceeds Datasheet Specification')

        return standby_current_ma

    def plot_current_sense(self,
                           tx_or_rx):
        n_frames = 63

        if tx_or_rx == 'tx':
            self.spi_set_adc_ch_sel('tx')
            self.spi_set_switch_data('tx_pol')
        else:
            self.spi_set_adc_ch_sel('rx')
            self.spi_set_switch_data('rx_pol')
        self.spi_apply_switch_data()
        self.spi_clear_rsp_fifo(0)
        self.spi_clear_rsp_fifo(1)

        # adjust sampling rate according to POL frequency
        himax_polarity_period_us = 1 / self.himax_frequency_hz * 1e6  # time for one full polarity flipping cycle
        self.desired_sampling_interval_us = himax_polarity_period_us / (n_frames + 1)

        # grab data from ADC
        self.spi_read_adcs_stream(sample_interval_us=self.desired_sampling_interval_us, n_frames=n_frames - 1)

        rsps_0 = self.spi_read_rsp(channel=0, n_rsp=n_frames)
        rsps_0_valid = [r for r in rsps_0 if self.spi_get_rsp_is_valid(r)]
        rsps_0_valid.pop(0)  # discard first sample
        current_sense_data = [0.00025 * (r >> 2 & 0x3fff) for r in rsps_0_valid]

        rsps_1 = self.spi_read_rsp(channel=1, n_rsp=n_frames)
        rsps_1_valid = [r for r in rsps_1 if self.spi_get_rsp_is_valid(r)]
        rsps_1_valid.pop(0)  # discard first sample
        pol_data = [0.00025 * (r >> 2 & 0x3fff) for r in rsps_1_valid]
        pol_average = sum(pol_data) / len(pol_data)

        if len(current_sense_data) and len(pol_data) != n_frames - 1:
            raise RuntimeError('Length of ADC responses do not match requested frame number')

        time_data_us = 0 + np.arange(n_frames - 1) * self.desired_sampling_interval_us
        plt.close('all')
        ax1 = plt.subplot(211)
        plt.plot(time_data_us, np.asarray(current_sense_data)/10*1000, '-ob')
        plt.xlabel('time (us)')
        plt.ylabel('Current Consumption (mA)')
        plt.title(f'ADC Readout for {tx_or_rx} LCM')
        ax2 = plt.subplot(212)
        plt.plot(time_data_us, pol_data, '-ok')
        plt.xlabel('time (us)')
        plt.ylabel('POL signal (V)')
        plt.tight_layout()

    def read_temp1_temp5(self):
        """Returns TEMP1-TEMP5 thermistor value in Ohms"""

        # Check himax frequency, turn down for duration of check
        prior_himax_freq = self.himax_frequency_hz
        self.himax_frequency_hz = 2000

        # Check ito voltage, turn down for duration of check
        prior_ito_voltage = self.ito_amplitude_vpp
        self.ito_amplitude_vpp = 0

        # Stabilize
        time.sleep(5)

        # Configure switches
        self.spi_set_adc_ch_sel('tx')
        self.spi_set_switch_data('tx_rtd_temp')
        self.spi_apply_switch_data()
        self.spi_clear_rsp_fifo(0)
        self.spi_clear_rsp_fifo(1)

        # Grab data from ADC
        n_frames = 63
        himax_polarity_period_us = 1 / self.himax_frequency_hz * 1e6  # time for one full polarity flipping cycle
        self.desired_sampling_interval_us = himax_polarity_period_us / (n_frames + 1)
        self.spi_read_adcs_stream(sample_interval_us=self.desired_sampling_interval_us, n_frames=n_frames - 1)

        rsps_1 = self.spi_read_rsp(channel=1, n_rsp=n_frames)
        rsps_1_valid = [r for r in rsps_1 if self.spi_get_rsp_is_valid(r)]
        rsps_1_valid.pop(0)  # discard first sample
        temp1_temp5_adc_volts = [0.00025 * (r >> 2 & 0x3fff) for r in rsps_1_valid]
        temp1_temp5_adc_volts = np.mean(temp1_temp5_adc_volts)
        temp1_temp5_rtd_volts = 5 * temp1_temp5_adc_volts
        temp1_temp5_rtd_amps = 100e-6
        temp1_temp5_rtd_ohms = temp1_temp5_rtd_volts / temp1_temp5_rtd_amps

        # Return to prior state
        self.himax_frequency_hz = prior_himax_freq
        self.ito_amplitude_vpp = prior_ito_voltage

        print(f'TEMP1-TEMP5 (Ohm): {int(temp1_temp5_rtd_ohms)}')

        return float(temp1_temp5_rtd_ohms)

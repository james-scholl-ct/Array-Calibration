from functools import reduce
import math
import os
import random
from scp import SCPClient
import sys
import time

from python_tools.zynq_connection import ZynqConnection


class ZynqAPI:
    def __init__(self, host, design, mem_depth):
        self.host = host
        self.design = design
        self.mem_depth = mem_depth
        self._zconn = ZynqConnection(host)
        #if self._zconn.get_bitstream_program_status():
        #    self._bitstream_programmed_this_time = False
        #else:
        #    self._zconn.program_bitstream(self.bitstream_path)
        #    self._bitstream_programmed_this_time = True
        self._zconn.program_bitstream(self.bitstream_path)
        self._bitstream_programmed_this_time = True

    @property
    def cmd_file_path(self):
        return '/root/{0}/cmd_file.txt'.format(self.design)

    @property
    def rsp_file_path(self):
        return '/root/{0}/rsp_file.txt'.format(self.design)

    @property
    def log_file_path(self):
        return '/root/{0}/{0}_app.log'.format(self.design)

    @property
    def remote_app_path(self):
        return '/root/{0}/{0}_app.elf'.format(self.design)

    @property
    def bitstream_path(self):
        return '/root/{0}/{0}.bit'.format(self.design)

    def get_connection(self):
        return self._zconn

    def close_connection(self):
        self._zconn.close()

    def remote_app_is_running(self):
        bash = 'pgrep -f "{}"'.format(self.remote_app_path)
        out = self._zconn.exec_command(bash)
        return out != []

    def kill_remote_app(self):
        bash = 'killall "{}"'.format(self.remote_app_path)
        out = self._zconn.exec_command(bash)
        return out != []

    def start_remote_app(self, overwrite_cmd_file=True, force_restart=False):
        if self.remote_app_is_running():
            if force_restart:
                self.kill_remote_app()
                print('Remote app is already running. Forcing restart!')
            else:
                raise RuntimeError("Remote app is already running.")
        if self._cmd_file_exists() and not overwrite_cmd_file:
            msg = 'Remote file exists: {}.'.format(self.cmd_file_path)
            raise RuntimeError(msg)
        self._clear_cmd_file()
        self._cmd_queue = []
        self._rand = random.randint(0, 2**16-1)
        bash = '{} {} {} 2>~root/lotus/stderr.tmp'.format(self.remote_app_path,
                                 self.cmd_file_path,
                                 self.rsp_file_path)
        self._zconn.exec_command(bash, background=True)
        if self._bitstream_programmed_this_time:
            self.reset()
        time.sleep(1)
        self._zconn.ds_open()

    def stop_remote_app(self, tries=3, sleep=1):
        if not self.remote_app_is_running():
            raise RuntimeError("Remote app is not running.")
        cmd = 'exit'
        rsps = self._send_cmds([cmd])
        if cmd not in rsps[0]:
            print("ERROR: bad response for cmd '{}': {}.".format(cmd, rsps[0]))
            return False
        else:
            for i in range(tries):
                if not self.remote_app_is_running():
                    return True
                time.sleep(sleep)
            msg = "Couldn't stop remote app. You will need to ssh into the "\
                  "Zynq and run `killall {}` to manually kill it."
            msg = msg.format(os.path.basename(self.remote_app_path))
            raise RuntimeError(msg)

    def reset(self):
        # This is needed for testing purposes
        pass

    def shutdown(self):
        self.stop_remote_app()
        self.close_connection()

    # -------------------------------------------------------------------------
    # File access                                                     (PRIVATE)
    # -------------------------------------------------------------------------
    def _cmd_file_exists(self):
        bash = 'if [ -f {} ]; then echo 1; else echo 0; fi'
        bash = bash.format(self.cmd_file_path)
        out = self._zconn.exec_command(bash)
        return out[0] == '1'

    def _clear_cmd_file(self):
        bash = 'rm -f {0}; touch {0}'.format(self.cmd_file_path)
        self._zconn.exec_command(bash)

    def _read_cmd_file(self):
        bash = 'cat {}'.format(self.cmd_file_path)
        return self._zconn.exec_command(bash)

    def _read_rsp_file(self):
        bash = 'cat {}'.format(self.rsp_file_path)
        return self._zconn.exec_command(bash)

    def _read_log_file(self):
        bash = 'cat {}'.format(self.log_file_path)
        return self._zconn.exec_command(bash)

    # -------------------------------------------------------------------------
    # Sending commands                                                (PRIVATE)
    # -------------------------------------------------------------------------
    def _get_rand(self):
        v = self._rand
        self._rand = (self._rand + 1) & 0xffff
        return v

    def _send_cmds(self, cmds, tries=3, sleep=1):
        cmd_strs = ["{} {}".format(self._get_rand(), cmd) for cmd in cmds]
        bash = 'echo "{}" >> {}'.format("\n".join(cmd_strs), self.cmd_file_path)
        self._zconn.exec_command(bash)
        bash = 'tail -n {} {}'.format(len(cmds), self.rsp_file_path)
        for i in range(tries):
            out = self._zconn.exec_command(bash)
            first_header_matches = out[0].split()[0] == cmd_strs[0].split()[0]
            if len(out) == len(cmds) and first_header_matches:
                return out
            time.sleep(sleep)
        print("ERROR: unable to get response for cmds '{}'.".format(cmds))
        return []

    def _send_cmd_queue(self):
        rsps = self._send_cmds(self._cmd_queue)
        if any(cmd not in rsp for cmd, rsp in zip(self._cmd_queue, rsps)):
            print("ERROR: bad response for cmd '{}': {}.".format(cmd, rsp))
            ret_val = None
        else:
            ret_val = [int(rsp.split()[-1], 16) for rsp in rsps]
        self._cmd_queue = []
        return ret_val

    def _send_read(self, addr, queue=False, periph='bsc'):
        if not 0 <= addr < self.mem_depth[periph]:
            raise RuntimeError("Address out of range ({}).".format(addr))
        cmd = 'read_{} 0x{:x}'.format(periph, addr)
        if queue:
            self._cmd_queue.append(cmd)
        else:
            rsps = self._send_cmds([cmd])
            if cmd not in rsps[0]:
                msg = "ERROR: bad response for cmd '{}': {}."
                msg = msg.format(cmd, rsps[0])
                print(msg)
                return None
            else:
                return int(rsps[0].split()[-1], 16)

    def _send_write(self, addr, data, queue=False, periph='bsc'):
        if not 0 <= addr < self.mem_depth[periph]:
            raise RuntimeError("Address out of range ({}).".format(addr))
        if not 0 <= data < 2**32:
            raise RuntimeError("Data out of range (0x{:x}).".format(data))
        cmd = 'write_{} 0x{:x} 0x{:x}'.format(periph, addr, data)
        if queue:
            self._cmd_queue.append(cmd)
        else:
            rsps = self._send_cmds([cmd])
            if cmd not in rsps[0]:
                msg = "ERROR: bad response for cmd '{}': {}."
                msg = msg.format(cmd, rsps[0])
                print(msg)
                return None
            else:
                return True

    def _send_read_range(self, lo, n_words, periph='bsc'):
        hi = lo + n_words
        depth = self.mem_depth[periph]
        if not 0 <= lo < depth and not 0 < hi <= depth:
            msg = "Invalid range [{}, {}).".format(lo, hi)
            raise RuntimeError(msg)
        for addr in range(lo, hi):
            self._send_read(addr, queue=True, periph=periph)
        return self._send_cmd_queue()

    def _send_write_range(self, lo, data, periph='bsc'):
        hi = lo + len(data)
        depth = self.mem_depth[periph]
        if not 0 <= lo < depth and not 0 < hi <= depth:
            msg = "Invalid range [{}, {}).".format(lo, hi)
            raise RuntimeError(msg)
        for addr, datum in zip(range(lo, hi), data):
            self._send_write(addr, datum, queue=True, periph=periph)
        self._send_cmd_queue()

    def _send_custom(self, cmd,
                           idx_data,
                           postproc=lambda x: x,
                           tries=3,
                           sleep=1):
        rsps = self._send_cmds([cmd], tries=tries, sleep=sleep)
        if cmd not in rsps[0]:
            msg = "ERROR: bad response for cmd '{}': {}."
            msg = msg.format(cmd, rsps[0])
            print(msg)
            return None
        else:
            return [postproc(i) for i in rsps[0].split()[idx_data:]]

    # -------------------------------------------------------------------------
    # API: memory helpers                                              (Public)
    # -------------------------------------------------------------------------
    def fold_array(self, arr, width):
        depth = (len(arr) - 1) // width + 1
        return [arr[width*i : width*(i+1)] for i in range(depth)]

    def unfold_array(self, arr):
        return [e for suba in arr for e in suba]

    def byte_array_to_int(self, arr):
        for e in arr:
            if not 0x00 <= e <= 0xff:
                msg = "Value exceeds limits of type byte: {}.".format(e)
                raise RuntimeError(msg)
        return reduce((lambda x, y: x | y[1] << 8*y[0]), enumerate(arr), 0)

    def int_to_byte_array(self, v, n_bits):
        # n_bits is necessary because we need to know quantity of leading zeros
        if not 0 <= v < 2**n_bits:
            msg = "Value ({}) must be in [0, 2**{}).".format(v, n_bits)
            raise RuntimeError(msg)
        n_bytes = max(1, math.ceil(n_bits/8))
        return [v >> 8*i & 0xff for i in range(n_bytes)]

    def byte_array_to_word_array(self, arr):
        folded = self.fold_array(arr, 4)
        return [self.byte_array_to_int(ba) for ba in folded]

    def word_array_to_byte_array(self, arr):
        folded = [self.int_to_byte_array(word, 32) for word in arr]
        return self.unfold_array(folded)


class OrchidZynqAPI(ZynqAPI):
    def __init__(self, host):
        super(OrchidZynqAPI, self).__init__(host, 'orchid', {'bsc': 72})

    def shutdown_sequence(self):
        self.stop()
        super(OrchidZynqAPI, self).shutdown()

    # -------------------------------------------------------------------------
    # API: control and config                                          (Public)
    # -------------------------------------------------------------------------
    def reset(self):
        self._send_write(71, 1)

    def start(self, mode=0):
        if mode not in range(4):
            raise RuntimeError("Invalid mode value ({}).".format(mode))
        self._send_write(71, mode << 2 | 1 << 1)

    def stop(self):
        self._send_write(71, 0)

    def set_dwell_normal(self, val):
        if not 0 <= val < 2**20:
            msg = "Dwell value ({}) must be on [0, 2**20).".format(val)
            raise RuntimeError(msg)
        self._send_write(69, val)

    def set_dwell_mode00(self, val):
        if not 0 <= val < 2**20:
            msg = "Dwell value ({}) must be on [0, 2**20).".format(val)
            raise RuntimeError(msg)
        self._send_write(68, val)

    def dwell_count_from_ns(self, time_ns):
        # 120MHz clock, 60 counts for TP1 state, 2 counts for POL state, and
        # 1 count for pipeline in dwell_expired.
        return int(time_ns / 1e9 * 120e6 - (60 + 2 + 1))

    def dwell_count_to_ns(self, dwell_count):
        # 120MHz clock, 60 counts for TP1 state, 2 counts for POL state, and
        # 1 count for pipeline in dwell_expired.
        return (dwell_count + 63) * 1e9 / 120e6

    # -------------------------------------------------------------------------
    # API: memory access                                               (Public)
    # -------------------------------------------------------------------------
    def get_channel_byte_addr(self, table_idx, channel_idx):
        return table_idx << 6 | channel_idx

    def get_channel(self, table_idx, channel_idx):
        if table_idx not in range(4):
            msg = "Invalid table_idx value ({}).".format(table_idx)
            raise RuntimeError(msg)
        if not 0 <= channel_idx < 64:
            msg = "Invalid channel_idx value ({}).".format(channel_idx)
            raise RuntimeError(msg)
        byte_addr = self.get_channel_byte_addr(table_idx, channel_idx)
        word_addr = byte_addr >> 2
        byte_sel = byte_addr & 0x3
        data = self._send_read(word_addr)
        return data >> 8*byte_sel & 0xff

    def set_channel(self, table_idx, channel_idx, coeff):
        if table_idx not in range(4):
            msg = "Invalid table_idx value ({}).".format(table_idx)
            raise RuntimeError(msg)
        if not 0 <= channel_idx < 64:
            msg = "Invalid channel_idx value ({}).".format(channel_idx)
            raise RuntimeError(msg)
        if not 0 <= coeff < 2**8:
            msg = "Invalid coeff value ({}).".format(coeff)
            raise RuntimeError(msg)
        byte_addr = self.get_channel_byte_addr(table_idx, channel_idx)
        word_addr = byte_addr >> 2
        byte_sel = byte_addr & 0x3
        data = self._send_read(word_addr)
        data &= ~(0xff << 8*byte_sel)
        data |= (coeff << 8*byte_sel)
        self._send_write(word_addr, data)

    def get_table(self, table_idx):
        if table_idx not in range(4):
            msg = "Invalid table_idx value ({}).".format(table_idx)
            raise RuntimeError(msg)
        start_word_addr = self.get_channel_byte_addr(table_idx, 0) >> 2
        words = self._send_read_range(start_word_addr, 64//4)
        return self.word_array_to_byte_array(words)

    def set_table(self, table_idx, coeffs):
        if table_idx not in range(4):
            msg = "Invalid table_idx value ({}).".format(table_idx)
            raise RuntimeError(msg)
        if len(coeffs) != 64:
            msg = "List has {} coeffs but 64 are expected.".format(len(coeffs))
            raise RuntimeError(msg)
        for i, coeff in enumerate(coeffs):
            if not 0 <= coeff < 2**8:
                msg = "Invalid coeff value {} for channel {}.".format(coeff, i)
                raise RuntimeError(msg)
        start_word_addr = self.get_channel_byte_addr(table_idx, 0) >> 2
        words = self.byte_array_to_word_array(coeffs)
        self._send_write_range(start_word_addr, words)


class LotusZynqAPI(ZynqAPI):
    def __init__(self, host):
        mem_depth = {'bsc': 64, 'spi': 32}
        super(LotusZynqAPI, self).__init__(host, 'lotus', mem_depth)
        self.pol_ovr = 0
        self.remote_csv_file = None

    def shutdown_sequence(self):
        if self.remote_csv_file is not None:
            bash = 'rm -f {}'.format(self.remote_csv_file)
            self._zconn.exec_command(bash)
        self.stop_laser()
        self.stop()
        self.spi_set_config(0)
        super(LotusZynqAPI, self).shutdown()

    # -------------------------------------------------------------------------
    # BSC API: control and config                                      (Public)
    # -------------------------------------------------------------------------
    def reset(self):
        self._send_write(62, 1<<0 | self.pol_ovr<<4)

    def init(self):
        self._send_write(62, 1<<1 | self.pol_ovr<<4)

    def enable(self, apply=1):
        self._send_write(62, 1<<2 | apply<<3 | self.pol_ovr<<4)

    def apply_table(self):
        self._send_write(62, 1<<2 | 1<<3 | self.pol_ovr<<4)

    def stop(self):
        self._send_write(62, 0 | self.pol_ovr<<4)

    def load_table(self, table_index):
        if not 0 <= table_index < 128:
            msg = "Table index ({}) must be on [0, 128).".format(table_index)
            raise RuntimeError(msg)
        self._send_write(60, table_index)

    def ds_load_apply(self, table_index, check=False):
        #t1 = time.time()
        if not 0 <= table_index < 128:
            msg = "Table index ({}) must be on [0, 128).".format(table_index)
            raise RuntimeError(msg)

        table_bytes = bytes([table_index])
        # Command byte 0x02 sets table with following 204 bytes, then applies
        self._zconn._ds.sendall(b'\x04' + table_bytes)

    def config_laser(self, clks_per_interval,
                           pulses_per_frame,
                           intervals_per_frame,
                           check=False):
        if not 0 <= clks_per_interval < 256:
            msg = "clks_per_interval ({}) must be on [0, 256)."
            msg = msg.format(clks_per_interval)
            raise RuntimeError(msg)
        if not 0 <= pulses_per_frame < 256:
            msg = "pulses_per_frame ({}) must be on [0, 256)."
            msg = msg.format(pulses_per_frame)
            raise RuntimeError(msg)
        if not 0 <= intervals_per_frame < 2**16:
            msg = "intervals_per_frame ({}) must be on [0, 2**16)."
            msg = msg.format(intervals_per_frame)
            raise RuntimeError(msg)
        wdata = (intervals_per_frame << 16 |
                 pulses_per_frame << 8 |
                 clks_per_interval)
        self._send_write(58, wdata)
        if check:
            rdata = self._send_read(58)
            return rdata == wdata

    def start_laser(self):
        self._send_write(59, 1)

    def stop_laser(self):
        self._send_write(59, 0)

    def fire_laser(self):
        self.start_laser()

    def set_bsc_config(self, check=False, **kwargs):
        dwell_cnt = kwargs['dwell_cnt']
        ito_tc = kwargs['ito_tc']
        ito_invert = kwargs['ito_invert']
        prog_trigger_mode = kwargs.get('prog_trigger_mode', 0)

        if not 0 <= dwell_cnt < 2**20:
            msg = "Dwell value ({}) must be on [0, 2**20).".format(dwell_cnt)
            raise RuntimeError(msg)
        if not 0 <= ito_tc < 2**8:
            msg = "ITO TC value ({}) must be on [0, 2**8).".format(ito_tc)
            raise RuntimeError(msg)
        if ito_invert not in (0, 1):
            msg = "ITO invert ({}) must be zero or one.".format(ito_invert)
            raise RuntimeError(msg)
        if prog_trigger_mode not in (0, 1):
            msg = "PROG_TRIGGER_MODE ({}) must be zero or one."
            msg = msg.format(prog_trigger_mode)
            raise RuntimeError(msg)

        val = ((prog_trigger_mode << 29) | 
               (ito_invert << 28) |
               (ito_tc << 20) |
               dwell_cnt)
        self._send_write(61, val)
        if check:
            rdata = self._send_read(61)
            return rdata == val

    def dwell_count_from_us(self, time_us):
        # 120MHz clock. 2 counts for POL state, 1 count for counter compare
        # offset, and 1 count for pipeline in dwell_expired.
        return int(time_us * 120 - 4)

    def dwell_count_to_us(self, dwell_count):
        # 120MHz clock. 2 counts for POL state, 1 count for counter compare
        # offset, and 1 count for pipeline in dwell_expired.
        return (dwell_count + 4) / 120

    def read_status(self):
        return self._send_read(63)

    def driver_is_done(self):
        return bool(self.read_status() >> 1 & 1)

    def table_transfer_is_busy(self):
        return bool(self.read_status() & 1)

    # -------------------------------------------------------------------------
    # BSC API: memory access                                           (Public)
    # -------------------------------------------------------------------------
    def bsc_read(self, addr):
        return self._send_read(addr, periph='bsc')

    def bsc_write(self, addr, data):
        self._send_write(addr, data, periph='bsc')

    def get_table(self):
        words = self._send_read_range(0, 204//4)
        return self.word_array_to_byte_array(words)

    def set_table(self, coeffs, check=False):
        if len(coeffs) != 204:
            msg = "List has {} coeffs but {} are expected."
            msg = msg.format(len(coeffs), 204)
            raise RuntimeError(msg)
        for i, coeff in enumerate(coeffs):
            if not 0 <= coeff < 2**8:
                msg = "Invalid coeff value {} for channel {}.".format(coeff, i)
                raise RuntimeError(msg)
        words = self.byte_array_to_word_array(coeffs)
        self._send_write_range(0, words)
        if check:
            rdata = self._send_read_range(0, len(words))
            return rdata == words

    def ds_set_apply(self, coeffs, check=False):
        #t1 = time.time()
        if len(coeffs) != 204:
            msg = "List has {} coeffs but {} are expected."
            msg = msg.format(len(coeffs), 204)
            raise RuntimeError(msg)
        for i, coeff in enumerate(coeffs):
            if not 0 <= coeff < 2**8:
                msg = "Invalid coeff value {} for channel {}.".format(coeff, i)
                raise RuntimeError(msg)
        if check != False:
            raise RuntimeError("Direct Sockets don't support readback checking")

        coeffbytes = bytes(coeffs.astype('byte'))
        # Command byte 0x02 sets table with following 204 bytes, then applies
        self._zconn._ds.sendall(b'\x02' + coeffbytes)

    def ds_load_apply_waitack(self):
        reply = self._zconn._ds.recv(1);
        if reply != b'\x05':
            raise RuntimeError("Received bad direct socket reply: {}".format(reply))

    def ds_set_apply_waitack(self):
        reply = self._zconn._ds.recv(1);
        if reply != b'\x03':
            raise RuntimeError("Received bad direct socket reply: {}".format(reply))

    # -------------------------------------------------------------------------
    # SPI API: control and config                                      (Public)
    # -------------------------------------------------------------------------
    SPI_CMD_MAP = {k: i for i, k in enumerate(('standard', 'jumbo', 'stream'))}

    SPI_SLAVE_MAP = {k: i for i, k in enumerate(('ADC0',
                                                 'TEMP',
                                                 'ADC1',
                                                 'DAISY',
                                                 'CAL'))}

    SPI_CONFIG_MAP = {'fill_cmd_fifo': (0, 0x1),
                      'adc0_chsel': (1, 0x1),
                      'daisy_en': (2, 0x1)}

    def spi_config_clkdivs(self, arr):
        data = reduce((lambda x, y: x | y[1] << 4*y[0]), enumerate(arr), 0)
        self._send_write(24, data, periph='spi')

    def spi_set_config(self, val):
        self._send_write(25, val, periph='spi')

    def spi_get_config(self):
        return self._send_read(25, periph='spi')

    def spi_send_cmd(self, cmd_idx, slave, payload):
        if not 0 <= cmd_idx <= 2:
            msg = "Value {} for cmd_idx is not supported".format(cmd_idx)
            raise RuntimeError(msg)
        if not 0 <= slave < 5:
            msg = "Value {} for slave is not supported".format(slave)
            raise RuntimeError(msg)
        if not 0 <= payload < 2**24:
            msg = "Value {} for payload cannot exceed 24 bits".format(payload)
            raise RuntimeError(msg)
        data = cmd_idx << 28 | slave << 24 | payload
        self._send_write(26, data, periph='spi')

    def spi_read_rsp(self, n_rsp=1):
        for i in range(n_rsp):
            self._send_read(30, queue=True, periph='spi')
        _list = self._send_cmd_queue()
        if n_rsp == 1:
            return _list[0]
        else:
            return _list

    def spi_get_rsp_is_valid(self, rsp):
        return rsp >> 31 == 1

    def spi_get_rsp_fifo_count(self, rsp):
        return rsp >> 19 & 0x3f

    def spi_get_rsp_slave_idx(self, rsp):
        return rsp >> 16 & 0x7

    def spi_get_rsp_payload(self, rsp):
        return rsp & 0xffff

    def spi_read_status(self):
        return self._send_read(31, periph='spi')

    def spi_get_cmd_fifo_is_full(self):
        return bool(self.spi_read_status() >> 4 & 1)

    def spi_get_cmd_fifo_count(self):
        return self.spi_read_status() & 0xf

    def spi_get_cmd_fifo_headroom(self):
        return 15 - self.spi_get_cmd_fifo_count()

    # -------------------------------------------------------------------------
    # SPI API: memory access                                           (Public)
    # -------------------------------------------------------------------------
    def spi_read(self, addr):
        return self._send_read(addr, periph='spi')

    def spi_write(self, addr, data):
        self._send_write(addr, data, periph='spi')

    # Maps rail index (0-203) to a tuple t = (axi word, bit in word)
    SPI_SWITCH_MAP = ((1, 31), (0, 18), (2, 15), (0, 10), #  0
                      (0, 22), (0,  1), (3, 14), (0, 14), #  1
                      (0, 30), (0,  2), (0,  6), (0, 11), #  3
                      (1,  6), (0, 24), (2,  6), (0, 13), #  4
                      (1, 14), (1,  8), (3, 23), (0,  8), #  5
                      (1, 23), (0, 25), (1, 22), (0, 12), #  6
                      (1, 30), (1,  9), (2, 22), (0,  9), #  7
                      (2,  7), (1, 10), (2, 31), (0, 26), #  8
                      (2, 14), (1, 25), (4,  6), (0,  0), #  9
                      (3, 15), (1, 26), (3, 30), (0, 16), # 10
                      (2, 23), (1,  0), (2, 30), (0, 17), # 11
                      (3,  6), (1, 16), (4,  7), (1,  1), # 12
                      (3,  7), (2, 10), (4, 22), (1,  2), # 13
                      (4, 15), (1, 17), (3, 31), (1, 18), # 14
                      (3, 22), (2,  2), (4, 14), (1, 24), # 15
                      (4, 30), (2,  1), (4, 23), (2,  9), # 16
                      (4, 31), (2,  0), (4, 29), (2,  8), # 17
                      (5, 15), (2, 24), (6,  7), (2, 26), # 18
                      (4, 21), (2, 18), (5, 13), (2, 25), # 19
                      (6, 15), (3,  9), (5, 14), (2, 16), # 20
                      (5, 23), (3,  2), (5, 31), (3, 10), # 21
                      (5, 22), (2, 17), (6, 14), (3,  0), # 22
                      (5, 30), (3,  8), (5,  7), (3,  1), # 23
                      (5, 29), (3, 17), (5,  6), (3, 26), # 24
                      (5,  5), (3, 24), (6, 13), (3, 16), # 25
                      (5,  3), (1, 15), (5, 21), (4, 10), # 26
                      (5, 11), (4,  0), (6, 12), (4,  9), # 27
                      (5,  4), (3, 18), (6, 11), (4, 26), # 28
                      (5, 28), (4, 17), (4,  3), (4, 16), # 29
                      (5, 10), (3, 25), (3, 19), (5,  0), # 30
                      (5, 27), (4,  1), (3, 20), (5, 26), # 31
                      (5,  9), (6,  0), (4,  4), (5, 16), # 32
                      (5, 20), (4,  8), (4, 25), (6, 10), # 33
                      (4,  5), (4, 18), (3,  4), (5, 25), # 34
                      (5,  8), (4,  2), (4, 24), (6,  9), # 35
                      (4, 27), (0,  5), (3, 21), (5, 24), # 36
                      (4, 20), (5,  2), (2, 19), (6,  3), # 37
                      (5, 12), (5, 18), (5, 19), (6,  8), # 38
                      (4, 28), (5,  1), (3,  5), (6,  4), # 39
                      (4, 19), (0,  4), (1, 27), (6,  2), # 40
                      (4, 12), (5, 17), (4, 11), (6,  5), # 41
                      (4, 13), (0, 21), (3, 13), (6,  6), # 42
                      (3, 28), (1,  5), (3, 27), (6,  1), # 43
                      (3, 29), (0,  3), (0, 27), (1,  7), # 44
                      (3, 11), (0, 20), (2, 11), (1, 21), # 45
                      (3,  3), (1,  4), (3, 12), (2,  5), # 46
                      (2, 21), (0, 19), (1, 11), (1, 20), # 47
                      (2, 20), (0, 29), (0, 28), (2,  4), # 48
                      (2, 12), (1,  3), (2, 27), (1, 19), # 49
                      (2, 28), (1, 29), (1, 28), (2,  3), # 50
                      (2, 13), (1, 13), (1, 12), (2, 29)) # 51

    def spi_write_jumbo_frame_data(self, spi0_words, spi1=None, check=False):
        if spi1 is None:
            spi1_words = [0] * len(spi0_words)
        else:
            spi1_words = spi1

        if len(spi0_words) > 7:
            msg = "List has more than 7 words.".format(len(spi0_words))
            raise RuntimeError(msg)
        if len(spi1_words) > 7:
            msg = "List has more than 7 words.".format(len(spi1_words))
            raise RuntimeError(msg)
        self._send_write_range(0, spi0_words, periph='spi')
        self._send_write_range(8, spi1_words, periph='spi')
        if check:
            rdata0 = self._send_read_range(0, len(spi0_words))
            rdata1 = self._send_read_range(8, len(spi1_words))
            return rdata0 == spi0_words and rdata1 == spi1_words

    # -------------------------------------------------------------------------
    # ONEWIRE API                                                      (Public)
    # -------------------------------------------------------------------------
    def _int_to_signed(self, val, n_bits, n_fractional):
        if val < 2**(n_bits-1):
            tmp = val
        else:
            tmp = val - 2**n_bits
        return tmp * 2**(-n_fractional)

    def onewire_read_temp0(self):
        data = self._send_read(29, periph='spi')
        if data & 1<<12 != 0:
            return self._int_to_signed(data & 0xfff, 12, 4)
        else:
            return None

    def onewire_read_temp1(self):
        data = self._send_read(28, periph='spi')
        if data & 1<<12 != 0:
            return self._int_to_signed(data & 0xfff, 12, 4)
        else:
            return None

    def onewire_read_temp2(self):
        data = self._send_read(27, periph='spi')
        if data & 1<<12 != 0:
            return self._int_to_signed(data & 0xfff, 12, 4)
        else:
            return None

    # -------------------------------------------------------------------------
    # MISC API                                                         (Public)
    # -------------------------------------------------------------------------
    def exec_adc_time_series(self, meas_channel, gnd_channel, n_wait_cycles):
        cmd = 'exec_adc_time_series {} {} {}'
        cmd = cmd.format(meas_channel, gnd_channel, n_wait_cycles)
        return self._send_custom(cmd, 5, postproc=lambda x: int(x, 16))

    def exec_adc_time_series_full(self, n_wait_cycles,
                                        tries=8,
                                        sleep=10,
                                        path=''):
        cmd = 'exec_adc_time_series_full {}'.format(n_wait_cycles)
        out = self._send_custom(cmd,
                                3,
                                postproc=lambda x: x,
                                tries=tries,
                                sleep=sleep)
        self.remote_csv_file = out[0]
        filename = self.remote_csv_file.split('/')[-1]
        with SCPClient(self.get_connection()._conn.get_transport()) as scp:
            scp.get(self.remote_csv_file, local_path=path)
        output_path_abs = os.path.abspath(os.path.join(path, filename))
        return output_path_abs

    def exec_adc_time_series_nearest_neighbors(self, n_wait_cycles,
                                        tries=8,
                                        sleep=10,
                                        path=''):
        cmd = 'exec_adc_time_series_nearest_neighbors {}'.format(n_wait_cycles)
        out = self._send_custom(cmd,
                                3,
                                postproc=lambda x: x,
                                tries=tries,
                                sleep=sleep)
        self.remote_csv_file = out[0]
        filename = self.remote_csv_file.split('/')[-1]
        with SCPClient(self.get_connection()._conn.get_transport()) as scp:
            scp.get(self.remote_csv_file, local_path=path)
        output_path_abs = os.path.abspath(os.path.join(path, filename))
        return output_path_abs


if __name__ == '__main__':
    # init
    host = sys.argv[1]
    zynq = OrchidZynqAPI(host)
    zynq.start_remote_app()

    print("\nLOG FILE")
    for line in zynq._read_log_file():
        print('    ' + line)

    # main
    rsp = zynq._send_write(0, 0xfedc_ba98)
    rsp = zynq._send_write(1, 0x7654_3210)
    rsp = zynq._send_read(0)
    rsp = zynq._send_read(1)

    # close and report
    zynq.stop_remote_app()

    print("\nCOMMAND FILE")
    for line in zynq._read_cmd_file():
        print('    ' + line)

    print("\nRESPONSE FILE")
    for line in zynq._read_rsp_file():
        print('    ' + line)

    print("\nLOG FILE")
    for line in zynq._read_log_file():
        print('    ' + line)

    zynq.close_connection()

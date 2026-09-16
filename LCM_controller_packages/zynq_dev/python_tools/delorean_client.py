from contextlib import contextmanager
import paramiko
import socket
import sys
import time

from python_tools.delorean_mem_map import DeloreanMemMap


class DeloreanAppHandler:
    def __init__(self, host, design='delorean'):
        self.host = host
        self.design = design
        self.username = 'root'
        self.password = 'incendia'
        self._conn = paramiko.client.SSHClient()
        self._conn.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())

    @contextmanager
    def get_connection(self):
        self._conn.connect(self.host,
                           username=self.username,
                           password=self.password)
        try:
            yield self._conn
        finally:
            self._conn.close()

    def connect(self):
        with self.get_connection() as conn:
            self._program_bitstream(conn)
            self._start_remote_app(conn)

    def disconnect(self):
        pass

    def exec_command(self, cmd, background=False):
        with self.get_connection() as conn:
            return self._exec_command(conn, cmd, background=background)

    def _exec_command(self, conn, cmd, background=False):
        # both versions open a new channel with the current transport
        if background:
            channel = conn.get_transport().open_session()
            cmd += " &"
            channel.exec_command(cmd)
            channel.close()
            lines = []
        else:
            sin, sout, serr = conn.exec_command(cmd)
            lines = [line.strip() for line in sout.readlines()]
        return lines

    @property
    def log_file_path(self):
        return '/root/{0}/{0}_app.log'.format(self.design)

    @property
    def stderr_path(self):
        return '/root/{0}/stderr.tmp'.format(self.design)

    @property
    def pattern_file_path(self):
        return '/root/{0}/patterns.txt'.format(self.design)

    @property
    def remote_app_path(self):
        return '/root/{0}/{0}_app.elf'.format(self.design)

    @property
    def bitstream_path(self):
        return '/root/{0}/{0}.bit'.format(self.design)

    def program_bitstream(self, tries=3, sleep=1):
        with self.get_connection() as conn:
            return self._program_bitstream(conn, tries=tries, sleep=sleep)

    def _program_bitstream(self, conn, tries=3, sleep=1):
        cmd = 'cat {} > /dev/xdevcfg'.format(self.bitstream_path)
        self._exec_command(conn, cmd)
        for i in range(tries):
            if self._get_bitstream_program_status(conn):
                return True
            time.sleep(sleep)
        raise Exception("Bitstream not programmed after {} tries".format(tries))

    def get_bitstream_program_status(self):
        with self.get_connection() as conn:
            return self._get_bitstream_program_status(conn)

    def _get_bitstream_program_status(self, conn):
        cmd = 'cat /sys/class/xdevcfg/xdevcfg/device/prog_done'
        out = self._exec_command(conn, cmd)
        return len(out) == 1 and out[0] == '1'

    def start_remote_app(self):
        with self.get_connection() as conn:
            self._start_remote_app(conn)

    def _start_remote_app(self, conn):
        if self._remote_app_is_running(conn):
            raise RuntimeError("Remote app is already running.")
        cmd = '{} {} 2>{}'.format(self.remote_app_path,
                                  self.pattern_file_path,
                                  self.stderr_path)
        self._exec_command(conn, cmd, background=True)

    def remote_app_is_running(self):
        with self.get_connection() as conn:
            return self._remote_app_is_running(conn)

    def _remote_app_is_running(self, conn):
        cmd = 'pgrep -f "{}"'.format(self.remote_app_path)
        out = self._exec_command(conn, cmd)
        return out != []


class DeloreanSocketHandler:
    def __init__(self, host, port=1844):
        self.host = host
        self.port = port
        self._sock = None

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))
        # send data as soon as possible, even if there is only a small amount.
        self._sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)

    def disconnect(self):
        self._sock.close()

    def get_connection(self):
        return self._sock

    def send_cmd(self, code, payload):
        if not 0 <= code < 128:
            # MSB is reserved
            raise RuntimeError("Invalid code ({}).".format(code))
        length = len(payload)
        if length >= 2080:
            raise RuntimeError("Invalid payload length ({}).".format(length))
        self._sock.sendall(bytes([code]) +
                           length.to_bytes(2, byteorder='little') +
                           bytes(payload))

    def recv_rsp(self, code, tries=10):
        if not 0 <= code < 128:
            # MSB is reserved
            raise RuntimeError("Invalid code ({}).".format(code))
        # indexing to a single element converts 'bytes' to 'int'
        rcode = self._sock.recv(1)[0]
        byte_str = self._sock.recv(1)
        byte_str += self._sock.recv(1)
        length = int.from_bytes(byte_str, byteorder='little')
        if rcode & 0x80 or (rcode & 0x7f) != code:
            raise RuntimeError("Invalid rsp code (0x{:02x}).".format(rcode))
        byte_str = b''
        bytes_remaining = length
        while bytes_remaining > 0 and tries > 0:
            data = self._sock.recv(bytes_remaining)
            tries -= 1
            byte_str += data
            bytes_remaining -= len(data)
            if data == b'':
                raise RuntimeError("Server closed socket.")
        if bytes_remaining != 0:
            msg = "Didn't read all bytes ({} remain).".format(bytes_remaining)
            raise RuntimeError(msg)
        return byte_str


class DeloreanApi:
    def __init__(self, host):
        self.host = host
        self.app_handler = DeloreanAppHandler(host)
        self.sock_handler = DeloreanSocketHandler(host)
        self.map = DeloreanMemMap()

    def connect(self):
        self.app_handler.connect()
        time.sleep(2)
        self.sock_handler.connect()

    def disconnect(self):
        self.sock_handler.disconnect()
        self.app_handler.disconnect()

    # -------------------------------------------------------------------------
    # Application commands
    # -------------------------------------------------------------------------
    def exec_exit(self):
        self.sock_handler.send_cmd(0x00, [])
        return self.sock_handler.recv_rsp(0x00)

    def exec_ping(self):
        self.sock_handler.send_cmd(0x01, [])
        return self.sock_handler.recv_rsp(0x01)

    def exec_hola(self):
        self.sock_handler.send_cmd(0x02, [])
        return self.sock_handler.recv_rsp(0x02)

    def exec_echo(self, lyst):
        self.sock_handler.send_cmd(0x03, lyst)
        return self.sock_handler.recv_rsp(0x03)

    def exec_read_lcm(self, addr):
        payload = addr.to_bytes(4, byteorder='little')
        self.sock_handler.send_cmd(0x04, payload)
        data = self.sock_handler.recv_rsp(0x04)
        return int.from_bytes(data, byteorder='little')

    def exec_read_spi(self, addr):
        payload = addr.to_bytes(4, byteorder='little')
        self.sock_handler.send_cmd(0x05, payload)
        data = self.sock_handler.recv_rsp(0x05)
        return int.from_bytes(data, byteorder='little')

    def exec_write_lcm(self, addr, data):
        payload = (addr.to_bytes(4, byteorder='little') +
                   data.to_bytes(4, byteorder='little'))
        self.sock_handler.send_cmd(0x06, payload)
        return self.sock_handler.recv_rsp(0x06)

    def exec_write_spi(self, addr, data):
        payload = (addr.to_bytes(4, byteorder='little') +
                   data.to_bytes(4, byteorder='little'))
        self.sock_handler.send_cmd(0x07, payload)
        return self.sock_handler.recv_rsp(0x07)

    def exec_cdma_down(self, ddr_idx, bram_idx):
        payload = (ddr_idx.to_bytes(4, byteorder='little') +
                   bram_idx.to_bytes(4, byteorder='little'))
        self.sock_handler.send_cmd(0x08, payload)
        return self.sock_handler.recv_rsp(0x08)

    def exec_cdma_up(self, ddr_idx, bram_idx):
        payload = (ddr_idx.to_bytes(4, byteorder='little') +
                   bram_idx.to_bytes(4, byteorder='little'))
        self.sock_handler.send_cmd(0x09, payload)
        return self.sock_handler.recv_rsp(0x09)

    def exec_cdma_reset(self):
        self.sock_handler.send_cmd(0x0a, [])
        return self.sock_handler.recv_rsp(0x0a)

    def exec_get_table(self, idx):
        payload = idx.to_bytes(4, byteorder='little')
        self.sock_handler.send_cmd(0xb, payload)
        return self.sock_handler.recv_rsp(0x0b)

    def exec_set_table(self, mask, idx, lyst):
        payload = (mask.to_bytes(4, byteorder='little') +
                   idx.to_bytes(4, byteorder='little') +
                   bytes(lyst))
        self.sock_handler.send_cmd(0xc, payload)
        return self.sock_handler.recv_rsp(0x0c)

    def exec_set_fabric_reset(self, reset_en):
        payload = reset_en.to_bytes(4, byteorder='little')
        self.sock_handler.send_cmd(0x7e, payload)
        return self.sock_handler.recv_rsp(0x7e)

    def exec_shutdown(self):
        self.sock_handler.send_cmd(0x7f, [])
        return self.sock_handler.recv_rsp(0x7f)


if __name__ == '__main__':
    host = sys.argv[1]
    delorean = DeloreanApi(host)
    delorean.connect()
    print("\nPING: ", delorean.exec_ping())
    print("\nHOLA: ", delorean.exec_hola())
    print("\nECHO: ", delorean.exec_echo([1, 2, 3]))
    print("\nREAD_LCM: ", delorean.exec_read_lcm(24))
    print("\nWRITE_LCM: ", delorean.exec_write_lcm(23, 0xf00dface))
    print("\nREAD_LCM: 0x{:08x}".format(delorean.exec_read_lcm(23)))
    print("\nWRITE_SPI: ", delorean.exec_write_spi(0, 0xdeadc0de))
    print("\nREAD_SPI: 0x{:08x}".format(delorean.exec_read_spi(0)))

    errs = 0
    n = 10**3
    t0 = time.time()
    for i in range(n):
        delorean.exec_write_spi(0, i)
        errs += (1 if i != delorean.exec_read_spi(0) else 0)
    t1 = time.time()
    print("Time: {} sec with {} errors".format((t1-t0)/n, errs))

    delorean.exec_exit()
    delorean.disconnect()

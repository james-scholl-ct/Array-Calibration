import os
import paramiko
import sys
import time
from contextlib import contextmanager
import socket

import python_tools.git_utils as git


class ZynqConnection:
    def __init__(self, host):
        self.host = host
        self.username = 'root'
        self.password = 'incendia'

        self._conn = paramiko.client.SSHClient()
        self._conn.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())
        self._conn.connect(self.host,
                           username=self.username,
                           password=self.password)

    def get_connection(self):
        return self._conn

    def ds_open(self):
        # Open TCP Side Channel for Quick Pattern Loading
        self._ds = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._ds.connect((self.host, 1844))

        # Be careful -- try not to send tinygrams
        self._ds.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)

    def close(self):
        self._conn.close()
        #self._ds.close()

    def exec_command(self, cmd, background=False):
        # both versions open a new channel with the current transport
        if background:
            channel = self._conn.get_transport().open_session()
            cmd += " &"
            channel.exec_command(cmd)
            channel.close()
            lines = []
        else:
            sin, sout, serr = self._conn.exec_command(cmd)
            lines = [line.strip() for line in sout.readlines()]
        return lines

    def file_exists(self, path):
        bash = 'if [ -f {} ]; then echo 1; else echo 0; fi'.format(path)
        out = self.exec_command(bash)
        return out[0] == '1'

    def upload_file(self, local_path, remote_path=None):
        if remote_path is None:
            remote_path = os.path.basename(local_path)
        sftp_conn = self._conn.open_sftp()
        sftp_conn.put(local_path, remote_path)
        sftp_conn.close()

    def get_local_bitstream_path(self, module):
        return os.path.join(git.get_git_root(),
                            '_build_{}'.format(module),
                            '{}.bit'.format(module))

    def program_bitstream(self, remote_path, tries=3, sleep=1):
        self.exec_command('cat {} > /dev/xdevcfg'.format(remote_path))
        for i in range(tries):
            if self.get_bitstream_program_status():
                return True
            time.sleep(sleep)
        raise Exception("Bitstream not programmed after {} tries".format(tries))

    def get_bitstream_program_status(self):
        cmd = 'cat /sys/class/xdevcfg/xdevcfg/device/prog_done'
        out = self.exec_command(cmd)
        return len(out) == 1 and out[0] == '1'


if __name__ == '__main__':
    host = sys.argv[1]
    design = sys.argv[2]
    zc = ZynqConnection(host)

    out = zc.exec_command('mkdir -p /root/' + design)
    cmd = 'ls -rtl /root/' + design
    out = zc.exec_command(cmd)
    print(cmd)
    [print(4 * ' ' + line) for line in out]
    print('')

    lpath = zc.get_local_bitstream_path(design)
    rpath = '/root/{0}/{0}.bit'.format(design)
    print('Uploading bitstream.\n')
    zc.upload_file(lpath, rpath)

    cmd = 'ls -rtl /root/' + design
    out = zc.exec_command(cmd)
    print(cmd)
    [print(4 * ' ' + line) for line in out]
    print('')

    print('Programming bitstream.\n')
    zc.program_bitstream(rpath)

    zc.close()
    print('Done.')

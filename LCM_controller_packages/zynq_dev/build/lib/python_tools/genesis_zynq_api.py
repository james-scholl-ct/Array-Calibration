import sys

from python_tools.zynq_api import ZynqAPI


class GenesisZynqAPI(ZynqAPI):
    def __init__(self, host):
        mem_depth = {'ip0': 64, 'ip1': 64}
        super(GenesisZynqAPI, self).__init__(host, 'genesis', mem_depth)

    def shutdown_sequence(self):
        super(GenesisZynqAPI, self).shutdown()

    # -------------------------------------------------------------------------
    # IP0 API: memory access                                           (Public)
    # -------------------------------------------------------------------------
    def ip0_read(self, addr):
        return self._send_read(addr, periph='ip0')

    def ip0_write(self, addr, data):
        self._send_write(addr, data, periph='ip0')

    # -------------------------------------------------------------------------
    # IP1 API: memory access                                           (Public)
    # -------------------------------------------------------------------------
    def ip1_read(self, addr):
        return self._send_read(addr, periph='ip1')

    def ip1_write(self, addr, data):
        self._send_write(addr, data, periph='ip1')


if __name__ == '__main__':
    # Example usage. Try-Except ensures the remote end hangs up for exceptions.
    host = sys.argv[1]
    zynq = GenesisZynqAPI(host)
    zynq.start_remote_app()

    print('Zynq connected. Press Ctrl-C to disconnect and close.')
    sys.stdout.flush()
    try:
        while True:
            pass
    except (SystemExit, KeyboardInterrupt):
        zynq.shutdown_sequence()
        print('Goodbye!')
    except:
        zynq.shutdown_sequence()
        raise

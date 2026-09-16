'''
Top-level script to verify Himax daughter boards.
'''
import argparse
import random
import sys

from python_tools.zynq_api import LotusZynqAPI
from python_tools.bsc_patgen import BSCPatternGenerator


N_CHANNELS = 204

DRIVE_VOLTAGE = 4.0

args = None
lotus = None
pg = None


def sample_channel_voltage():
    cmd = LotusZynqAPI.SPI_CMD_MAP['standard']
    slave = LotusZynqAPI.SPI_SLAVE_MAP['ADC1']
    # 1st: sample and convert; 2nd: readout
    lotus.spi_send_cmd(cmd, slave, 0x0000)
    lotus.spi_send_cmd(cmd, slave, 0x0000)
    rsps = lotus.spi_read_rsp(n_rsp=2)
    test = lotus.spi_get_rsp_is_valid(rsps[1]) and \
           lotus.spi_get_rsp_slave_idx(rsps[1]) == slave and \
           lotus.spi_get_rsp_fifo_count(rsps[1]) == 1
    if not test:
        raise RuntimeError("Bad response 0x{:08x}".format(rsps[1]))
    sample = lotus.spi_get_rsp_payload(rsps[1]) >> 2
    voltage = sample * 0.00025 * 1.5
    return voltage


def select_channel(channel):
    # generate frame data
    vt_words = [0x0000_0000] * 7
    gnd_words = [0xffff_ffff] * 7
    word_addr, bit = LotusZynqAPI.SPI_SWITCH_MAP[channel]
    vt_words[word_addr] = vt_words[word_addr] | (1<<bit)
    gnd_words[word_addr] = gnd_words[word_addr] & ~(1<<bit)

    # write frame data
    lotus.spi_write_jumbo_frame_data(gnd_words, vt_words)

    # send frame data to switches
    cmd = LotusZynqAPI.SPI_CMD_MAP['jumbo']
    slave = LotusZynqAPI.SPI_SLAVE_MAP['DAISY']
    lotus.spi_send_cmd(cmd, slave, 12)


def program_onehot_pattern(channel):
    bsc_v_pattern = [0] * N_CHANNELS
    bsc_v_pattern[channel] = DRIVE_VOLTAGE
    codes = pg.get_channel_codes(bsc_v_pattern)
    lotus.set_table(codes)


def enable_daisy_chain_mode():
    cmd = LotusZynqAPI.SPI_CMD_MAP['standard']
    slave = LotusZynqAPI.SPI_SLAVE_MAP['DAISY']
    lotus.spi_send_cmd(cmd, slave, 0x2500)
    lotus.spi_read_rsp()


def reset_switches():
    lotus.spi_set_config(0)
    pos, mask = LotusZynqAPI.SPI_CONFIG_MAP['daisy_en']
    lotus.spi_set_config(mask << pos)


def init_pg():
    pg.n_channels = N_CHANNELS
    pg.channel_parity = [True for i in range(N_CHANNELS)]
    pg.pol = 0


def init_lotus():
    lotus.start_remote_app()
    lotus.pol_ovr = 1
    lotus.set_dwell(lotus.dwell_count_from_us(10))
    lotus.init()
    lotus.spi_config_clkdivs([2, 9, 2, 2, 2])
    reset_switches()
    enable_daisy_chain_mode()


def main(args_):
    global args
    global lotus
    global pg

    args = args_
    if args.verbose:
        print('Connecting to Zynq host {}.'.format(args.zynq))
    lotus = LotusZynqAPI(args.zynq)
    pg = BSCPatternGenerator()
    init_lotus()
    init_pg()

    random_channel = random.randrange(N_CHANNELS)
    for channel in range(N_CHANNELS):
        program_onehot_pattern(channel)
        lotus.apply_table()
        if channel == random_channel:
            switch_list = range(N_CHANNELS)
        else:
            switch_list = [channel]
        for i in switch_list:
            select_channel(i)
            v = sample_channel_voltage()
            print("{}, {}, {:.3f}".format(channel, i, v))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-z', '--zynq', '--host',
                        required=True,
                        help="zynq host name, e.g. 'microzed-12-34-56'")
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help="Enable verbosity")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    try:
        print("\nPlease remove all jumpers on the short-check board.")
        ans = input("Proceed? [y/n] ")
        if ans == 'y':
            main(args)
            lotus.shutdown_sequence()
    except:
        lotus.shutdown_sequence()
        raise

'''
Script to drive Laser signals on Delorean
'''
import argparse
import math
import sys

from python_tools.delorean_client import DeloreanApi

DEBUG = False


def set_fields(delorean, periph, **kwargs):
    """Writes a set of fields to the indicated peripheral in the FPGA. NB:
    this function does read-modify-writes but the order of programming is
    not guaranteed due to kwargs being a dict. The user is responsible for
    ensuring proper ordering of register writes.
    """
    if periph == 'lcm':
        wr_func = delorean.exec_write_lcm
        rd_func = delorean.exec_read_lcm
    elif periph == 'spi':
        wr_func = delorean.exec_write_spi
        rd_func = delorean.exec_read_spi
    else:
        raise RuntimeError("Invalid periph: {}.".format(periph))

    for field, value in kwargs.items():
        addr = delorean.map.get_field_word_addr(periph, field)
        mask = delorean.map.get_field_mask(periph, field)
        vmask = delorean.map.get_field_mask(periph, field, value)
        data = rd_func(addr)
        data = (data & ~mask) | vmask
        wr_func(addr, data)


def get_fields(delorean, periph, *args):
    """Returns a dictionary of values for each of the given fields.
    """
    if periph == 'lcm':
        rd_func = delorean.exec_read_lcm
    elif periph == 'spi':
        rd_func = delorean.exec_read_spi
    else:
        raise RuntimeError("Invalid periph: {}.".format(periph))

    ret = {k: None for k in args}
    for field in args:
        addr = delorean.map.get_field_word_addr(periph, field)
        data = rd_func(addr)
        ret[field] = delorean.map.get_field_value(periph, field, data)
    return ret


def dump_lcm_fields(delorean, with_print=False):
    """Returns a dictionary of all lcm fields and their current values.
    """
    keys = delorean.map['fields']['lcm'].keys()
    lcm_fields = get_fields(delorean, 'lcm', *keys)
    if with_print:
        for k, v in sorted(lcm_fields.items()):
            print(f'{k:<28s}  {v:08x}  {v}')
        print('')
    return lcm_fields


def get_laser_config(delorean,
                     prf_khz=1,
                     interval_us=1,
                     ppf=1,
                     pw_index=4):
    """Return all laser configuration per the provided settings. The
    return value is suitable for use with set_fields.
    """
    clks_per_interval = int(interval_us * delorean.clk_freq_mhz)
    prf_period_us = 1000 / prf_khz
    intervals_per_frame = math.ceil(prf_period_us / interval_us)
    pulses_per_frame = int(min(ppf, intervals_per_frame))

    if True:
        rep_rate_sec = (clks_per_interval *
                        intervals_per_frame /
                        (delorean.clk_freq_mhz * 1e6))
        msg = "Laser rep rate set to {:0.1f} kHz ({:0.1f} us)."
        msg = msg.format(1/rep_rate_sec/1e3, rep_rate_sec*1e6)
        print(msg)
    return {'laser_pw_sel': (1 << pw_index),
            'clks_per_interval': (clks_per_interval - 1),
            'pulses_per_frame': (pulses_per_frame - 1),
            'intervals_per_frame': (intervals_per_frame - 1)}


def config_delorean(args, delorean):
    """Configure some sensible non-zero (non-default) values with special
    mention of Bravo- and Delta-specific fields.
    """
    lcm_config = {
        'tp1_period': 9996,
        'reset_code': 0x00,         # 0xff for Bravo
        'pol_finish_ovr': 0,        # 1 for Bravo
        'tp1_done_high': 0,         # 1 for Bravo
        'ito_async': 1,
        'ito_invert': 0,
        'ito_tc': 49999,
        'n_steps': 170,
        'rst_pw': 4,
        'tx_wait': 7,
        'tp1_pw': int(0.5 * delorean.clk_freq_mhz - 1),
        'prog_trigger_mode': 1}     # pulse mode
    laser_config = get_laser_config(delorean,
                                    prf_khz=args.freq,
                                    pw_index=args.pw)
    lcm_config.update(laser_config)
    spi_config = {
        'clk_div_adc': 2,           # 33.3 MHz
        'clk_div_switch': 9,        # 10.0 MHz
        'clk_div_tmp': 9,           # 10.0 MHz
        'clk_div_pot_ito': 19,      #  5.0 MHz
        'clk_div_pot_tx': 19}       #  5.0 MHz
    set_fields(delorean, 'lcm', **lcm_config)
    set_fields(delorean, 'spi', **spi_config)


def main(args, delorean):
    delorean.clk_freq_mhz = get_fields(delorean, 'lcm', 'clk_freq')['clk_freq']
    config_delorean(args, delorean)
    set_fields(delorean, 'lcm', tx_pwr_en=1)
    if DEBUG:
        dump_lcm_fields(delorean, with_print=True)

    input("\nPress Enter to start the laser.")
    set_fields(delorean, 'lcm', laser_start=1)
    print("Laser started.")
    input("\nPress Enter to stop the laser.")
    set_fields(delorean, 'lcm', laser_start=0)
    print("Laser stopped.")


def validate_args(args):
    if args.freq > 20:
        msg = "Error: Frequency is too high: ({}) kHz.".format(args.freq)
        raise RuntimeError(msg)
    if args.pw not in range(16):
        msg = "Error: Pulse width is invalid: ({}).".format(args.pw)
        raise RuntimeError(msg)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-z', '--zynq',
                        required=True,
                        help="Zynq IPv4 Address, e.g. '192.168.128.xxx'.")
    parser.add_argument('-f', '--freq',
                        default=1.,
                        type=float,
                        help="Pulse repetition frequency in kHz.")
    parser.add_argument('-p', '--pw',
                        default=0,
                        type=int,
                        help="Laser drive pulse width index, an int on range(16).")
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help="Enable verbosity.")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    validate_args(args)
    if args.verbose:
        print('\nConnecting to Zynq device {}.'.format(args.zynq))

    delorean = DeloreanApi(args.zynq)
    delorean.connect()
    main(args, delorean)
    delorean.exec_exit()
    delorean.disconnect()

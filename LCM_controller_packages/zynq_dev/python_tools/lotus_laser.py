'''
Script to drive Laser signals on Lotus
'''
import argparse
import math
import sys

from python_tools.zynq_api import LotusZynqAPI

args = None
lotus = None


def init_lotus(prf_khz=20, interval_us=1, pw_index=0, trig_index=0):
    global args
    global lotus

    lotus.start_remote_app()

    # Configure laser rep rate
    pulses_per_frame = 1
    clks_per_interval = int(interval_us * 100)  # 1 cycle / 0.010 us = 100
    prf_period_us = 1000 / prf_khz
    intervals_per_frame = math.ceil(prf_period_us / interval_us)

    if args.verbose:
        rep_rate_sec = 10e-9 * clks_per_interval * intervals_per_frame
        msg = "Laser rep rate set to {:0.1f} kHz ({:0.1f} us)."
        msg = msg.format(1/rep_rate_sec/1e3, rep_rate_sec*1e6)
        print(msg)

    # for all fields, values map to a range starting at zero
    lotus.config_laser(clks_per_interval - 1,
                       pulses_per_frame - 1,
                       intervals_per_frame - 1,
                       check=True)

    # Configure laser drive timing
    trig_mask = 1 << trig_index
    pw_mask = 1 << pw_index
    lotus.bsc_write(57, (trig_mask << 16 | pw_mask))


def main(args_, lotus_):
    global args
    global lotus
    args = args_
    lotus = lotus_

    init_lotus(prf_khz=args.freq,
               pw_index=args.pw,
               trig_index=args.trig)
    input("\nPress Enter to start the laser.")
    lotus.start_laser()
    print("Laser started.")
    input("\nPress Enter to stop the laser.")
    lotus.stop_laser()
    print("Laser stopped.")


def validate_args(args):
    if args.freq > 300:
        msg = "Error: Frequency is too high: ({}) kHz.".format(args.freq)
        raise RuntimeError(msg)
    if args.pw not in range(16):
        msg = "Error: Pulse width is invalid: ({}).".format(args.pw)
        raise RuntimeError(msg)
    if args.trig not in range(8):
        msg = "Error: Trigger delay is invalid: ({}).".format(args.trig)
        raise RuntimeError(msg)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-z', '--zynq',
                        required=True,
                        help="Zynq hostname, e.g. 'microzed-12-34-56'.")
    parser.add_argument('-f', '--freq',
                        default=20.,
                        type=float,
                        help="Pulse repetition frequency in kHz.")
    parser.add_argument('-p', '--pw',
                        default=5,
                        type=int,
                        help="Laser drive pulse width index, an int on range(16).")
    parser.add_argument('-t', '--trig',
                        default=0,
                        type=int,
                        help="Laser trigger delay index, an int on range(8).")
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help="Enable verbosity.")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    validate_args(args)
    if args.verbose:
        print('\nConnecting to Zynq device {}.'.format(args.zynq))
    lotus = LotusZynqAPI(args.zynq)
    try:
        main(args, lotus)
        lotus.shutdown_sequence()
    except:
        lotus.shutdown_sequence()
        raise

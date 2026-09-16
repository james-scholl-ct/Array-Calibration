'''
Template script for DeloreanBoard.
'''
import argparse
import numpy as np
import os
import sys
import time

import lcm_board.delorean as delorean


def main(args, db):
    # Dump registers
    db.return_all_lcm_fields(with_print=args.verbose)
    db.return_all_spi_fields(with_print=args.verbose)


def validate_args(args):
    pass


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-z', '--zynq',
                        required=True,
                        help="Zynq IPv4 address, e.g. '192.168.128.xxx'.")
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help="Enable verbose messages.")
    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help="Enable debug messages and functionality.")
    parser.add_argument('-x', '--no_short_check',
                        action='store_true',
                        help="Disable LCM short checking during init.")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    validate_args(args)
    delorean.VERBOSE = args.verbose
    delorean.DEBUG = args.debug
    db = delorean.DeloreanBoard(args.zynq, args.no_short_check)

    main(args, db)

    db.disconnect()

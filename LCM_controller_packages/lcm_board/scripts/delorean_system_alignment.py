import argparse
import sys

import lcm_board.delorean as delorean
import lcm_board.voltage_pattern as vp

vpg = vp.VoltagePatternGenerator(
    wavelength=910,
    channels=1021,
    theta_i=-70,
    pitch=300,
    delv_min=0.1,
    delv_max=3,
)


def main(db, args):
    laser_config = db.get_laser_config(
        prf_khz=args.freq,
        interval_us=1,
        ppf=1,
        pw_index=args.pw
    )

    db.set_fields('lcm', **laser_config)
    print('Laser driver configured')

    db.tx_ci_a = args.ci
    print(f"TX Digipot A set to {db.tx_ci_a} V.")

    db.set_fields('lcm', tx_pwr_switch=0)
    print('Tx Digipot A enabled')

    db.set_fields('lcm', tx_pwr_en=1)
    print('Tx VLDA enabled')

    input("\nPress Enter to start the laser.")
    db.set_fields('lcm', laser_start=1)
    print("Laser started.")

    if args.tx_order and args.rx_order:
        delv_tx, _ = vpg.clipped_ramp(m=args.tx_order,
                                      ramp_fraction=0.5,
                                      plot=False,
                                      flip=True
                                      )

        delv_rx, _ = vpg.clipped_ramp(m=args.rx_order,
                                      ramp_fraction=0.5,
                                      plot=False,
                                      flip=True
                                      )

        input('\nPress Enter to steer both LCMs')
        db.write_pattern_delv(delv_pattern=delv_tx, tx_or_rx='tx')
        db.write_pattern_delv(delv_pattern=delv_rx, tx_or_rx='rx')
    elif args.tx_order:
        delv_tx, _ = vpg.clipped_ramp(m=args.tx_order,
                                      ramp_fraction=0.5,
                                      plot=False,
                                      flip=True
                                      )

        input('\nPress Enter to steer TX LCM')
        db.write_pattern_delv(delv_pattern=delv_tx, tx_or_rx='tx')
    elif args.rx_order:
        delv_rx, _ = vpg.clipped_ramp(m=args.rx_order,
                                      ramp_fraction=0.5,
                                      plot=False,
                                      flip=True
                                      )

        input('\nPress Enter to steer RX LCM')
        db.write_pattern_delv(delv_pattern=delv_rx, tx_or_rx='rx')
    else:
        pass

    return


def stop_laser(db):
    input("\nPress Enter to stop the laser.")
    db.set_fields('lcm', laser_start=0)
    db.set_fields('lcm', tx_pwr_en=0)
    print("Laser stopped.")


def validate_args(args):
    if args.freq > 250:
        msg = f"Error: Frequency is too high: ({args.freq}) kHz."
        raise RuntimeError(msg)
    if args.pw not in range(16):
        msg = f"Error: Pulse width is invalid: ({args.pw})."
        raise RuntimeError(msg)
    if args.ci > 5:
        msg = f'Error: CI voltage is too high: ({args.ci}).'
        raise RuntimeError(msg)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-z', '--zynq',
                        required=True,
                        help="Zynq IPv4 address, e.g. '192.168.128.xxx'.")
    parser.add_argument('-f', '--freq',
                        default=20.,
                        type=float,
                        help="Pulse repetition frequency in kHz.")
    parser.add_argument('-p', '--pw',
                        default=5,
                        type=int,
                        help="Laser drive pulse width index, an int on range(16).")
    parser.add_argument('-c', '--ci',
                        default=2.5,
                        type=float,
                        help="iC Haus CI voltage setting in volts, a float <= 5.")
    parser.add_argument('-t', '--tx_order',
                        default=None,
                        type=int,
                        help='Order for Tx steering')
    parser.add_argument('-r', '--rx_order',
                        default=None,
                        type=int,
                        help='Order for Rx steering')
    return parser.parse_args(argv)


def steer_rx_to_order(db, order):
    delv_rx, _ = vpg.clipped_ramp(m=order,
                                  ramp_fraction=0.5,
                                  plot=False,
                                  flip=True
                                  )
    db.write_pattern_delv(delv_pattern=delv_rx, tx_or_rx='rx')


def steer_tx_to_order(db, order):
    delv_tx, _ = vpg.clipped_ramp(m=order,
                                  ramp_fraction=0.5,
                                  plot=False,
                                  flip=True
                                  )
    db.write_pattern_delv(delv_pattern=delv_tx, tx_or_rx='tx')


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    validate_args(args)
    delorean.VERBOSE = True
    delorean.DEBUG = True
    if delorean.DEBUG:
        print('\nConnecting to Zynq device {}.'.format(args.zynq))

    db = delorean.DeloreanBoard(args.zynq)

    db.ito_amplitude_vpp = 0

    main(db, args)

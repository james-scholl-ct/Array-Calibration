import numpy as np
import sys
import time

import lcm_board.delorean as delorean

def main(db):
    # Dump registers
    if True:
        db.return_all_lcm_fields(with_print=True)
        db.return_all_spi_fields(with_print=True)


    # Check RTD output
    if False:
        db.spi_set_adc_ch_sel('tx')
        db.spi_set_switch_data('tx_rtd_temp')
        db.spi_apply_switch_data()
        output = db.spi_read_adcs_simple()
        rtd_output = output[1]
        input(f"Measure Voltage across TP150 and GND ensure its value is equal to {rtd_output}\nPress Enter to continue")
        input(f"Measure Voltage across TP65 and GND. Ensure its value is equal to {rtd_output * 5}\nPress Enter to finish")





    # Check mapping functions
    if False:
        rails_gold = list(range(1, 1021+1))
        d_pins = db.map_rails_to_driver_pins(rails_gold)
        rails = db.map_driver_pins_to_rails(d_pins)
        if rails_gold == rails:
            print("Passed rail conversion test.")
        else:
            print("FAILED rail conversion test.")

    # Dump mapping tables (iter_lcm_map)
    if False:
        for i in range(4):
            for r, d, t, h in db.iter_lcm_map(sort_pos=i):
                print(f'{r:>4d}  {d:>4d}  {t:>4d}  {h:>4d}')
            print("\n\n")

    # Read temp sensor
    if False:
        input("Set trigger")
        db.spi_read_temp_sensor()
        print('RSP FIFO 0 : {:08x}'.format(db.spi_read_rsp(0)))
        print('RSP FIFO 1 : {:08x}'.format(db.spi_read_rsp(1)))

    # Check CMD and RSP FIFO functionality
    if False:
        db.set_fields('spi', fill_cmd_fifo=1)
        db.return_all_spi_fields(with_print=True)
        for i in range(15):
            db.spi_send_cmd('standard', 'temp', 0x0000)
            print(f'Iter {i}')
            print("  CMD FIFO count: ", db.spi_get_cmd_fifo_count())
            print("  CMD FIFO is full: ", db.spi_get_cmd_fifo_is_full())
            print("  CMD FIFO headroom: ", db.spi_get_cmd_fifo_headroom())
        db.set_fields('spi', fill_cmd_fifo=0)
        time.sleep(1)
        db.return_all_spi_fields(with_print=True, exclude_rsp_fifo=True)
        print('RSP_FIFO_0: ', db.spi_read_rsp(0, 3))
        db.return_all_spi_fields(with_print=True)
        db.spi_clear_rsp_fifo(0)
        db.return_all_spi_fields(with_print=True)
        db.spi_clear_rsp_fifo(1)
        db.return_all_spi_fields(with_print=True)

    # Read single ADC sample
    if False:
        db.spi_clear_rsp_fifo(0)
        db.spi_clear_rsp_fifo(1)
        db.set_fields('spi', adc_ch_sel=0b11)
        db.return_all_spi_fields(with_print=True)
        input("Set trigger")
        db.spi_read_adcs_simple()
        print('RSP FIFO 0 : {:08x}'.format(db.spi_read_rsp(0)))
        print('RSP FIFO 1 : {:08x}'.format(db.spi_read_rsp(1)))

    # Loop through switch signals
    if False:
        db.set_fields('lcm', tp1_period=db.tp1_period_from_us(10))
        for signal in db.spi_get_all_switch_signals():
            print('Signal: ', signal)
            reg = db.spi_set_switch_data(signal)
            msg = 'Updated {} = {:04x}'
            print(msg.format(reg, db.get_fields('spi', reg)[reg]))
            regs = db.get_fields('spi', 'rx_switch', 'tx_switch')
            for k, v in regs.items():
                print(k, f'{v:04x}')
            db.spi_apply_switch_data()
            input("Applied")

    # Stream ADC samples
    if False:
        print("Setting TP1 period to 10 us.")
        db.set_fields('lcm', tp1_period=db.tp1_period_from_us(10))
        db.spi_set_adc_ch_sel('tx')
        db.spi_set_switch_data('tx_pol')
        #db.spi_set_switch_data('rx_tp1')
        db.return_all_spi_fields(with_print=True)
        #input("About to apply switch data.")
        db.spi_apply_switch_data()
        #input("Applied. About to start TP1 and POL.")
        db.set_fields('lcm', apply0=1)
        input("Check switch output and all switch inputs for backflow.")

        ts_us = 1
        db.spi_read_adcs_stream(ts_us, n_frames=63)
        rsp_0 = db.spi_read_rsp(0, n_rsp=70)
        rsp_1 = db.spi_read_rsp(1, n_rsp=70)

        print("Discard first sample, remaining 62 samples should be valid.")
        for i, (r0, r1) in enumerate(zip(rsp_0, rsp_1)):
            print("  ".join(['{:>2d}'.format(i),
                             '{:>1d}'.format(db.spi_get_rsp_is_valid(r0)),
                             '{:>1d}'.format(db.spi_get_rsp_is_valid(r1)),
                             '{:>8.3f}'.format(0.00025 * ((r0 >> 2) & 0x3fff)),
                             '{:>8.3f}'.format(0.00025 * ((r1 >> 2) & 0x3fff))]))

    # Loop over ITO amplitudes and slew rates
    if False:
        db.set_fields('lcm', apply0=1)
        for vpp in range(0, 55, 5):
            db.ito_amplitude_vpp = vpp
            input(f"ITO Vpp set to {vpp} V ({9+vpp/2} - {9-vpp/2}).\n")
        print('')
        for exp in range(9):
            db.ito_slew = int(2**exp - 1)
            input(f"ITO slew set to {db.ito_slew}.\n")

    # Check TX switch (for laser driver CI) functionality
    if False:
        db.tx_ci_a = 2.3
        print(f"TX Digipot A set to {db.tx_ci_a} V.")
        db.tx_ci_b = 4.5
        print(f"TX Digipot B set to {db.tx_ci_b} V.")
        input("Default switch is A (tx_pwr_switch = 0). Check.")
        db.set_fields('lcm', tx_pwr_switch=1)
        input("Toggled the switch. Check.")

    # Loop through observable driver output channels
    if False:
        for test_point, rail in db.hv_tp_to_rail_map.items():
            input(f'Press Enter to drive rail {rail}.')
            print(f'Driving rail {rail}. See test point "{test_point}".')
            voltages = 9 * np.ones(db.channel_count)
            voltages[rail - 1] = 6
            db.write_pattern_v(voltages)
            table_rx, table_tx = db.get_table(0)
            columns = ('r', 'd', 't', 'h', 'table[t]')
            print('{:>4s} {:>4s} {:>4s} {:>4s} {}'.format(*columns))
            for r, d, t, h in db.iter_lcm_map(sort_pos=0):
                if r < 8 or r > 1015 or 508 < r < 516:
                    c = table_tx[t]
                    print(f'{r:>4d} {d:>4d} {t:>4d} {h:>4d}     0x{c:02x}')
            print('\nNon-zero channels:')
            for t, code in enumerate(table_tx):
                if code != 0:
                    print(f'    Code at t = {t} is non-zero: 0x{code:02x}.')
            print('\n')
        input("Press Enter to stop LCM controller.")
        db.set_fields('lcm', tcon_enable=0)
        input("Press Enter to close.")

    # Check driver voltage accuracy on a single channel
    if False:
        print("WARNING: bypassing DeloreanBoard.delv_max")
        proceed = input("Proceed? [y/n]: ")
        if proceed.lower() != 'y':
            return
        db.delv_max = 99
        test_point = 'OUT_2'
        rail = db.hv_tp_to_rail_map[test_point]
        v_step = 0.5
        for v in np.arange(9, 18 + v_step, v_step):
            voltages = 9 * np.ones(db.channel_count)
            voltages[rail - 1] = v
            db.write_pattern_v(voltages)
            input(f'Driving test point "{test_point}" to {v} V.')

    # Run single short check
    if False:
        # Set ADC1 and switches to TX_POL
        db.spi_set_adc_ch_sel('tx')
        db.spi_set_switch_data('tx_pol')
        db.spi_apply_switch_data()

        # Configure TP1_PERIOD for 3 periods within fixed sampling window
        ts_us = 10
        n_frames = 63
        tp1_period_us = ts_us * (n_frames - 1) / 3
        db.set_fields('lcm', tp1_period=db.tp1_period_from_us(tp1_period_us))

        # Write onehot pattern and apply
        v_bias = 4
        test_point = 'OUT_2'
        rail = db.hv_tp_to_rail_map[test_point]
        voltages = db.v_gnd * np.ones(db.channel_count)
        voltages[rail - 1] = db.v_gnd - v_bias
        db.write_pattern_v(voltages)

        # Get data
        #   * Send 63 frames (set stream command field to 62 to send 63).
        #   * Receive 63 frames but toss first and use the rest.
        #   * The signal path of the switches divides by five before the ADC.
        db.spi_read_adcs_stream(ts_us, n_frames=n_frames-1)
        rsp_0 = db.spi_read_rsp(0, n_rsp=n_frames)
        rsp_1 = db.spi_read_rsp(1, n_rsp=n_frames)
        sns_vec = [0.00025 * (r >> 2 & 0x3fff) for i, r in enumerate(rsp_0)
                   if i != 0 and db.spi_get_rsp_is_valid(r)]
        pol_vec = [5 * 0.00025 * (r >> 2 & 0x3fff) for i, r in enumerate(rsp_1)
                   if i != 0 and db.spi_get_rsp_is_valid(r)]
        assert(len(sns_vec) == n_frames - 1)
        assert(len(pol_vec) == n_frames - 1)

        # Process data
        #   * Threshold POL so np.diff() returns -1, 0, or 1.
        #   * Average the first `window` samples before each POL edge. The
        #     window is set to a quarter of the POL period but will need to be
        #     calibrated for capacitive transience when real LCMs are used.
        #   * Check for two full TP1 periods-worth of samples. There should be
        #     three POL edges or two edges if one edge was just missed.
        pol_vec = [1 if p > 3.3 / 2 else 0 for p in pol_vec]
        pol_edges_pos = np.where(np.diff(pol_vec) ==  1)[0]
        pol_edges_neg = np.where(np.diff(pol_vec) == -1)[0]
        num_edges = len(pol_edges_pos) + len(pol_edges_neg)
        window = int(tp1_period_us / ts_us / 2)
        if delorean.DEBUG:
            print("  ".join(['{:>2s}'.format('#'),
                             '{:>8s}'.format('pol_vec'),
                             '{:>8s}'.format('sns_vec')]))
            for i, (p, s) in enumerate(zip(pol_vec, sns_vec)):
                print("  ".join([f'{i:>2d}', f'{p:>8.3f}', f'{s:>8.3f}']))
            print('')
            print('pol_edges_pos:', pol_edges_pos)
            print('pol_edges_neg:', pol_edges_neg)
            print('')
            print(f'TP1_PERIOD = {tp1_period_us} us.')
            print('window =', window)
            print('')
        check = (num_edges == 3) or (num_edges == 2 and
                min(pol_edges_pos[-1], pol_edges_neg[-1]) >= window)
        assert(check)
        sns_hi = sns_vec[pol_edges_neg[-1] - window : pol_edges_neg[-1]]
        sns_lo = sns_vec[pol_edges_pos[-1] - window : pol_edges_pos[-1]]
        avg_sns_hi = np.average(sns_hi)
        avg_sns_lo = np.average(sns_lo)
        if delorean.DEBUG:
            print('sns_hi =', sns_hi)
            print('sns_lo =', sns_lo)
            print(f'avg_sns_hi = {avg_sns_hi}')
            print(f'avg_sns_lo = {avg_sns_lo}')
            print('')
        delta_mv = 1000 * (avg_sns_hi - avg_sns_lo)
        print(f'Short check: delta = {delta_mv:.3f} mV.')


if __name__ == '__main__':
    delorean.DEBUG = True
    host = sys.argv[1]
    db = delorean.DeloreanBoard(host)

    main(db)

    db.disconnect()

'''
Top-level script to interact with the Lotus test system.
'''

import argparse
import jinja2
import os
import random
import sys
import textwrap
import yaml

import python_tools.git_utils as git
from python_tools.bsc_patgen import BSCPatternGenerator
from python_tools.zynq_api import LotusZynqAPI

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
_SETTINGS_YAML_FILE = os.path.join(git.get_git_root(),
                                'python_tools',
                                'yaml',
                                'lotus_memory_map.yml')

_SETTINGS_YAML_DUMP_FILE = 'lotus_dump.yml'

_JINJA2_TEMPLATE = textwrap.dedent('''
    {{- '# config' }}
    dwell_cnt: {{settings.dwell_cnt}}
    clks_per_interval: {{settings.clks_per_interval}}
    intervals_per_frame: {{settings.intervals_per_frame}}
    pulses_per_frame: {{settings.pulses_per_frame}}

    # coeffs[channel] is an 8-bit driver coefficient
    coeffs:
      {%- for coeff in settings.coeffs %}
      - {{'0x{coeff:02X}'.format(coeff=coeff)}}
      {%- endfor %}

''')

_DV_LIMIT = 4.5

_N_CHANNELS = 204

__args = None
__settings = None
__lotus = None
__pg = None

# -----------------------------------------------------------------------------
# Menu commands
# -----------------------------------------------------------------------------
def reset_bsc_fsm():
    '''Reset BSC controller
    '''
    __lotus.reset()

def init_bsc_fsm():
    '''Initialize BSC controller
    '''
    __lotus.init()

def _are_valid_codes(codes):
    '''Checks that the delta-V magnitudes are less than a limit.
    '''
    dvs = __pg.get_channel_delta_voltages(codes)
    return all([abs(dv) <= _DV_LIMIT for dv in dvs])

def _validate_table():
    if not _are_valid_codes(__settings['coeffs']):
        msg = "Table has delta-V beyond safe limit {}."
        msg = msg.format(i, _DV_LIMIT)
        print(msg)
        return False
    else:
        return True

def _check_switch_before_driving():
    daisy_en = __lotus.spi_get_config() >> 2 & 1
    if daisy_en == 1:
        if __args.debug:
            print("Switches are turned on.")
            ans = input("Are you sure you want to proceed? [y/n] ")
            proceed = ans == 'y'
        else:
            print("Switches are turned on. Will not proceed.")
            proceed = False
    else:
        proceed = True
    return proceed

def _check_drive_before_switching():
    if not __lotus.driver_is_done():
        if __args.debug:
            print("Driver is running or not initialized.")
            ans = input("Are you sure you want to proceed? [y/n] ")
            proceed = ans == 'y'
        else:
            print("Driver is running or not initialized. Will not proceed.")
            proceed = False
    else:
        proceed = True
    return proceed

def apply_table():
    '''Apply table to Bravo
    '''
    if not __args.debug and not _validate_table():
        return
    if not _check_switch_before_driving():
        return
    __lotus.enable(apply=1)

def stop_bsc_fsm():
    '''Stop BSC controller (HV outputs go Hi-Z)
    '''
    __lotus.stop()

def get_driver_status():
    '''Get BSC driver status
    '''
    status = __lotus.read_status()
    print("mvr_txfer_busy = {}".format(status >> 0 & 1))
    print("driver_done    = {}".format(status >> 1 & 1))

def display_config():
    '''Display config values
    '''
    k = 'dwell_cnt'
    v = __settings[k]
    s = "{:.3f} us".format(__lotus.dwell_count_to_us(v))
    print("{0}: {1} / 0x{1:x} / {2}".format(k, v, s))

    k = 'clks_per_interval'
    v = __settings[k]
    s = "{:.3f} us interval".format(v * 0.010)
    print("{0}: {1} / 0x{1:x} / {2}".format(k, v, s))

    tmp = v
    k = 'intervals_per_frame'
    v = __settings[k]
    if v == 0:
        s = 'single-shot mode'
    else:
        s = "{:.3f} us per frame".format(v * tmp * 0.010)
    print("{0}: {1} / 0x{1:x} / {2}".format(k, v, s))

    k = 'pulses_per_frame'
    v = __settings[k]
    s = "-"
    print("{0}: {1} / 0x{1:x} / {2}".format(k, v, s))

    k = 'daisy_en'
    v = __settings[k]
    if v == 0:
        s = 'disabled (powered-down)'
    else:
        s = 'enabled'
    print("{0}: {1} / {2}".format(k, v, s))

    k = 'adc0_chsel'
    v = __settings[k]
    if v == 1:
        s = '9-/18-V Current Sense'
    else:
        s = 'Bravo Resistor Sense'
    print("{0}: {1} / {2}".format(k, v, s))

    k = 'fill_cmd_fifo'
    v = __settings[k]
    if v == 1:
        s = 'fill (hold) FIFO'
    else:
        s = 'drain (release) FIFO'
    print("{0}: {1} / {2}".format(k, v, s))

def set_dwell_cnt():
    '''Set dwell count
    '''
    while True:
        try:
            us = float(input("Enter time in us: "))
            count = __lotus.dwell_count_from_us(us)
            stat = __lotus.set_dwell(count, check=True)
            __settings['dwell_cnt'] = count
        except ValueError:
            print("Please type a number.")
        except RuntimeError as e:
            print(e)
        else:
            break
    if not stat:
        print('Failed to program dwell count.')
    print('Done.')

def set_laser_clks_per_interval():
    '''Set clks_per_interval for laser
    '''
    while True:
        try:
            v = int(input("Enter value: "))
            stat = __lotus.config_laser(v,
                                        __settings['pulses_per_frame'],
                                        __settings['intervals_per_frame'],
                                        check=True)
            __settings['clks_per_interval'] = v
        except ValueError:
            print("Please type a number.")
        except RuntimeError as e:
            print(e)
        else:
            break
    if not stat:
        print('Failed to program clks_per_interval.')
    print('Done.')

def set_laser_pulses_per_frame():
    '''Set pulses_per_frame for laser
    '''
    while True:
        try:
            v = int(input("Enter value: "))
            stat = __lotus.config_laser(__settings['clks_per_interval'],
                                        v,
                                        __settings['intervals_per_frame'],
                                        check=True)
            __settings['pulses_per_frame'] = v
        except ValueError:
            print("Please type a number.")
        except RuntimeError as e:
            print(e)
        else:
            break
    if not stat:
        print('Failed to program pulses_per_frame.')
    print('Done.')

def set_laser_intervals_per_frame():
    '''Set intervals_per_frame for laser
    '''
    while True:
        try:
            v = int(input("Enter value: "))
            stat = __lotus.config_laser(__settings['clks_per_interval'],
                                        __settings['pulses_per_frame'],
                                        v,
                                        check=True)
            __settings['intervals_per_frame'] = v
        except ValueError:
            print("Please type a number.")
        except RuntimeError as e:
            print(e)
        else:
            break
    if not stat:
        print('Failed to program intervals_per_frame.')
    print('Done.')

def set_fill_cmd_fifo():
    '''Enable filling of SPI CMD FIFO
    '''
    _set_spi_config_bit(0, "Enter value (0 = drain, 1 = fill): ")

def set_adc0_chsel():
    '''Set ADC0 channel select
    '''
    _set_spi_config_bit(1, "Enter value (0 = Bravo Res, 1 = Curr Sense): ")

def set_enable_daisy_switches():
    '''Set daisy switch enable (power-on)
    '''
    _set_spi_config_bit(2, "Enter value (0 = disabled, 1 = enabled): ")

def _set_spi_config_bit(index, msg):
    config = (__settings['daisy_en'] << 2 |
              __settings['adc0_chsel'] << 1 |
              __settings['fill_cmd_fifo'])
    bit_map = ['fill_cmd_fifo', 'adc0_chsel', 'daisy_en']

    while True:
        try:
            val = int(input(msg))
            if val == 1:
                config |= 1 << index
            elif val == 0:
                config &= ~(1 << index)
            else:
                raise RuntimeError("Please enter 0 or 1.")
            __lotus.spi_set_config(config)
            check = __lotus.spi_get_config()
            stat = check == config
            __settings[bit_map[index]] = val
        except ValueError:
            print("Please type a number.")
        except RuntimeError as e:
            print(e)
        else:
            break
    if not stat:
        print('Failed to program config.')
    print('Done.')

def display_voltage_settings():
    '''Display voltage settings for channels
    '''
    for i in range(4):
        _display_voltage_settings_page(i)

def _display_voltage_settings_page(page_idx):
    p_len = 60
    lo = p_len * page_idx
    hi = min(_N_CHANNELS, p_len * (page_idx + 1))
    half = (hi - lo) // 2

    sep = ' '
    fmt_v = lambda x: "{:.3f}".format(x).rjust(7, ' ')
    codes = __settings['coeffs']
    vs = __pg.get_channel_voltages(codes)
    dvs = __pg.get_channel_delta_voltages(codes)
    max_mag = max([abs(dv) for dv in dvs])
    epsilon = 0.001
    argmax = [abs(dv) > max_mag - epsilon for dv in dvs]
    header = "{: >3s} {: <4s} {: <7s} {: <7s}".format('#', 'Code', 'Vabs', 'Vdelta')
    print((' ' * 8).join([header, header]))

    for i in range(half):
        j = i + lo
        k = i + lo + half
        line = sep.join(['{: >3d}'.format(j),
                         '0x{:02x}'.format(codes[j]),
                         fmt_v(vs[j]),
                         fmt_v(dvs[j])])
        line += ' *' if argmax[j] else '  '
        line += ' ' * 6
        line += sep.join(['{: >3d}'.format(k),
                         '0x{:02x}'.format(codes[k]),
                         fmt_v(vs[k]),
                         fmt_v(dvs[k])])
        line += ' *' if argmax[k] else '  '
        print(line)
    input("\nPress Enter to see next page.")

def _set_table_with_check(pattern):
    stat = __lotus.set_table(pattern, check=True)
    if not stat:
        print('Failed to program table.')
        return
    __settings['coeffs'] = pattern[:]

def clear_all_coeffs():
    '''Clear all channels to 0x00 (9V)
    '''
    pattern = [0] * _N_CHANNELS
    _set_table_with_check(pattern)

def flush_all_settings_to_hw():
    '''Write all settings to hardware and read back to verify.
    '''
    table = __settings['coeffs']
    stat = __lotus.set_table(table, check=True)
    if not stat:
        print('Failed to program table.')

    __lotus.set_dwell(__settings['dwell_cnt'], check=True)
    if not stat:
        print('Failed to program dwell count.')

    stat = __lotus.config_laser(__settings['clks_per_interval'],
                                __settings['pulses_per_frame'],
                                __settings['intervals_per_frame'],
                                check=True)
    if not stat:
        print('Failed to program laser config.')

    config = (__settings['daisy_en'] << 2 |
              __settings['adc0_chsel'] << 1 |
              __settings['fill_cmd_fifo'])
    __lotus.spi_set_config(config)
    check = __lotus.spi_get_config()
    stat = check == config
    if not stat:
        print('Failed to program SPI config.')
    print('Done.')

def onehot_channel_0():
    '''Onehot channel 0
    '''
    _onehot_channel(0)

def onehot_channel_1():
    '''Onehot channel 1
    '''
    _onehot_channel(1)

def onehot_channel_2():
    '''Onehot channel 2
    '''
    _onehot_channel(2)

def onehot_channel_3():
    '''Onehot channel 3
    '''
    _onehot_channel(3)

def onehot_channel_200():
    '''Onehot channel 200
    '''
    _onehot_channel(200)

def onehot_channel_201():
    '''Onehot channel 201
    '''
    _onehot_channel(201)

def onehot_channel_202():
    '''Onehot channel 202
    '''
    _onehot_channel(202)

def onehot_channel_203():
    '''Onehot channel 203
    '''
    _onehot_channel(203)

def _onehot_channel(channel):
    if not _check_switch_before_driving():
        return
    pattern = __pg.onehot_pattern(channel)
    _set_table_with_check(pattern)
    print("Done. Apply table when ready.")

def set_single_channel():
    '''Set single channel voltage
    '''
    if not _check_switch_before_driving():
        return
    while True:
        try:
            channel = int(input("Enter channel(0..203): "))
            if not 0 <= channel < _N_CHANNELS:
                raise RuntimeError("Enter value on [0,203].")
        except ValueError:
            print("Please type a number.")
        except RuntimeError as e:
            print(e)
        else:
            break
    while True:
        try:
            voltage = float(input("Enter voltage: "))
            if not 0 <= voltage <= 9:
                raise RuntimeError("Enter value on [0,9].")
        except ValueError:
            print("Please type a number.")
        except RuntimeError as e:
            print(e)
        else:
            break
    table = __lotus.get_table()
    code = __pg.get_channel_code(channel, voltage)
    table[channel] = code
    _set_table_with_check(table)
    print("Done. Apply table when ready.")

def sawtooth_pattern():
    '''Generate sawtooth with integral voltage values: (0..9 or 9..18)
    '''
    if not _check_switch_before_driving():
        return
    pattern = __pg.sawtooth_pattern()
    _set_table_with_check(pattern)
    print("Done. Apply table when ready.")

def read_onewire_temp_sensors():
    '''Read one-wire temperature sensors
    '''
    data = [__lotus.onewire_read_temp0(),
            __lotus.onewire_read_temp1(),
            __lotus.onewire_read_temp2()]
    for i, d in enumerate(data):
        if d is None:
            print("Temp #{} is INVALID".format(i))
        else:
            print("Temp #{} is {:.4f}F ({:.4f}C)".format(i, 9/5*d+32, d))

def read_spi_temp_sensor():
    '''Read SPI temperature sensor
    '''
    __lotus.spi_send_cmd(0, 1, 0x0000)
    data = __lotus.spi_read_rsp()
    print("Word = 0x{:08x}".format(data))
    print("RSP Valid = {}".format(__lotus.spi_get_rsp_is_valid(data)))
    print("RSP FIFO Count = {}".format(__lotus.spi_get_rsp_fifo_count(data)))
    print("RSP SLAVE_IDX = {}".format(__lotus.spi_get_rsp_slave_idx(data)))
    print("RSP Payload = 0x{:04x}".format(__lotus.spi_get_rsp_payload(data)))
    if data >> 31 != 1:
        print("Data is invalid")
    else:
        celsius = __lotus._int_to_signed(data & 0xfff8, 16, 7)
        print("Temp is {:.4f}F ({:.4f}C)".format(9/5*celsius+32, celsius))

def read_spi_adc0():
    '''Read SPI ADC0 (Lotus ADC)
    '''
    _read_spi_adc(0)

def read_spi_adc1():
    '''Read SPI ADC1 (Test Board ADC)
    '''
    _read_spi_adc(1)

def _read_spi_adc(adc_idx):
    slave = [0, 2][adc_idx]
    ## start acquisition
    #__lotus.spi_send_cmd(0, 0, 0x0000)
    # sample and convert
    __lotus.spi_send_cmd(0, slave, 0x0000)
    # readout
    __lotus.spi_send_cmd(0, slave, 0x0000)

    #data = __lotus.spi_read_rsp()
    #print("Word = 0x{:08x}".format(data))
    data = __lotus.spi_read_rsp()
    print("Word = 0x{:08x}".format(data))
    data = __lotus.spi_read_rsp()
    print("Word = 0x{:08x}".format(data))

    if data >> 31 != 1:
        print("Data is invalid")
    else:
        word = (data >> 2) & 0x3fff
        vltg = word * 0.00025
        print("Voltage is {:.5f}V (0x{:04x})".format(vltg, word))
        if adc_idx == 1:
            print("VT net is {:.5f}V".format(vltg * 1.5))
        elif __settings['adc0_chsel'] == 1:
            print("18V current is {:.3f}mA".format(vltg / 2 * 1e3))
        else:
            # Bravo temp sense
            pass

def test_spare_switch(slave=4):
    '''Test spare switch on test board
    '''
    for i in range(9):
        wdata = 0<<15 | 0x01<<8 | (1<<i & 0xff)
        if i == 8:
            print("\nOpening all switches (wdata = 0x{:04x})".format(wdata))
        else:
            print("\nClosing switch {} (wdata = 0x{:04x})".format(i, wdata))
        __lotus.spi_send_cmd(0, slave, wdata)
        __lotus.spi_read_rsp()

        wdata = 1<<15 | 0x01<<8
        print("Reading switch setting (wdata = 0x{:04x})".format(wdata))
        __lotus.spi_send_cmd(0, slave, wdata)
        rdata = __lotus.spi_read_rsp()
        print("Setting = 0x{:02x} (RSP = 0x{:08x})".format(rdata & 0xff, rdata))
        if i == 8:
            input("\nPress Enter to return to main menu.")
        elif i == 7:
            input("\nPress Enter to open all switches.")
        else:
            input("\nPress Enter to advance to next switch.")

def set_spare_switch_to_daisy_chain_mode(slave=4):
    '''Set spare switch to daisy-chain mode
    '''
    wdata = 0x2500
    print("\nEntering daisy-chain mode (wdata = 0x{:04x})".format(wdata))
    __lotus.spi_send_cmd(0, slave, wdata)
    rdata = __lotus.spi_read_rsp()
    print("RSP = 0x{:08x}".format(rdata))

def test_spare_switch_in_daisy_chain_mode(slave=4):
    '''Test daisy-chain mode using the spare switch
    '''
    wdata = random.randint(0, 0xffff)
    print("\nSending wdata = 0x{:04x}".format(wdata))
    __lotus.spi_send_cmd(0, slave, wdata)
    rdata = __lotus.spi_read_rsp()
    print("RSP = 0x{:08x}".format(rdata))

def set_daisy_chains_to_daisy_chain_mode(slave=3):
    '''Set daisy-chained switches to daisy-chain mode
    '''
    if not _check_drive_before_switching():
        return
    wdata = 0x2500
    print("\nEntering daisy-chain mode (wdata = 0x{:04x})".format(wdata))
    __lotus.spi_send_cmd(0, slave, wdata)
    __lotus.spi_read_rsp()

def switch_rail_to_adc(rail_idx=0):
    '''Connect a rail to the ADC. Ground all other rails.
    '''
    if not _check_drive_before_switching():
        return
    # generate bitstreams
    vt_words = [0x0000_0000] * 7
    gnd_words = [0xffff_ffff] * 7
    word_addr, bit = LotusZynqAPI.SPI_SWITCH_MAP[rail_idx]
    vt_words[word_addr] = vt_words[word_addr] | (1<<bit)
    gnd_words[word_addr] = gnd_words[word_addr] & ~(1<<bit)

    print("GND switches (MOSI_0)")
    for i, word in enumerate(gnd_words):
        print("    {}: 0x{:08x}".format(i, word))
    print("VT switches (MOSI_1)")
    for i, word in enumerate(vt_words):
        print("    {}: 0x{:08x}".format(i, word))

    # write periph memory
    for i, (lo, hi) in enumerate(zip(gnd_words, vt_words)):
        __lotus.spi_write(i, lo)
        __lotus.spi_write(i+8, hi)

    # program switches
    __lotus.spi_send_cmd(1, 3, 12)

def exec_adc_time_series(rail1=1, rail2=2):
    '''Run exec_adc_time_series hardware accelerator
    '''
    if not _check_drive_before_switching():
        return
    ts = 0.6e-6
    n_wait_cycles = round(ts/10e-9 - 18*2 - 21)
    ts_effective = 10e-9*(n_wait_cycles + 21 + 18*2)
    print("ts effective = {:.4g} s".format(ts_effective))
    rsps = __lotus.exec_adc_time_series(rail1, rail2, n_wait_cycles)
    vltgs = [(__lotus.spi_get_rsp_payload(rsp) >> 2) * 0.00025 * 1.5
             for rsp in rsps]
    max_ = max(vltgs)
    for i, v in enumerate(vltgs):
        print('    {:3d} | {:6.3f}V |'.format(i, v) + '.' * round(50*v/max_))

def exec_adc_time_series_full():
    '''Run exec_adc_time_series_full hardware accelerator
    '''
    if not _check_drive_before_switching():
        return
    ts = 0.6e-6
    n_wait_cycles = round(ts/10e-9 - 18*2 - 21)
    ts_effective = 10e-9*(n_wait_cycles + 21 + 18*2)
    print("ts effective = {:.4g} s".format(ts_effective))
    __lotus.exec_adc_time_series_full(n_wait_cycles)

def dump_settings_to_yaml():
    '''Dump settings to YAML
    '''
    t = jinja2.Template(_JINJA2_TEMPLATE)
    out = t.render(settings=__settings)
    outfile = os.path.dirname(_SETTINGS_YAML_FILE)
    outfile = os.path.join(outfile, _SETTINGS_YAML_DUMP_FILE)
    with open(outfile, 'w') as f:
        f.write(out)
    print('Done. Settings written to {}.'.format(outfile))

def parse_yaml():
    '''Parse configuration YAML file
    '''
    global __args
    global __settings
    with open(__args.yaml, 'r') as f:
        __settings = yaml.safe_load(f)

def remote_app_is_running():
    '''Check if remote app is running
    '''
    running = __lotus.remote_app_is_running()
    if running:
        print('Yes, the remote app is running.')
    else:
        print('No, the remote app is NOT running.')

def exit_script():
    '''Exit
    '''
    # raises the SystemExit exception
    sys.exit(0)

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
_CMDS = {0: (init_bsc_fsm, False),
         1: (apply_table, False),
         2: (stop_bsc_fsm, False),
         3: (reset_bsc_fsm, True),
         4: (get_driver_status, False),

         10: (display_config, False),
         11: (set_dwell_cnt, False),
         12: (set_laser_clks_per_interval, False),
         13: (set_laser_intervals_per_frame, False),
         14: (set_laser_pulses_per_frame, False),
         15: (set_fill_cmd_fifo, False),
         16: (set_adc0_chsel, False),
         17: (set_enable_daisy_switches, False),

         30: (display_voltage_settings, False),
         31: (clear_all_coeffs, False),
         32: (flush_all_settings_to_hw, False),

         40: (onehot_channel_0, True),
         41: (onehot_channel_1, True),
         42: (onehot_channel_2, True),
         43: (onehot_channel_3, True),
         44: (onehot_channel_200, True),
         45: (onehot_channel_201, True),
         46: (onehot_channel_202, True),
         47: (onehot_channel_203, True),
         48: (set_single_channel, True),
         49: (sawtooth_pattern, True),

         50: (read_onewire_temp_sensors, False),

         60: (set_daisy_chains_to_daisy_chain_mode, False),
         61: (switch_rail_to_adc, False),
         62: (exec_adc_time_series, False),
         63: (exec_adc_time_series_full, False),

         70: (read_spi_temp_sensor, False),
         71: (read_spi_adc0, False),
         72: (read_spi_adc1, False),
         73: (test_spare_switch, False),
         74: (set_spare_switch_to_daisy_chain_mode, False),
         75: (test_spare_switch_in_daisy_chain_mode, False),

         90: (dump_settings_to_yaml, False),
         91: (parse_yaml, False),
         92: (remote_app_is_running, False),

         100: (exit_script, False)}

def print_main_menu():
    print('\n' + '-' * 80)
    last = 0
    for i in sorted(_CMDS.keys()):
        fp, fp_debug_only = _CMDS[i]
        if last // 10 != i//10:
            print('')
        if not (fp_debug_only and not __args.debug):
            print('{: >3d}. {}'.format(i, fp.__doc__.strip()))
        last = i
    print('-' * 80)

def process_input(prompt):
    done = False
    while not done:
        input_str = input(prompt)
        try:
            i = int(input_str)
            fp, fp_debug_only = _CMDS[i]
            if fp_debug_only and not __args.debug:
                raise KeyError
        except ValueError:
            print("Please type a number.")
        except KeyError:
            print("Invalid value.")
        else:
            print("You chose: " + fp.__doc__)
            fp()
            done = True

def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help="Enable debug mode--NOT SAFE FOR LIQUID CRYSTALS!")
    parser.add_argument('-y', '--yaml',
                        default=_SETTINGS_YAML_FILE,
                        help="yaml file containing settings.")
    parser.add_argument('-z', '--zynq', '--host',
                        required=True,
                        help="zynq host name, e.g. 'microzed-12-34-56'")
    return parser.parse_args(argv)


def main():
    global __args
    global __settings
    global __lotus
    global __pg

    __args = parse_args(sys.argv[1:])
    if os.path.exists(__args.yaml):
        print('Parsing yaml file {}'.format(__args.yaml))
        parse_yaml()
    else:
        msg = 'YAML file not found ({}). Nothing to parse.'.format(__args.yaml)
        raise RuntimeError(msg)

    print('Connecting to Zynq host {}.'.format(__args.zynq))
    sys.stdout.flush()
    __lotus = LotusZynqAPI(__args.zynq)
    __lotus.start_remote_app()

    print('Programming memory maps.')
    flush_all_settings_to_hw()
    __lotus.spi_config_clkdivs([2, 9, 2, 2, 2])

    print('Instantiating Pattern Generator.')
    __pg = BSCPatternGenerator()
    __pg.n_channels = _N_CHANNELS
    __pg.channel_parity = [True for i in range(_N_CHANNELS)]

    print('Setup done.')
    while True:
        try:
            print_main_menu()
            process_input('Enter command #: ')
        except (SystemExit, KeyboardInterrupt):
            __lotus.shutdown_sequence()
            print('Goodbye!')
            break
        except:
            __lotus.shutdown_sequence()
            raise

if __name__ == '__main__':
    main()

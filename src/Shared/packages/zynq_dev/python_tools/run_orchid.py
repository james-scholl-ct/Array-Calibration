'''
Top-level script to interact with the Orchid test system.
'''

import argparse
import jinja2
import os
import sys
import textwrap
import yaml

import python_tools.git_utils as git
from python_tools.bsc_patgen import BSCPatternGenerator
from python_tools.zynq_api import OrchidZynqAPI

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
_SETTINGS_YAML_FILE = os.path.join(git.get_git_root(),
                                'python_tools',
                                'yaml',
                                'orchid_memory_map.yml')

_JINJA2_TEMPLATE = textwrap.dedent('''
    {{- '# config' }}
    dwell_mode00: {{settings.dwell_mode00}}
    dwell_normal: {{settings.dwell_normal}}

    # coeffs[table_idx][channel] is an 8-bit driver coefficient
    coeffs:
      {%- for table in settings.coeffs %}
      # Table {{loop.index0}}
      -
        {%- for coeff in table %}
        - {{'0x{coeff:02X}'.format(coeff=coeff)}}
        {%- endfor %}
      {%- endfor %}

''')

_DV_LIMIT = 4.5

__args = None
__settings = None
__orchid = None
__pg = None

# -----------------------------------------------------------------------------
# Menu commands
# -----------------------------------------------------------------------------
def reset_fabric():
    '''Reset fabric
    '''
    __orchid.reset()

def _are_valid_codes(codes):
    '''Checks that the delta-V magnitudes are less than a limit. Doesn't
    check for DC offsets across tables (i.e. safety of Mode 1 and Mode 3).
    '''
    dvs = __pg.get_channel_delta_voltages(codes)
    return all([abs(dv) <= _DV_LIMIT for dv in dvs])

def _validate_tables(table_list):
    valid = True
    for i in table_list:
        if not _are_valid_codes(__settings['coeffs'][i]):
            msg = "Table {} has delta-V beyond safe limit {}."
            msg = msg.format(i, _DV_LIMIT)
            print(msg)
            valid = False
    return valid

def start_mode_0():
    '''Start mode 0: TBL_0, ~TBL_0, ...
    '''
    if not __args.debug and not _validate_tables([0]):
        return
    __orchid.start(mode=0)
    input("Press Enter to stop.")
    __orchid.stop()

def start_mode_1():
    '''Start mode 1: TBL_0,  TBL_1, ...
    '''
    if not __args.debug and not _validate_tables([0, 1]):
        return
    __orchid.start(mode=1)
    input("Press Enter to stop.")
    __orchid.stop()

def start_mode_2():
    '''Start mode 2: TBL_0, ~TBL_0, TBL_1, ~TBL_1, ...
    '''
    if not __args.debug and not _validate_tables([0, 1]):
        return
    __orchid.start(mode=2)
    input("Press Enter to stop.")
    __orchid.stop()

def start_mode_3():
    '''Start mode 3: TBL_0,  TBL_1, TBL_2,  TBL_3, ...
    '''
    if not __args.debug and not _validate_tables([0, 1, 2, 3]):
        return
    __orchid.start(mode=3)
    input("Press Enter to stop.")
    __orchid.stop()

def stop():
    '''Stop current mode
    '''
    __orchid.stop()

def display_config():
    '''Display config values
    '''
    keys = ('dwell_mode00', 'dwell_normal')
    for k in sorted(keys):
        v = __settings[k]
        time_ns = __orchid.dwell_count_to_ns(v)
        print("{0}: {1} / 0x{1:x} / {2:.3f} us".format(k, v, time_ns/1000))

def set_dwell_mode00():
    '''Set dwell for mode 0
    '''
    while True:
        try:
            ns = float(input("Enter time in ns: "))
            count = __orchid.dwell_count_from_ns(ns)
            __orchid.set_dwell_mode00(count)
            __settings['dwell_mode00'] = count
        except ValueError:
            print("Please type a number.")
        except RuntimeError as e:
            print(e)
        else:
            break
    rdata = __orchid._send_read_range(68, 1)
    if rdata[0] != __settings['dwell_mode00']:
        print('Failed to program dwell count.')
    print('Done.')

def set_dwell_normal():
    '''Set dwell for modes 1, 2, and 3
    '''
    while True:
        try:
            ns = float(input("Enter time in ns: "))
            count = __orchid.dwell_count_from_ns(ns)
            __orchid.set_dwell_normal(count)
            __settings['dwell_normal'] = count
        except ValueError:
            print("Please type a number.")
        except RuntimeError as e:
            print(e)
        else:
            break
    rdata = __orchid._send_read_range(69, 1)
    if rdata[0] != __settings['dwell_normal']:
        print('Failed to program dwell count.')
    print('Done.')

def display_voltage_settings_0():
    '''Display voltage settings for table 0
    '''
    _display_voltage_settings(0)

def display_voltage_settings_1():
    '''Display voltage settings for table 1
    '''
    _display_voltage_settings(1)

def display_voltage_settings_2():
    '''Display voltage settings for table 2
    '''
    _display_voltage_settings(2)

def display_voltage_settings_3():
    '''Display voltage settings for table 3
    '''
    _display_voltage_settings(3)

def _display_voltage_settings(table_idx):
    codes = __settings['coeffs'][table_idx]
    sep = ' '
    fmt_v = lambda x: "{:.3f}".format(x).rjust(7, ' ')
    vs = __pg.get_channel_voltages(codes)
    dvs = __pg.get_channel_delta_voltages(codes)
    max_mag = max([abs(dv) for dv in dvs])
    epsilon = 0.001
    argmax = [abs(dv) > max_mag - epsilon for dv in dvs]
    header = "{: >2s} {: <4s} {: <7s} {: <7s}".format('#', 'Code', 'Vabs', 'Vdelta')
    print((' ' * 8).join([header, header]))
    for i in range(len(codes)//2):
        line = sep.join(['{: >2d}'.format(i),
                         '0x{:02x}'.format(codes[i]),
                         fmt_v(vs[i]),
                         fmt_v(dvs[i])])
        line += ' *' if argmax[i] else '  '
        line += ' ' * 6
        line += sep.join(['{: >2d}'.format(32+i),
                         '0x{:02x}'.format(codes[32+i]),
                         fmt_v(vs[32+i]),
                         fmt_v(dvs[32+i])])
        line += ' *' if argmax[32+i] else '  '
        print(line)
    input("\nPress Enter to return to the main menu.")

def _set_table_with_check(idx, pattern):
    __orchid.set_table(idx, pattern)
    rdata = __orchid.get_table(idx)
    if pattern != rdata:
        print('Failed to program table {}.'.format(idx))
        return
    __settings['coeffs'][idx] = pattern[:]

def clear_all_coeffs():
    '''Clear all channels to 0x00 (9V)
    '''
    for i in range(4):
        pattern = [0]*64
        _set_table_with_check(i, pattern)

def set_channel_voltage_all_tables():
    '''Set channel voltage for all tables
    '''
    while True:
        try:
            channel = int(input("Enter channel index (0..63): "))
            if not 0 <= channel <= 63:
                raise ValueError
            voltage = float(input("Enter voltage: "))
            code = __pg.get_channel_code(channel, voltage)
        except ValueError:
            print("Invalid value.")
        except RuntimeError as e:
            print(str(e))
        else:
            break
    for i in range(4):
        __orchid.set_channel(i, channel, code)
        rdata = __orchid.get_channel(i, channel)
        if code != rdata:
            print('Failed to program table {}, channel {}.'.format(i, channel))
            return
        __settings['coeffs'][i][channel] = code

def max_safe_deltav_all():
    '''Mode 0 and 2 char: set all 64 channels to the maximally safe delta-V
    '''
    print("Max safe delta voltage is {}V".format(_DV_LIMIT))
    voltages = [9]*64
    for i, v in enumerate(voltages):
        delta = _DV_LIMIT if i < 32 else -_DV_LIMIT
        if i % 2 == 1:
            voltages[i] = v + delta
    table = __pg.get_channel_codes(voltages)
    for i in range(2):
        _set_table_with_check(i, table)

def max_safe_deltav_half():
    '''Mode 0 and 2 char: set 32 channels to the maximally safe delta-V
    '''
    print("Max safe delta voltage is {}V".format(_DV_LIMIT))
    voltages = [9]*64
    for i, v in enumerate(voltages):
        delta = _DV_LIMIT if i < 32 else -_DV_LIMIT
        if i & 0x3 >= 2:
            voltages[i] = v + delta
    table = __pg.get_channel_codes(voltages)
    for i in range(2):
        _set_table_with_check(i, table)

def max_safe_deltav_quarter():
    '''Mode 0 and 2 char: set 16 channels to the maximally safe delta-V
    '''
    print("Max safe delta voltage is {}V".format(_DV_LIMIT))
    voltages = [9]*64
    for i, v in enumerate(voltages):
        delta = _DV_LIMIT if i < 32 else -_DV_LIMIT
        if i & 0x7 >= 4:
            voltages[i] = v + delta
    table = __pg.get_channel_codes(voltages)
    for i in range(2):
        _set_table_with_check(i, table)

#def max_safe_deltav_all_mode_1():
#    '''Mode 1 char: set all table 0 and 1 channels to the maximally safe delta-V
#    '''
#    print("Max safe delta voltage is {}V".format(_DV_LIMIT))
#    voltages_0 = [9]*64
#    voltages_1 = [9]*64
#    for i, v in enumerate(voltages_0):
#        if i not in (0, 31, 32, 63):
#            delta = _DV_LIMIT if i < 32 else -_DV_LIMIT
#            if i % 2 == 1:
#                voltages_0[i] = v + delta
#                voltages_1[i] = v - delta
#    for i, voltages in enumerate((voltages_0, voltages_1)):
#        table = __pg.get_channel_codes(voltages)
#        __orchid.set_table(i, table)
#        rdata = __orchid.get_table(i)
#        if table != rdata:
#            print('Failed to program table {}.'.format(i))
#        __settings['coeffs'][i] = table[:]

def flush_all_settings_to_hw():
    '''Write all settings to hardware and read back to verify.
    '''
    for i, table in enumerate(__settings['coeffs']):
        __orchid.set_table(i, table)
        rdata = __orchid.get_table(i)
        if table != rdata:
            print('Failed to program table {}.'.format(i))
    __orchid.set_dwell_mode00(__settings['dwell_mode00'])
    __orchid.set_dwell_normal(__settings['dwell_normal'])
    rdata = __orchid._send_read_range(68, 2)
    if rdata[0] != __settings['dwell_mode00']:
        print('Failed to program dwell count for mode 0.')
    if rdata[1] != __settings['dwell_normal']:
        print('Failed to program dwell count for normal mode.')
    print('Done.')

def onehot_channel_0():
    '''Onehot channel 0 such that {0 -> VDD or VSS, o.w. -> VDD/2}
    '''
    _onehot_channel(0)

def onehot_channel_31():
    '''Onehot channel 31 such that {31 -> VDD or VSS, o.w. -> VDD/2}
    '''
    _onehot_channel(31)

def onehot_channel_32():
    '''Onehot channel 32 such that {32 -> VDD or VSS, o.w. -> VDD/2}
    '''
    _onehot_channel(32)

def onehot_channel_63():
    '''Onehot channel 63 such that {63 -> VDD or VSS, o.w. -> VDD/2}
    '''
    _onehot_channel(63)

def _onehot_channel(channel):
    pat = __pg.onehot_pattern(channel)
    _use_mode1_as_debug_hack(pat)

def sawtooth_pattern():
    '''Generate sawtooth with integral voltage values: (0..9 or 9..18)
    '''
    pat = __pg.sawtooth_pattern()
    _use_mode1_as_debug_hack(pat)

def _use_mode1_as_debug_hack(pattern):
    for i in (0, 1):
        _set_table_with_check(i, pattern)
    print("Starting h/w. Press enter to stop.")
    __orchid.start(mode=1)
    input()
    __orchid.stop()

def dump_settings_to_yaml():
    '''Dump settings to YAML
    '''
    t = jinja2.Template(_JINJA2_TEMPLATE)
    out = t.render(settings=__settings)
    outfile = os.path.dirname(_SETTINGS_YAML_FILE)
    outfile = os.path.join(outfile, 'orchid_dump.yml')
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
    running = __orchid.remote_app_is_running()
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
_CMDS = {0: (start_mode_0, False),
         1: (start_mode_1, False),
         2: (start_mode_2, False),
         3: (start_mode_3, False),
         4: (stop, True),
         5: (reset_fabric, True),

         10: (display_config, False),
         11: (set_dwell_mode00, False),
         12: (set_dwell_normal, False),

         20: (display_voltage_settings_0, False),
         21: (display_voltage_settings_1, False),
         22: (display_voltage_settings_2, False),
         23: (display_voltage_settings_3, False),

         30: (clear_all_coeffs, False),
         31: (set_channel_voltage_all_tables, False),
         #32: (set_channel_voltage_table_0, False),
         #33: (set_channel_voltage_table_1, False),
         #34: (set_channel_voltage_table_2, False),
         #35: (set_channel_voltage_table_3, False),

         40: (max_safe_deltav_all, False),
         41: (max_safe_deltav_half, False),
         42: (max_safe_deltav_quarter, False),
         #41: (max_safe_deltav_all_mode_1, False),

         50: (flush_all_settings_to_hw, False),

         60: (onehot_channel_0, True),
         61: (onehot_channel_31, True),
         62: (onehot_channel_32, True),
         63: (onehot_channel_63, True),
         64: (sawtooth_pattern, True),

         #80: (validate_tables, False),

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
    global __orchid
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
    __orchid = OrchidZynqAPI(__args.zynq)
    __orchid.start_remote_app()

    print('Programming memory map.')
    flush_all_settings_to_hw()

    print('Instantiating Pattern Generator.')
    __pg = BSCPatternGenerator()

    print('Setup done.')
    while True:
        try:
            print_main_menu()
            process_input('Enter command #: ')
        except (SystemExit, KeyboardInterrupt):
            __orchid.shutdown_sequence()
            print('Goodbye!')
            break
        except:
            __orchid.shutdown_sequence()
            raise

if __name__ == '__main__':
    main()

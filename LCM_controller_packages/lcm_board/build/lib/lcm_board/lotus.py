import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings as wn
from lcm_board.voltage_translator_fft_bravo import VoltageTranslatorFFT
from matplotlib.colors import LogNorm
from python_tools import himax_model
from python_tools.zynq_api import LotusZynqAPI
from scipy.optimize import curve_fit
from numato_relay import NumatoRelayController

class InvalidVoltageError(Exception):
    pass


class LotusBoard:
    # __instances is a dict of {'addr': object} in this class
    __instances = {}

    def __new__(cls,
                addr):
        """This function is how the class returns new instances"""
        # Put addresses to lower case to eliminate ambiguity
        addr = addr.upper()
        # If there is no instrument at this address, make one
        if not addr in cls.__instances:
            cls.__instances[addr] = super(LotusBoard, cls).__new__(cls)
            cls.__instances[addr].__initialized = False
        # Either way, there is now an instrument at addr.  Return it.
        return cls.__instances[addr]

    def __init__(self,
                 addr):
        """
        Args:
            addr: address of the board, e.g. microzed-3a-a3-14
        """

        # If already initialized, return without doing anything
        if self.__initialized:
            return
        self.__initialized = True
        # Tell the instrument what to do when program exits.
        # atexit.register(self.shutdown)
        print("Initialized " + addr)

        self.addr = addr
        self.connect()
        self.v_min = 0
        self.v_max = 9
        self.v_gnd = 9
        self.max_del_v = 6.1
        # self.vpt = old_vt.VoltagePatternTranslator(self.v_min, self.v_max)
        self.vpt = VoltageTranslatorFFT(self.v_min, self.v_max, self.max_del_v)

        # Initialization for short checking board
        self.r_mux = 30  # Skip calibration, for now
        self.r_test = 1910
        self.v_source = 0.2
        self.r_rail_estimate = 50
        self.sampling_interval = 1.5e-6
        self.clock_config = 2
        self.r_leak_threshold = 100000  # Threshold for disabling driving of shorted channel
        self.channel_count = 204

        # Initialization for Himax driver
        vgma = {'vgma1': 18,
                'vgma2': 18,
                'vgma9': 9,
                'vgma10': 9,
                'vgma11': 9,
                'vgma12': 9,
                'vgma19': 0,
                'vgma20': 0}
        self.h = himax_model.HimaxModel(vgma)

        # Connect to ITO, and shut off
        for port in ('com7', 'com8'):
            try:
                self.ito = NumatoRelayController(port, 2).relay_1
                break
            except SerialException:
                continue
        else:
            raise SerialException('Could not find ITO relay port')

        # Run initial short check for future reference, if not checking live
        print('Running initial short check!')
        df_impedance_results, df_impedance_summary, fig = self.map_impedance(mode='steady', only_neighbors=True,
                                                                             show_map=False)
        self.df_shorts = df_impedance_results[df_impedance_results.r_leak < self.r_leak_threshold]
        self.short_filter = np.ones(self.channel_count)
        for index, row in self.df_shorts.iterrows():
            self.short_filter[index] = 0
        self.number_of_shorts = len(self.df_shorts)
        print(f'Number of shorts: {self.number_of_shorts}')
        
        # Turn ITO on
        self.ito.on()


    def connect(self):
        """Connects to the Zynq host"""
        self._conn = LotusZynqAPI(self.addr)
        self._conn.start_remote_app(force_restart=True)
        self._conn.init()
        # Set clock speeds
        self._conn.spi_config_clkdivs([2, 9, 2, 2, 2])
        self.frequency_config()
        print("New connection to " + self.addr)
        print("Connected to " + self.addr)

    def voltage_on(self):
        """Turns on the voltage output of the Himax driver"""
        daisy_en = self._conn.spi_get_config() >> 2 & 1
        if daisy_en == 0:
            self._conn.enable()
        else:
            raise RuntimeError('Switches still closed! (Daisy enable = {})  Shutting down.'.format(daisy_en))

    def voltage_off(self):
        """Turns off the voltage output of the Himax driver"""
        self._conn.stop()

    def frequency_config(self,
                         frequency_hz_himax=2000,
                         ito_tc=0,
                         ito_out_of_phase=True,
                         prog_trigger_mode=1):
        """
        Sets the AC frequency configuration of the Himax and ITO driver

        Args:
            frequency_hz_himax (float): desired AC Himax frequency, in Hz
            ito_tc (int): number of additional Himax pol clock cycles before ITO pol flip
            ito_invert (int): one for invert, zero for no invert
            prog_trigger_mode (int): one for pulse, zero for toggle

        """
        if ito_out_of_phase:
            ito_invert = 0
        else:
            ito_invert = 1
        time_us = 1 / (2 * frequency_hz_himax) * 1e6
        dwell = self._conn.dwell_count_from_us(time_us)
        frequency_dict = {
            'dwell_cnt': dwell,
            'ito_tc': ito_tc,
            'ito_invert': ito_invert,
            'prog_trigger_mode': prog_trigger_mode
        }

        self._conn.set_bsc_config(**frequency_dict)

    def read_pattern_v(self, tx_or_rx = 'tx') -> np.ndarray:
        """
        Returns the current voltage output of the Himax driver

        Args:
            tx_or_rx: dummy input to match Delorean board

        Returns:
            v (numpy): array of v values

        """
        v = self._conn.get_table()
        v = self.h.get_voltages(v)
        v = np.asarray(v)
        return v

    def read_pattern_delv(self, tx_or_rx = 'tx') -> np.ndarray:
        """Read delv pattern from the Himax driver.

        Args:
            tx_or_rx: dummy input to match Delorean board

        Returns:
            delv: np.ndarray of delv values
        """
        v = self.read_pattern_v()
        delv = self.vpt.vtodelv(v)
        return delv

    def write_pattern_delv(self,
                           delv,
                           tx_or_rx='tx',
                           diff_order=0,
                           max_depth=4,
                           num_bests=4,
                           error_factor=1,
                           voltage_factor=0.01,
                           target_v_avg=0,
                           fft_factor=0,
                           sidelobe_factor=0):
        """Writes a delv pattern to the Himax driver, correcting for shorts

        Args:
            delv: array of delv values.  Any element that is identically 0 will be preserved.
            diff_order: desired diff_order for fourier optimization
            max_depth: depth of each tree
            num_bests: number of best results to keep from each tree
            error_factor: how much to weight voltage errors
            voltage_factor: how much to weight mean voltage
            target_v_avg: what voltage to aim for in weighting algorithm
            fft_factor: how much to weight maximizing the fourier component at diff_order
            sidelobe_factor: how much to weight minimizing the next-highest fourier component

        Returns:
            applied_delv: array of applied delv values
            applied v: array of the applied v values
            """
        delv_copy = copy.deepcopy(delv)
        # If we have a trivial all zeros dv pattern, generate trivial v pattern
        if np.sum(delv_copy) == 0:
            v_pattern = np.ones_like(delv_copy) * self.v_gnd
        # Otherwise, continue
        else:
            # (v_pattern, clipping_cases) = self.vpt.delvtov_anchor(self.short_filter * delv_copy)
            target_del_v = self.short_filter * delv_copy
            v_pattern = self.vpt.delvtov(target_del_v,
                                         diff_order=diff_order,
                                         max_depth=max_depth,
                                         num_bests=num_bests,
                                         error_factor=error_factor,
                                         voltage_factor=voltage_factor,
                                         target_v_avg=target_v_avg,
                                         fft_factor=fft_factor,
                                         sidelobe_factor=sidelobe_factor)

        code_pattern = self.h.get_codes(v_pattern)
        applied_v = np.asarray(self.h.get_voltages(code_pattern))
        applied_delv = self.vpt.vtodelv(applied_v)
        # self._conn.set_table(code_pattern, check=False)
        self._conn.ds_set_apply(np.asarray(code_pattern), check=False)
        # self._conn.ds_set_apply_waitack()
        return applied_delv, applied_v

    def write_pattern_v(self, v_pattern, tx_or_rx = 'tx') -> [np.ndarray, np.ndarray]:
        """
        Writes a previously saved v pattern to the Himax driver

        Args:
            v_pattern (numpy): array of v_pattern values
            tx_or_rx: dummy input to match Delorean board

        Returns:
            applied_delv: array of applied delv values
            applied v: array of the applied v values

        """
        # test if voltages are in range
        if np.any(np.greater(v_pattern, self.v_max)):
            raise InvalidVoltageError('An input voltage is greater than allowed range!')
        elif np.any(np.less(v_pattern, self.v_min)):
            raise InvalidVoltageError('An input voltage is less than allowed range!')

        code_pattern = self.h.get_codes(v_pattern)
        applied_v = np.asarray(self.h.get_voltages(code_pattern))
        applied_delv = self.vpt.vtodelv(applied_v)
        self._conn.ds_set_apply(np.asarray(code_pattern), check=False)
        return applied_delv, applied_v

    def load_rom_pattern(self,
                         pattern_number):
        """
        Recalls a stored voltage pattern from the ROM

        Args:
            pattern_number (int): pattern number, ranging from 0 to 128

        """
        self._conn.load_table(pattern_number)

    def impedance_model(self,
                        adctime,
                        r_rail,
                        c_rail,
                        r_leak):
        adctime = np.array(adctime)
        r_x = self.r_test + self.r_mux + r_rail
        tau = c_rail * r_leak * r_x / (r_leak + r_x)
        voltages = (self.v_source / r_x) * (
                self.r_mux + r_rail + (self.r_test * r_leak / (r_x + r_leak)) * (1 - np.exp(-adctime / tau)))
        return voltages

    def get_pair_impedance(self,
                           rail_signal,
                           rail_ground,
                           mode='steady'):

        if mode == 'transient':
            times, voltages = self.get_adc_timeseries(rail_signal,
                                                      rail_ground,
                                                      mode=mode)
            popt, pcov = curve_fit(self.impedance_model,
                                   times,
                                   voltages,
                                   p0=[self.r_rail_estimate, 10E-9, 10000])
            perr = np.sqrt(np.diag(pcov))
            for i in range(len(perr)):
                if abs(perr[i] / popt[i]) > 0.4:
                    wn.warn('Model fitting error > 40%.  Likely incorrect!')
            impedance = {'r_rail': popt[0],
                         'c_rail': popt[1],
                         'r_leak': popt[2],
                         'fitting_error': np.average(perr / popt),
                         'raw_data': {'times': times,
                                      'voltages': voltages}}
        elif mode == 'steady':
            times, voltages = self.get_adc_timeseries(rail_signal,
                                                      rail_ground,
                                                      mode=mode)
            voltage = np.average(voltages)
            if voltage > 0.999 * self.v_source:
                voltage = 0.999 * self.v_source
            r_leak = self.r_test / (1 - voltage / self.v_source) - self.r_test - self.r_mux - self.r_rail_estimate
            impedance = {'r_rail': None,
                         'c_rail': None,
                         'r_leak': r_leak,
                         'fitting_error': None,
                         'raw_data': {'times': times,
                                      'voltages': voltages}}

        return impedance

    def map_impedance(self,
                      mode='steady',
                      only_neighbors=True,
                      fig=None,
                      map_axis=None,
                      imp_axis=None,
                      show_map=True):

        # Turn off driver outputs if active
        driver_was_active = False
        i = 0
        while not self._conn.driver_is_done():
            print('Himax driver voltage output is on! Shutting output off for short checking.')
            self.voltage_off()
            driver_was_active = True
            i += 1
            if i > 3:
                raise RuntimeError("Failed to shut Himax driver voltage off!")

        # Run full short check matrix with hardware acceleration
        rails = {}
        rails['signal'] = []
        rails['ground'] = []
        rails['r_rail'] = []
        rails['c_rail'] = []
        rails['r_leak'] = []
        rails['fitting_error'] = []
        wait_cycles = self.get_wait_cycles()
        if only_neighbors:
            output_path_abs = self._conn.exec_adc_time_series_nearest_neighbors(wait_cycles)
        else:
            output_path_abs = self._conn.exec_adc_time_series_full(wait_cycles)
        if mode == 'steady':
            rsps_buffer_count = 10  # Take only last 10 measurements, which should be steady state
        else:
            rsps_buffer_count = 63  # Take all measurements
            raise RuntimeError('Transient mode still in dev, run steady state!')
        df = pd.read_csv(output_path_abs, header=None)
        os.remove(output_path_abs)
        # Index from one, not zero
        df[0] += 1
        df[1] += 1
        df = df.reset_index(drop=True)
        for i in df.index:
            rails['signal'].append(df.iloc[i, 0])
            rails['ground'].append(df.iloc[i, 1])
            # Read ADC voltages, throwing out invalid last value
            voltages = [0.00025 * (int(r, 16) >> 2 & 0x3fff) for r in df.iloc[i, -(rsps_buffer_count + 1):-1]]
            # Scale voltages per calibration
            voltages_scaled = self.adc_voltage_scaling(voltages)
            # For steady state:
            avg_voltage = np.average(voltages_scaled)
            if avg_voltage > 0.999 * self.v_source:
                avg_voltage = 0.999 * self.v_source
            rails['r_leak'].append(max(
                [self.r_test / (1 - avg_voltage / self.v_source) - self.r_test - self.r_mux - self.r_rail_estimate,
                 10]))
            rails['r_rail'].append(None)
            rails['c_rail'].append(None)
            rails['fitting_error'].append(None)

        df_results = pd.DataFrame.from_dict(rails)
        df_results['r_leak_oom'] = np.log10(df_results['r_leak'])

        # Summarize data
        dict_summary = {}
        dict_summary['<100 Ohm'] = (df_results.r_leak_oom < 2).sum()
        dict_summary['100-1000 Ohm'] = ((df_results.r_leak_oom > 2) & (df_results.r_leak_oom < 3)).sum()
        dict_summary['1-10 kOhm'] = ((df_results.r_leak_oom > 3) & (df_results.r_leak_oom < 4)).sum()
        dict_summary['>10 kOhm'] = (df_results.r_leak_oom > 4).sum()
        df_summary = pd.DataFrame.from_dict(dict_summary, orient='index').T

        if show_map:
            # Build figure and map
            fig = plt.figure(figsize=(9, 9), tight_layout=True)
            ax_nearest = plt.subplot2grid((3, 3), (0, 0), colspan=3)
            ax_hist = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
            ax_map = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
            ax_map.set_xlabel('Driven Rail')
            ax_map.set_ylabel('Grounded Rail')
            ax_map.set_title('Leakage Resistance (Ohms)')
            cmap = plt.get_cmap('hot')
            cmap.set_bad('lightgray')
            map_image = ax_map.imshow(np.full((self.channel_count + 2, self.channel_count + 2), -1), origin='lower',
                                      cmap=cmap,
                                      norm=LogNorm(vmin=10, vmax=100000))
            array = np.full((self.channel_count + 2, self.channel_count + 2), -1)
            for i in range(len(rails['r_leak'])):
                array[rails['signal'][i], rails['ground'][i]] = rails['r_leak'][i]
            map_image.set_data(array)
            fig.colorbar(map_image, ax=ax_map)

            # Build histogram
            n, bins, patches = ax_hist.hist(df_results.r_leak_oom, bins=[0, 1, 2, 3, 4, 5, 10])
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = bin_centers - min(bin_centers)
            col /= max(col)
            cmap = plt.get_cmap('hot', 5)
            for idx_p, p in enumerate(patches):
                plt.setp(p, 'facecolor', cmap(idx_p - 1))
            if only_neighbors:
                hist_text_offset = 1
            else:
                hist_text_offset = 50
            for i, n in enumerate(n):
                if i > 0:
                    ax_hist.text(bins[i] + 0.5, n + hist_text_offset, str(round(n)), horizontalalignment='center')
            ax_hist.set_facecolor('lightgray')
            ax_hist.set_xlim([1, 6])
            ax_hist.set_xticks([1, 2, 3, 4, 5, 6])
            ax_hist.set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$'])
            ax_hist.set_xlabel('Leakage Resistance (Ohms)')
            ax_hist.set_ylabel('Counts')
            ax_hist.set_title('Channel Distribution')

            # Build nearest neighbor plot
            df_results_nearest = df_results[np.abs(df_results.signal - df_results.ground) % self.channel_count == 1]
            ax_nearest.scatter(df_results_nearest.signal, df_results_nearest.r_leak, color='k')
            ax_nearest.set_yscale('log')
            ax_nearest.set_xlabel('Signal Rail')
            ax_nearest.set_ylabel('Leakage Resistance (Ohms)')
            ax_nearest.set_facecolor('lightgray')
            ax_nearest.set_title('Nearest Neighbor Leakage')

        # Restore voltage, if necessary
        if driver_was_active:
            print('Restoring Himax driver voltage output.')
            self.voltage_on()

        # Update initial short check results
        self.df_shorts = df_results[df_results.r_leak < self.r_leak_threshold]
        self.short_filter = np.ones(self.channel_count)
        for index, row in self.df_shorts.iterrows():
            self.short_filter[index] = 0

        return df_results, df_summary, fig

    def plot_impedance_timeseries(self, rail_signal, rail_ground, mode='steady'):

        # Get data
        impedance = self.get_pair_impedance(rail_signal, rail_ground, mode=mode)
        # Plot data
        fig, ax = plt.subplots(figsize=(9, 9), tight_layout=True)
        ax.set_xlabel('Time')
        ax.set_ylabel('Voltage')
        ax.plot(impedance['raw_data']['times'], impedance['raw_data']['voltages'], label='Measurement')
        if mode == 'transient':
            ax.plot(impedance['raw_data']['times'], self.impedance_model(
                impedance['raw_data']['times'],
                impedance['r_rail'],
                impedance['c_rail'],
                impedance['r_leak']
            ), label='Model')

        ax.legend(loc='best')
        ax.set_title('Fitted Impedance Model: R_leak = ' + str(round(impedance['r_leak'])))

    def get_wait_cycles(self):

        alpha = self.sampling_interval - 170e-9 * (self.clock_config + 1)
        alpha = alpha / 10e-9 - 4 - self.clock_config
        return round(alpha)

    def get_adc_timeseries(self, rail_signal, rail_ground, mode='steady'):
        '''Takes two lcm rails, which are indexed on [1,204], and returns
        the times series ....

            Args:
                * rail_signal: integer on [1,204]
                * rail_ground: integer on [1, 204]
            Returns:
                * times: array of points in seconds
                * voltages: array of points in Volts
        '''
        n_wait_cycles = self.get_wait_cycles()
        valid_rsps = self._conn.exec_adc_time_series(rail_signal - 1, rail_ground - 1, n_wait_cycles)

        # Get payload of each element and convert to analog voltage
        # (Digital word is 14 bits and each bit is worth 0.25 mV).
        times = [i * self.sampling_interval for i in range(len(valid_rsps))]
        voltages = [0.00025 * (r >> 2 & 0x3fff) for r in valid_rsps]
        if mode == 'steady':
            times = times[-10:]
            voltages = voltages[-10:]
        if mode == 'transient':
            # Throw out first few values
            times = times[5:]
            voltages = voltages[5:]

        voltages_corrected = self.adc_voltage_scaling(voltages)

        return times, voltages_corrected

    def adc_voltage_scaling(self, adc_voltages):

        adc_voltage_example = [0.235, 0.266, 0.298, 0.368, 0.541, 0.712, 1.059, 1.492, 1.921, 2.872, 3.651]
        vt_example = [0.00253, 0.00445, 0.00638, 0.0104, 0.02045, 0.033, 0.0504, 0.0754, 0.1002, 0.1552, 0.2003]

        return np.interp(adc_voltages, adc_voltage_example, vt_example)

    def plot_current_patterns(self, tx_or_rx = 'tx'):
        """Plots the current voltage and delv patterns stored in Himax driver

        Args:
            tx_or_rx: dummy input to match Delorean board

        """
        v = self.read_current_pattern()
        delv = self.vpt.vtodelv(v)
        plt.close('all')
        plt.subplot(211)
        plt.plot(np.arange(len(delv)), delv, 'ob', markersize=4)
        plt.xlabel('gap number')
        plt.ylabel('delta V')
        plt.subplot(212)
        plt.plot(np.arange(len(v)), v, '-ok')
        plt.xlabel('channel number')
        plt.ylabel('V')
        plt.tight_layout()

    def shutdown(self):
        """Executes shutdown sequence for Lotus"""
        self.voltage_off()
        self.ito.off()
        self._conn.stop_laser()
        self._conn.shutdown_sequence()
        print('Lotus shutdown sequence complete.')

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

import lcm_board.voltage_translator_fft_bravo as vtfft
import lcm_board.voltage_pattern as voltage_pattern
import time
import numpy as np
import random
from timeit import default_timer as timer
import itertools


class AlgorithmTester:
    """This class tests a VoltageTranslator algorithm, and plots the performance and results"""
    def __init__(self, v_noise=0.0):
        self.v_noise = v_noise
        self.test_del_v_patterns = None
        min_voltage = 0
        max_voltage = 9
        max_del_v = 6.1
        self.vt = vtfft.VoltageTranslatorFFT(min_voltage, max_voltage, max_del_v)
        self.vpg = voltage_pattern.VoltagePatternGenerator()

    @staticmethod
    def load_test_v_data(path):
        """Loads a set of mustang voltage patterns into memory"""
        csv_dataframe = pd.read_csv(path)
        test_v_patterns = []
        for diff_order in csv_dataframe.columns[1:]:
            test_v_patterns += [csv_dataframe[diff_order].to_list()]
        return test_v_patterns

    def generate_random_del_v_data(self, m=-1):
        # v_noise adds noise in range [-v_noise, +v_noise]
        test_del_v_pattern = self.vpg.logistic(m=m)[0]
        for i in range(0, len(test_del_v_pattern)):
            test_del_v_pattern[i] += (random.random()*2-1) * self.v_noise
        return test_del_v_pattern

    def run_timed_algorithm(self,
                            target_pattern,
                            diff_order,
                            max_depth,
                            num_bests,
                            error_factor=1,
                            voltage_factor=0,
                            fft_factor=10,
                            sidelobe_factor=25,
                            max_voltage=None):
        t_start = timer()
        v_pattern = self.vt.delvtov(target_pattern,
                               diff_order=diff_order,
                               max_depth=max_depth,
                               num_bests=num_bests,
                               error_factor=error_factor,
                               voltage_factor=voltage_factor,
                               fft_factor=fft_factor,
                               sidelobe_factor=sidelobe_factor,
                               max_voltage=max_voltage)
        runtime = timer() - t_start
        return v_pattern, runtime

    def plot_del_v(self, axis, target_del_v, actual_del_v):
        axis.plot(target_del_v, 'o')
        axis.plot(actual_del_v)
        axis.set_xlabel('Gap')
        axis.set_ylabel('Delta V')
        axis.legend(['Target', 'Actual'])
        axis.xaxis.labelpad = -2
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    def plot_v(self, axis, actual_v, original_v=None):
        axis.plot(actual_v)
        axis.set_xlabel('Channel')
        axis.set_ylabel('Voltage')
        axis.xaxis.labelpad = -2
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        avg_v = np.array(actual_v).mean()
        left, right = axis.get_xlim()
        bottom, top = axis.get_ylim()
        label_x = 0.85*(right-left) + left
        label_y = 0.88*(top-bottom) + bottom
        label = f'Avg: {avg_v:0.2f}'
        if original_v is not None:
            original_avg_v = np.array(original_v).mean()
            label_y = 0.78 * (top - bottom) + bottom
            label = f'Original avg: {original_avg_v:0.2f}V\nNew avg: {avg_v:0.2f}V'
        axis.annotate(label, (label_x, label_y))

    def error_hist(self, axis, errors):
        nonzero_errors = []
        for error in errors:
            if abs(error) > 10E-6:
                # ignore errors smaller than 10uV as meaningless
                nonzero_errors += [abs(error)]
        logbins = np.logspace(-2, 0.8, 20)
        axis.hist(nonzero_errors, bins=logbins)
        axis.set_xscale('log')
        axis.set_xlabel('Voltage Error')
        axis.set_ylabel('Quantity')
        # axis.set_xlim([1E-2,5])
        avg_err = abs(np.array(errors)).mean()
        max_err = abs(np.array(errors)).max()
        count = len(nonzero_errors)
        left, right = axis.get_xlim()
        bottom, top = axis.get_ylim()
        label_x = 0.3 * (right - left) + left
        label_y = 0.58 * (top - bottom) + bottom
        label = f'Avg: {1000*avg_err:0.2f}mV\nMax: {1000*max_err:0.2f}mV\nQty: {count}'
        axis.annotate(label, (label_x, label_y))

    def plot_v_spectrum(self, axis, fft_results, m):
        axis.plot(np.arange(len(fft_results[1:])) + 1, fft_results[1:])
        axis.axvline(x=m, c='r', ls='--')
        left, right = axis.get_xlim()
        bottom, top = axis.get_ylim()
        label_x = 0.8 * (right - left) + left
        label_y = 0.7 * (top - bottom) + bottom
        sidelobes = np.delete(fft_results, [0, m, 204-m])
        label = f'Order {m}: {fft_results[m]:0.2f}\nPeak sidelobe: {sidelobes.max():0.2f}'
        axis.annotate(label, (label_x, label_y))

    def plot_one_output(self, depth, num_bests, m=10):
        error_factor = 1
        voltage_factor = 0
        fft_factor = 10
        sidelobe_factor = 0
        target_del_v = self.generate_random_del_v_data(m=m)
        voltage_pattern, runtime = self.run_timed_algorithm(target_del_v,
                                                            m,
                                                            depth,
                                                            num_bests,
                                                            error_factor=error_factor,
                                                            voltage_factor=voltage_factor,
                                                            fft_factor=fft_factor,
                                                            sidelobe_factor=sidelobe_factor)
        actual_del_v, error, fft_results = self.vt.assess_results(target_del_v, voltage_pattern)
        fig1, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(9, 8)) # , gridspec_kw={'height_ratios': [4, 1]})
        title = f'Results of Voltage Translator Algorithm\nDepth={depth}, Keeps={num_bests}, Runtime={1000*runtime:0.1f}ms'
        title += f'\nWeighting: Err: {error_factor}, V: {voltage_factor}, FFT peak: {fft_factor}, Sidelobe: {sidelobe_factor} '
        fig1.suptitle(title)
        self.plot_del_v(ax0, target_del_v, actual_del_v)
        self.plot_v(ax1, voltage_pattern) #, original_v)
        self.error_hist(ax2, error)
        self.plot_v_spectrum(ax3, fft_results, m)
        return voltage_pattern

        # Compare the algorithm's results with the patterns used for Mustang
        #    Do this by generating a histogram of voltage change after optimization
        # Make a set of plots analyzing algorithm response to weighting parameters
        #   -plot average error and average voltage as a function of weight ratios
        #   -Implement error weighting based on phase, not voltage

    def get_comparison_data(self,
                            diff_orders,
                            depths=[4],
                            num_bests_list=[4],
                            error_factors=[1],
                            voltage_factors=[0],
                            fft_factors=[10],
                            sidelobe_factors=[0],
                            max_voltages = [9]):
        """Runs the target algorithm and returns results for every combination of input parameters."""
        datalist = []
        num_rails = 204
        for order in diff_orders:
            target_del_v = self.generate_random_del_v_data(m=order)
            condition_set = itertools.product(depths,
                                              num_bests_list,
                                              error_factors,
                                              voltage_factors,
                                              fft_factors,
                                              sidelobe_factors,
                                              max_voltages)
            for condition in list(condition_set):
                depth, num_bests, error_factor, voltage_factor, fft_factor, sidelobe_factor, max_voltage = condition
                v_pattern, runtime = self.run_timed_algorithm(target_del_v,
                                                         order,
                                                         depth,
                                                         num_bests,
                                                         error_factor=error_factor,
                                                         voltage_factor=voltage_factor,
                                                         fft_factor=fft_factor,
                                                         sidelobe_factor=sidelobe_factor,
                                                         max_voltage=max_voltage)
                actual_del_v, error, fft_results = self.vt.assess_results(target_del_v, v_pattern)
                avg_err = abs(np.array(error)).mean()
                max_err = abs(np.array(error)).max()
                avg_v = abs(np.array(v_pattern)).mean()
                fft_peak = fft_results[order]
                side_lobe_list = np.delete(fft_results, [0, order, num_rails - order])
                highest_sidelobe = side_lobe_list.max()
                data = [order, depth, num_bests, error_factor, voltage_factor, fft_factor, sidelobe_factor, max_voltage]
                data += [avg_err, max_err, avg_v, fft_peak, highest_sidelobe]
                datalist += [data]
        df = pd.DataFrame(datalist)
        names = ['order', 'depth', 'num_bests', 'error_factor', 'voltage_factor', 'fft_factor', 'sidelobe_factor', 'max_voltage']
        names += ['avg_err', 'max_err', 'avg_v', 'fft_peak', 'highest_sidelobe']
        df.columns = names
        return df

    def compare_with_mustang(self, depth_limit, num_bests_limit):
        test_del_v_patterns = []
        # original_avg_vs = []
        for test_pattern in self.test_v_patterns:
            test_del_v_patterns += [self.v_to_del_v(test_pattern)]
            # original_avg_vs += [abs(np.array(test_pattern)).mean()]
        # original_avg_v = abs(np.array(original_avg_vs)).mean()
        self.test_algorithm(depth_limit, num_bests_limit, test_del_v_patterns)

    def test_algorithm(self, depth_limit, num_bests_limit, test_del_v_patterns=None, ):
        if test_del_v_patterns == None:
            self.test_del_v_patterns = self.generate_random_del_v_data(self.v_noise)
            test_del_v_patterns = self.test_del_v_patterns
        df = self.get_comparison_data(test_del_v_patterns, depth_limit, num_bests_limit)
        #plt.figure()
        fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(9, 8))
        title = f'Voltage Translator Parameter Study\nInitial Algorithm\nLogistic Functions from m=1 to m=110, {self.v_noise:.2f}V of noise'
        fig2.suptitle(title)
        self.plot_param_vs_depth(ax3, df, 'avg_error', 'Avg. |Error| (mV)')
        self.plot_param_vs_depth(ax4, df, 'avg_voltage', 'Avg. |Voltage| (V)')
        #ax4.axhline(y=6, c='r', ls='--')
        self.plot_param_vs_depth(ax5, df, 'runtime', 'Runtime (ms)')
        ax5.set_yscale('log')

        depths = [2,3,4,5,6]
        bests = [1,2,4]
        for depth in depths:
            fig3, (ax6, ax7, ax8) = plt.subplots(3, 1, figsize=(9, 8))
            title = f'Voltage Translator FOV results\nInitial Algorithm'
            fig3.suptitle(title)
            self.plot_param_vs_order(ax6, df, 'avg_error', 'Avg. |Error| (mV)', [depth], bests)
            self.plot_param_vs_order(ax7, df, 'max_err', 'Max |Error| (V)', [depth], bests)
            self.plot_param_vs_order(ax8, df, 'avg_voltage', 'Avg. |Voltage| (V)', [depth], bests)
        return df

    def plot_across_fov(self,
                        diff_orders,
                        depths,
                        num_bests,
                        error_factors,
                        voltage_factors,
                        fft_factors,
                        sidelobe_factors,
                        max_voltages):
        df = self.get_comparison_data(diff_orders,
                                    depths,
                                    num_bests,
                                    error_factors,
                                    voltage_factors,
                                    fft_factors,
                                    sidelobe_factors,
                                    max_voltages)
        fig3, (ax6, ax7, ax8, ax9, ax10) = plt.subplots(5, 1, figsize=(9, 9))
        title = f'Voltage Translator FOV results\nFFT Algorithm - Effect of Max V - Logit patterns'
        title += f'\nWeighting: Err: {error_factors[0]}, V: {voltage_factors[0]}, FFT peak: {fft_factors[0]}, Sidelobe: (sweep)'
        fig3.suptitle(title)
        self.plot_param_vs_order(ax6, df, 'avg_err', 'Avg. |Error| (mV)', depths, sidelobe_factors)
        self.plot_param_vs_order(ax7, df, 'max_err', 'Max |Error| (V)', depths, sidelobe_factors)
        self.plot_param_vs_order(ax8, df, 'avg_v', 'Avg. |Voltage| (V)', depths, sidelobe_factors)
        self.plot_param_vs_order(ax9, df, 'fft_peak', 'FFT Peak', depths, sidelobe_factors)
        self.plot_param_vs_order(ax10, df, 'highest_sidelobe', 'Highest Sidelobe', depths, sidelobe_factors)

    def plot_param_vs_depth(self, axis, df, key, axis_label):
        xtick_list = [' ']
        data = []
        units = 1
        if not key == 'avg_voltage':
            units = 1000
        for depth in df.depth.unique():
            for n in df.num_bests.unique():
                subset = df.loc[(df['depth'] == depth) & (df['num_bests']==n)]
                multiplied_data = units*np.array(subset[key].to_list())
                data.append(multiplied_data)
                xtick_list.append(f'{depth},{n}')
                #axis.scatter(df['depth'],df[key], marker='_')
                #legend_list += [f'N={num_bests}']
        axis.boxplot(data)
        axis.set_xlabel('Depth, Num Bests')
        axis.set_ylabel(axis_label)
        axis.set_xticks(np.arange(len(xtick_list)))
        axis.set_xticklabels(xtick_list)
        #axis.legend(legend_list)
        axis.xaxis.labelpad = -2
        #axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    def plot_param_vs_order(self, axis, df, key, axis_label, depths, max_voltages):
        legend_list = []
        units = 1
        if key == 'avg_err':
            units = 1000
        for depth in depths:
            for sidelobe_factor in sidelobe_factors:
                subset = df.loc[(df['depth'] == depth) & (df['sidelobe_factor'] == sidelobe_factor)]
                multiplied_data = units * np.array(subset[key].to_list())
                # if key == 'highest_sidelobe':
                #     axis.plot(subset['order'], np.array(subset['fft_peak'].to_list()))
                #     alt_legend = ['Peak', 'Highest Sidelobe']
                axis.plot(subset['order'], multiplied_data)
                legend_list += [f'w_sidelobe: {sidelobe_factor}']
        axis.set_ylabel(axis_label)
        # axis.xaxis.labelpad = -3
        # if key == 'avg_err':
        #     axis.set_yscale('log')
        axis.legend(legend_list, loc='upper right', prop={'size': 5})
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if not key == 'highest_sidelobe':
            axis.xaxis.set_ticklabels([])
        else:
            axis.set_xlabel('Diffraction Order')


if __name__ == '__main__':
    # mustang_data_path = r'..\voltage_translator_test_data\LCM000033_Rx_Configuration.csv'
    tester = AlgorithmTester(v_noise=0.0)
    t_start = time.time()
    voltage_pattern = tester.plot_one_output(depth=3, num_bests=3, m=101)
    diff_orders = range(1, 110)
    depths = [3]
    num_bests = [3]
    error_factors = [1]
    voltage_factors = [0.01]
    fft_factors = [1]
    sidelobe_factors = [0, 1]
    max_voltages = [9]
    # tester.plot_across_fov(diff_orders, depths, num_bests, error_factors, voltage_factors, fft_factors, sidelobe_factors, max_voltages)
    # path = r'G:\Shared drives\Engineering\LCM\LCM Driver\Delta V to V Algorithm\Fourier Test Patterns\mid.csv'
    # np.savetxt(path, voltage_pattern)
    print(f'Processing completed in {time.time()-t_start}s')




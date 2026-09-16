import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
#from lotus import voltage_pattern
import lcm_board.voltage_translator_delta as vpt
import lcm_board.voltage_pattern as voltage_pattern
import time
import numpy as np
import random
from timeit import default_timer as timer
import itertools


plt.ion()


class AlgorithmTester:
    """This class tests a VoltageTranslator algorithm, and plots the performance and results"""
    def __init__(self, v_noise=0.0):
        self.v_noise = v_noise
        self.test_del_v_patterns = None
        min_voltage = 0
        max_voltage = 9
        self.max_del_v = 4.0
        self.vt = vpt.VoltageTranslator(max_del_v=self.max_del_v,
                                        error_factor=1,
                                        voltage_factor=0,
                                        max_depth=2,
                                        num_bests=2,
                                        aggressive=False,
                                        soft_seam_factor=1.0)
        self.vpg = voltage_pattern.VoltagePatternGenerator(channels=1021,
                                                           delv_min=0,
                                                           delv_max=self.max_del_v)

    def assess_results(self, target_del_v_pattern, v_pattern):
        """Calculates some details about the algorithm results"""
        actual_del_v = self.vt.vtodelv(v_pattern)
        error = -1*np.ones(len(target_del_v_pattern))
        for i in range(len(v_pattern)):
            error[i] = target_del_v_pattern[i] - actual_del_v[i]
        fft_results = np.absolute(np.fft.fft(v_pattern)/len(v_pattern))
        return actual_del_v, error, fft_results

    def generate_random_del_v_data(self, m=-1, fraction_shorted=0.00):
        # v_noise adds noise in range [-v_noise, +v_noise]
        # fraction shorted ranges from 0 to 1.
        vpg = voltage_pattern.VoltagePatternGenerator(channels=1021,
                                                      delv_min=0,
                                                      delv_max=self.max_del_v)
        test_del_v_pattern = vpg.logistic(m=m)[0]
        short_filter = [1]*len(test_del_v_pattern)
        for i in range(0, len(test_del_v_pattern)):
            test_del_v_pattern[i] += (random.random()*2-1) * self.v_noise
            if random.random() < fraction_shorted:
                short_filter[i] = 0
        return test_del_v_pattern, short_filter

    def full_test(self, aggressive_mode=False):
        max_del_v_list = [3]
        orders = np.arange(10, 550, 1)
        shorting_list = [0.80]
        for max_del_v in max_del_v_list:
            print('-----------------------------------', flush=True)
            print(f'Testing max del v of {max_del_v}', flush=True)
            self.max_del_v = max_del_v
            vt = vpt.VoltageTranslator(max_del_v=max_del_v, aggressive=aggressive_mode)
            for fraction in shorting_list:
                print(f'     Testing with {100*fraction} percent shorts ', flush=True, end ="")
                # max_del_v = 0
                for order in orders:
                    # if order%20 == 0:
                    #     print('.', flush=True, end ="")
                    print(f'{order}', flush=True)
                    target_pattern, short_filter = self.generate_random_del_v_data(m=order, fraction_shorted=fraction)
                    # try:
                    v_pattern = vt.delvtov(target_pattern, short_filter)
                    # max_del_v = max(max_del_v, max(target_pattern))
                    # except Exception as e:
                    #     print(e)
                # print(f' Max del v: {max_del_v}')
                print('')

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
        short_filter = [1]*len(target_pattern)
        vt = vpt.VoltageTranslator(max_del_v=self.max_del_v,
                                        error_factor=error_factor,
                                        voltage_factor=voltage_factor,
                                        max_depth=max_depth,
                                        num_bests=num_bests,
                                        soft_seam_factor=0.3)
        t_start = timer()
        v_pattern = vt.delvtov(target_pattern, short_filter)
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
        sidelobes = np.delete(fft_results, [0, m, 1021-m])
        label = f'Order {m}: {fft_results[m]:0.2f}\nPeak sidelobe: {sidelobes.max():0.2f}'
        axis.annotate(label, (label_x, label_y))
        axis.set_ylabel('FFT amplitude')

    def plot_one_output(self, depth, num_bests, m=10, roll=0):
        error_factor = 1
        voltage_factor = 0
        fft_factor = 0
        sidelobe_factor = 0
        target_del_v, _ = self.generate_random_del_v_data(m=m)
        target_del_v = np.roll(target_del_v, roll)
        voltage_pattern, runtime = self.run_timed_algorithm(target_del_v,
                                                            m,
                                                            depth,
                                                            num_bests,
                                                            error_factor=error_factor,
                                                            voltage_factor=voltage_factor,
                                                            fft_factor=fft_factor,
                                                            sidelobe_factor=sidelobe_factor)
        actual_del_v, error, fft_results = self.assess_results(target_del_v, voltage_pattern)
        fig1, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 8))
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
        num_rails = 1021
        for order in diff_orders:
            target_del_v, short_filter = self.generate_random_del_v_data(m=order)
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
                actual_del_v, error, fft_results = self.assess_results(target_del_v, v_pattern)
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

    def test_algorithm(self, depth_limit, num_bests_limit, test_del_v_patterns=None, ):
        if test_del_v_patterns == None:
            self.test_del_v_patterns, short_filter = self.generate_random_del_v_data(self.v_noise)
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
        fig3, (ax6, ax7) = plt.subplots(2, 1, figsize=(9, 9))
        title = f'Voltage Translator FOV results\nFFT Algorithm - Effect of Max V - Logistic patterns'
        title += f'\nWeighting: Err: {error_factors[0]}, V: {voltage_factors[0]}, num_bests: {num_bests[0]}'  # ', FFT peak: {fft_factors[0]}, Sidelobe: (sweep)'
        fig3.suptitle(title)
        self.plot_param_vs_order(ax6, df, 'avg_err', 'Avg. |Error| (mV)', depths, sidelobe_factors)
        self.plot_param_vs_order(ax7, df, 'max_err', 'Max |Error| (V)', depths, sidelobe_factors)
        # self.plot_param_vs_order(ax8, df, 'avg_v', 'Avg. |Voltage| (V)', depths, sidelobe_factors)
        # self.plot_param_vs_order(ax9, df, 'fft_peak', 'FFT Peak', depths, sidelobe_factors)
        # self.plot_param_vs_order(ax10, df, 'highest_sidelobe', 'Highest Sidelobe', depths, sidelobe_factors)

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
                legend_list += [f'depth: {depth}']
        axis.set_ylabel(axis_label)
        # axis.xaxis.labelpad = -3
        # if key == 'avg_err':
        #     axis.set_yscale('log')
        axis.legend(legend_list, loc='upper right', prop={'size': 5})
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if not key == 'max_err':
            axis.xaxis.set_ticklabels([])
        else:
            axis.set_xlabel('Diffraction Order')

    def generate_all_delv(self, depth=2, num_bests=2, soft_seams=1.0, enable_roll=False):
        vt = vpt.VoltageTranslator(max_del_v=self.max_del_v,
                                   error_factor=1.0,
                                   voltage_factor=0.0,
                                   max_depth=depth,
                                   num_bests=num_bests,
                                   soft_seam_factor=soft_seams)
        vpg = voltage_pattern.VoltagePatternGenerator(wavelength=914,
                                                      phi_i=-70,
                                                      channels=1021,
                                                      pitch=300,
                                                      delv_min=0,
                                                      delv_max=self.max_del_v)
        orders = np.arange(0, vpg.maxm + 1).astype(int)
        df_v = pd.DataFrame(columns=orders)
        df_delv = pd.DataFrame(columns=orders)
        for i in orders:
            if i % 10 == 0:
                print(f'calculating order {i}')
            if i == 0:
                target_delv_pattern = np.zeros(1021) * 9
            else:
                target_delv_pattern, plot_pattern = vpg.clipped_ramp(m=i, ramp_fraction=0.5, flip=False)
            if enable_roll:
                target_delv_pattern = np.roll(target_delv_pattern, np.random.randint(1021))
            v = vt.delvtov(target_delv_pattern, np.ones(1021))
            actual_delv = vt.vtodelv(v)
            df_v[i] = v
            df_delv[i] = actual_delv
        suffix = ''
        if enable_roll:
            suffix = f'_roll'
        df_v.to_csv(path_or_buf=f'../delv_test_data/v_d{depth}_n{num_bests}_s{soft_seams}{suffix}.csv')
        df_delv.to_csv(path_or_buf=f'../delv_test_data/delv_d{depth}_n{num_bests}_s{soft_seams}{suffix}.csv')

    def plot_delta_v(self, seam_list, enable_roll):
        fig, axes = plt.subplots(len(seam_list), 2)
        depth = 4
        num_bests = 4
        suffix = ''
        if enable_roll:
            suffix = f'_roll'
        fig.suptitle(f'Impact of Soft Seams\ndepth: {depth} num_bests: {num_bests} random_roll: {enable_roll} ',
                     fontsize=10)
        for i, seam_val in enumerate(seam_list):
            delv_data = pd.read_csv(f'../delv_test_data/delv_d{depth}_n{num_bests}_s{seam_val}{suffix}.csv')
            delv_data = delv_data.drop(columns=['Unnamed: 0'])
            delv_data['mean'] = delv_data.mean(axis=1)
            axes[i, 0].plot(delv_data['mean'])
            axes[i, 0].set_ylim((0, 2.8))
            # axes[i, 0].set_title(f'soft_seams={seam_val}', fontsize=10)
            axes[i, 1].plot(delv_data['mean'])
            axes[i, 1].set_ylim((0, 2.8))
            axes[i, 1].set_xlim((980, 1030))
            last_del_v = delv_data['mean'][1020]
            axes[i, 1].annotate(f'avg last delV = {last_del_v:.02f}', (1000, .1), fontsize=8)
            # axes[i, 1].set_yticks([])
            axes[i, 0].set_ylabel(f'soft_seams={seam_val}\naverage delta V', fontsize=8)


if __name__ == '__main__':
    # mustang_data_path = r'..\voltage_translator_test_data\LCM000033_Rx_Configuration.csv'
    tester = AlgorithmTester(v_noise=0.0)
    t_start = time.time()
    # voltage_pattern = tester.plot_one_output(depth=2, num_bests=2, m=80)
    # tester.full_test(aggressive_mode=True)
    tester.plot_one_output(depth=4, num_bests=4, m=5, roll=75)
    # diff_orders = range(10, 500, 5)
    # depths = [2]
    # num_bests = [4]
    # error_factors = [1]
    # voltage_factors = [0]
    # fft_factors = [0]
    # sidelobe_factors = [0]
    # max_voltages = [18]
    # tester.plot_across_fov(diff_orders, depths, num_bests, error_factors, voltage_factors, fft_factors, sidelobe_factors, max_voltages)
    # path = r'G:\Shared drives\Engineering\LCM\LCM Driver\Delta V to V Algorithm\Fourier Test Patterns\mid.csv'
    # np.savetxt(path, voltage_pattern)
    # for soft_param in [1.0, 0.5, 0.25]:
    #     tester.generate_all_delv(depth=4, num_bests=4, soft_seams=soft_param, enable_roll=True)
    tester.plot_delta_v(seam_list=[1.0, 0.5, 0.25], enable_roll=True)
    print(f'Processing completed in {time.time()-t_start}s')






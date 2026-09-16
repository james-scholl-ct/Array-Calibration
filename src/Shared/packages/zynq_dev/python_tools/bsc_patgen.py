import numpy as np

from python_tools.himax_model import HimaxModel


class BSCPatternGenerator:
    def __init__(self):
        vgma = {'vgma1': 18, 'vgma2': 18,
                'vgma9':  9, 'vgma10': 9,
                'vgma11': 9, 'vgma12': 9,
                'vgma19': 0, 'vgma20': 0}
        self.hm = HimaxModel(vgma)
        self.n_channels = 64
        self.channel_parity = [i >= 32 for i in range(self.n_channels)]
        self.pol = 0

    def is_highside(self, channel_idx):
        if self.pol == 1:
            return self.channel_parity[channel_idx]
        else:
            return not self.channel_parity[channel_idx]

    def get_channel_code(self, channel_idx, channel_voltage):
        high = self.is_highside(channel_idx)
        if high and channel_voltage < self.hm.vdd/2.:
            msg = "Invalid voltage ({}) for channel ({})."
            msg = msg.format(channel_voltage, channel_idx)
            raise RuntimeError(msg)
        elif not high and channel_voltage > self.hm.vdd/2.:
            msg = "Invalid voltage ({}) for channel ({})."
            msg = msg.format(channel_voltage, channel_idx)
            raise RuntimeError(msg)
        return self.hm.get_codes([channel_voltage])[0]

    def get_channel_voltage(self, channel_idx, channel_code):
        high = self.is_highside(channel_idx)
        return self.hm.get_voltages([channel_code], high)

    def get_channel_codes(self, channel_voltages):
        if len(channel_voltages) != self.n_channels:
            msg = "Invalid list length ({}).".format(len(channel_voltages))
            raise RuntimeError(msg)
        return [self.get_channel_code(i, v) 
                for i, v in enumerate(channel_voltages)]

    def get_channel_voltages(self, channel_codes):
        if len(channel_codes) != self.n_channels:
            msg = "Invalid list length ({}).".format(len(channel_codes))
            raise RuntimeError(msg)
        high_vec = [self.is_highside(i) for i, _ in enumerate(channel_codes)]
        return self.hm.get_voltages(channel_codes, high_vec)

    def get_channel_delta_voltages(self, channel_codes):
        voltages = self.get_channel_voltages(channel_codes)
        voltages.append(voltages[0])
        return [voltages[i+1] - voltages[i] for i in range(self.n_channels)]

    def onehot_pattern(self, channel):
        return [self.hm.MAX_CODE if i == channel else self.hm.MIN_CODE
                for i in range(self.n_channels)]

    def sawtooth_pattern(self):
        # this function relies on the odds and evens being grouped
        voltages_lo = [v for v in range(int(self.hm.vdd+1))
                       if v <= self.hm.vdd/2]
        voltages_hi = [v for v in range(int(self.hm.vdd+1))
                       if v >= self.hm.vdd/2]
        codes_lo = self.hm.get_codes(voltages_lo)
        codes_hi = self.hm.get_codes(voltages_hi)
        len_lo = len(codes_lo)
        len_hi = len(codes_hi)
        return [codes_hi[i%len_hi] if self.is_highside(i) else codes_lo[i%len_lo]
                for i in range(self.n_channels)]

    def get_valid_steering_angles(self):
        theta_inc = -70
        theta_min = -60
        theta_max =  60
        wavelength = 905
        width_channel = 300
        #FIXME
        #n_channels = self.n_channels
        n_channels = 128
        width_tile = width_channel * n_channels

        ## gleb's m
        #patterns_per_tile = n_channels / n_channels_in_pattern
        ## gleb's P
        #width_pattern = n_channels_in_pattern * width_channel
        #arg = wavelength / width_pattern - np.sin(-theta_inc * np.pi/180.)
        #if not -1 <= arg <= 1:
        #    msg = 'Invalid n_channels_in_pattern ({}). '.format(n_channels_in_pattern)
        #    msg += 'Arg is {}.'.format(arg)
        #    raise RuntimeError(msg)
        #theta_out = (180./np.pi) * -np.arcsin(arg)
        #if not theta_min < theta_out < theta_max:
        #    msg = 'Invalid n_channels_in_pattern ({}). '.format(n_channels_in_pattern)
        #    msg += 'theta_out is {}.'.format(theta_out)
        #    raise RuntimeError(msg)
        #return theta_out

        patterns_per_tile = np.arange(1, n_channels+1)
        width_pattern = width_tile / patterns_per_tile
        arg = wavelength / width_pattern - np.sin(-theta_inc * np.pi/180.)
        filt = (-1 <= arg) & (arg<=1)
        patterns_per_tile = patterns_per_tile[filt]
        width_pattern = width_pattern[filt]
        arg = arg[filt]

        theta_out = (180. / np.pi) * -np.arcsin(arg)
        filt = (theta_out < theta_max) & (theta_out > theta_min)
        patterns_per_tile = patterns_per_tile[filt]
        width_pattern = width_pattern[filt]
        arg = arg[filt]
        theta_out = theta_out[filt]

        ret_dict = {'patterns_per_tile': patterns_per_tile,
                    'width_pattern': width_pattern,
                    'channels_per_pattern': width_pattern / width_channel,
                    'theta_out': theta_out}
        return theta_out, ret_dict

    def _get_modulation_deltav(self, width_pattern):
        width_channel = 300
        # FIXME
        #n_channels = self.n_channels
        n_channels = 128
        delta_V_min = 0
        delta_V_max = 5

        x = np.arange(1,n_channels+1)
        slope = (delta_V_max - delta_V_min)/(width_pattern/width_channel)
        delta_V = slope * x
        delta_V = np.mod(delta_V, delta_V_max - delta_V_min) + delta_V_min
        return delta_V

    def get_modulation_voltages(self, i):
        # FIXME
        #n_channels = self.n_channels
        n_channels = 128
        #V_max = self.vdd/2
        V_max = 7.5
        V_min = 0
        V_range = V_max - V_min

        angles, a_dict = self.get_valid_steering_angles()
        delta_V = self._get_modulation_deltav(a_dict['width_pattern'][i])


def gleb():
    theta_inc = -70
    theta_min = -60
    theta_max =  60

    # wavelength in nm
    wavelength = 905
    # unit cell pitch in nm
    W_unit = 300
    N_channels = 128
    W_tile = W_unit*N_channels

    V_max = 7.5
    V_min = 0
    V_range = V_max - V_min

    delta_V_min_array = np.linspace(0,1,4)
    delta_V_max = 5

    m = np.arange(1,N_channels+1)

    P = W_tile / m
    arg = wavelength / P - np.sin(-theta_inc * np.pi/180.)
    filt = (-1 <= arg) & (arg<=1)
    P = P[filt]
    arg = arg[filt]

    theta_out = (180. / np.pi) * -np.arcsin(arg)
    filt = (theta_out < theta_max) & (theta_out > theta_min)
    P = P[filt]
    arg = arg[filt]
    theta_out = theta_out[filt]


    x = np.arange(1,N_channels+1)

    sum_delta_V_error = np.zeros((N_channels,4))
    n_delta_V_error = np.zeros((N_channels,4))
    delta_V_actual_all = np.zeros((N_channels, N_channels,4))
    delta_V_all = np.zeros((N_channels, N_channels,4))
    V_all = np.zeros((N_channels, N_channels,4))

    #for k in range(len(delta_V_min_array)):
    for k in [0]:
        delta_V_min = delta_V_min_array[k]

        #for j in range(len(theta_out)):
        for j in [0]:
            slope = (delta_V_max - delta_V_min)/(P[j]/W_unit)
            delta_V = slope*x
            delta_V = np.mod(delta_V, delta_V_max - delta_V_min) + delta_V_min

            V = np.zeros((len(x)+1, 1))
            V[0] = V_max

            for i in range(len(delta_V)-1):
                V_up = V[i] + delta_V[i]
                V_down = V[i] - delta_V[i]

                if V_up <= V_max and V_down >= V_min:
                    if abs(V_up - V_max) < abs(V_down - V_min):
                        V[i+1] = V_up
                    else:
                        V[i+1] = V_down
                elif V_up < V_max and V_down < V_min:
                    V[i+1] = V_up
                elif V_up > V_max and V_down > V_min:
                    V[i+1] = V_down
                elif V_up >= V_max and V_down <= V_min:
                    if abs(V_up - V_max) < abs(V_down - V_min):
                        V[i+1] = V_max
                    else:
                        V[i+1] = V_min
                else:
                    # error case, to see if any points didn't meet the conditions above
                    V[i+1] = -20

            # HeRE
            delta_V_actual = abs(np.diff(V, axis=0))
            delta_V_actual = delta_V_actual.ravel()
            delta_V_error = delta_V_actual.ravel() - delta_V

            sum_delta_V_error[j, k] = np.sum(abs(delta_V_error))
            n_delta_V_error[j, k] = np.sum(abs(delta_V_error) > 0.01)

            delta_V_actual_all[:, j, k] = delta_V_actual
            delta_V_all[:, j, k] = delta_V
            V_all[:, j, k] = V[:-1].ravel()

    delta_V_error_all = delta_V_all - delta_V_actual_all

    g = {}
    g['P'] = P
    g['theta_out'] = theta_out
    g['sum_delta_V_error'] = sum_delta_V_error[0,0]
    g['n_delta_V_error'] = n_delta_V_error[0,0]
    g['delta_V_actual_all'] = delta_V_actual_all[:,0,0]
    g['delta_V_all'] = delta_V_all[:,0,0]
    g['V_all'] = V_all[:,0,0]
    return g

    #clf
    #
    #% plot modulation pattern for particular angle and delta_V_min
    #theta_select = 10;
    #k_idx = 1;
    #
    #[~, theta_idx] = min(abs(theta_out - theta_select));
    #theta_actual = theta_out(theta_idx);
    #
    #subplot 211
    #hold off
    #plot(delta_V_all(:,theta_idx, k_idx))
    #hold all
    #plot(delta_V_actual_all(:,theta_idx, k_idx))
    #plot(delta_V_error_all(:,theta_idx, k_idx))
    #
    #xlabel('Slot number')
    #ylabel('\DeltaV')
    #ylim([0,6.5])
    #lgd = legend('Desired \DeltaV', 'Actual \DeltaV', '\DeltaV error');
    #plot_title = ['\theta_{out} = ', num2str(theta_actual, '%.1f'), '\circ,  V_{range} = ', num2str(V_range), ' V', ',  \DeltaV_{min} = ', num2str(delta_V_min_array(k_idx))];
    #title(plot_title)
    #%lgd.Location = 'northoutside'
    #
    #subplot 212
    #plot(V_all(:,theta_idx, k_idx))
    #xlabel('Rail number')
    #ylabel('V')
    #
    #%%
    #
    #subplot 411
    #plot(theta_out', sum_delta_V_error'/N_channels)
    #xlabel('Output angle \theta (deg)')
    #title('Average \DeltaV error per channel for all \DeltaV_{min}')
    #legend '\DeltaV_{min} = 0 V' '\DeltaV_{min} = 0.33 V' '\DeltaV_{min} = 0.67 V' '\DeltaV_{min} = 1 V'
    #subplot 412
    #plot(theta_out', min(sum_delta_V_error')/N_channels)
    #xlabel('Output angle \theta (deg)')
    #ylabel('Average \DeltaV error')
    #title('Average \DeltaV error per channel for optimal \DeltaV_{min}')
    #
    #subplot 413
    #plot(theta_out, n_delta_V_error)
    #xlabel('Output angle \theta (deg)')
    #ylabel('Num channels with \DeltaV error')
    #title('Number of channels with \DeltaV error for all \DeltaV_{min}')
    #
    #subplot 414
    #plot(theta_out, min(n_delta_V_error'))
    #xlabel('Output angle \theta (deg)')
    #ylabel('Num channels with \DeltaV error')
    #title('Number of channels with \DeltaV error for optimal \DeltaV_{min}')

def main():
    g = gleb()
    print('\n***sum_delta_V_error')
    print(g['sum_delta_V_error'])
    print('\n***n_delta_V_error')
    print(g['n_delta_V_error'])
    print('\n***delta_V_actual_all')
    print(g['delta_V_actual_all'])
    print('\n***delta_V_all')
    print(g['delta_V_all'])
    print('\n***V_all')
    print(g['V_all'])
    print('\n***theta_out')
    print(g['theta_out'])
    print('\n***P')
    print(g['P'])

    pg = BSCPatternGenerator()
    angles, a_dict = pg.get_valid_steering_angles()
    print(angles)
    print('\n***theta_out')
    print(a_dict['theta_out'])
    print('\n***width_pattern')
    print(a_dict['width_pattern'])
    print('\n***patterns_per_tile')
    print(a_dict['patterns_per_tile'])
    print('\n***channels_per_pattern')
    print(a_dict['channels_per_pattern'])

if __name__ == '__main__':
    main()

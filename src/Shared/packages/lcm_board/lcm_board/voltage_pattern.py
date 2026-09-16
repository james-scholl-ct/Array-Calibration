import matplotlib.pyplot as plt
import numpy as np


pi = np.pi
sin = np.sin
asin = np.arcsin


def sind(x):
    return np.sin(x * pi / 180)


def asind(x):
    return np.arcsin(x) * 180 / pi


class OutsideFOVException(Exception):
    pass


class VoltagePatternGenerator:

    def __init__(self,
                 wavelength=913,
                 channels=204,
                 theta_i=-65,
                 pitch=300,
                 delv_min=1,
                 delv_max=6,
                 **kwargs):
        """This class generates simple delv patterns for use with lcm_board board

        Args:
            wavelength (float): laser wavelength, in nm
            channels (int): number of voltage channels in LCM
            theta_i (float): incident angle, in degrees
            pitch (float): pitch of rails, in nm, e.g. 300nm
            delv_min (float): minimum delv to use, in Vrms
            delv_max (float): maximum delv to use, in Vrms
            v_min (float): minimum v that lcm_board board can supply, in Vrms
            v_max (float): maximum v that lcm_board board can supply, in Vrms
            **kwargs: optional key word arguments

        Additional comments:
            delv is defined as abs(v[i] - v[i-1]) everywhere
            len(delv) = channels = len(v) = e.g. 204

        """
        self.wavelength = wavelength
        self.channel_count = channels
        self.gap_count = channels
        self.theta_i = theta_i
        self.pitch = pitch
        self.tile = self.pitch * self.channel_count
        self.delv_min = delv_min
        self.delv_max = delv_max
        self.minm = 1

    def square(self,
               m=-1,
               plot=False,
               flip=False):
        """
        Calculates a ramp delv profile according to the calculated pattern periodicity

        Args:
            m (int): diffraction order
            plot (boolean): boolean indicating whether to plot delv pattern
            flip (boolean): boolean indicating whether to flip direction of the ramp

        Returns:
            delv (numpy): array of delv values

        """
        channel_count = self.channel_count
        theta_i = self.theta_i
        delv_min = self.delv_min
        delv_max = self.delv_max
        theta_m = self.calculate_output_angle(m)  # calculate steer angle

        if abs(theta_m) < abs(theta_i):
            pattern_period = abs(channel_count / m)  # calculate spatial frequency
            gap_number = np.arange(0, channel_count, 1)  # gap list
            delv = np.zeros(len(gap_number))

            # calculate delta v
            for i, gap in enumerate(gap_number):
                delv[i] = (gap % pattern_period) / pattern_period * (delv_max - delv_min) + delv_min  # generate ramp
                # pin ramp to min or max to make it into a square wave
                if delv[i] < (self.delv_max + self.delv_min) / 2:
                    delv[i] = delv_min
                else:
                    delv[i] = delv_max

                # flip the pattern if desired
                if flip:
                    delv[i] = (1 - (gap % pattern_period) / pattern_period) * (delv_max - delv_min) + delv_min
                    if delv[i] < (self.delv_max + self.delv_min) / 2:
                        delv[i] = delv_min
                    else:
                        delv[i] = delv_max

            # plot delv
            xplot = np.arange(0, gap_number[-1] + 0.01, 0.01)  # higher resolution in x, for plotting
            delv_plot = np.zeros(len(xplot))
            for i, gap in enumerate(xplot):
                delv_plot[i] = (gap % pattern_period) / pattern_period * (delv_max - delv_min) + delv_min
                if delv_plot[i] < (self.delv_max + self.delv_min) / 2:
                    delv_plot[i] = delv_min
                else:
                    delv_plot[i] = delv_max

                if flip:
                    delv_plot[i] = (1 - (gap % pattern_period) / pattern_period) * (delv_max - delv_min) + delv_min
                    if delv_plot[i] < (self.delv_max + self.delv_min) / 2:
                        delv_plot[i] = delv_min
                    else:
                        delv_plot[i] = delv_max

            if plot:
                plt.cla()
                plt.plot(xplot, delv_plot, 'r', alpha=0.8)
                plt.plot(gap_number, delv, 'ob', markersize=3)
                plt.xlabel('gap number')
                plt.ylabel('delta V')
                plt.title('m: ' + str(m) + '   pattern period: ' + "{:.2f}".format(pattern_period)
                          + '   steer angle: ' + "{:.2f}".format(theta_m))
                plt.xlim([0, channel_count])
                plt.ylim([delv_min, delv_max])
                plt.show()
                plt.pause(0.01)
            return delv, delv_plot

        else:
            print('no steering angle')

    def logit(self,
              m=-1,
              k=7,
              offset_max=6,
              offset_min=-6,
              plot=False,
              flip=False):
        """
        Calculates a logit (inverse logistic) delv profile according to the calculated pattern periodicity
        Args:
            m (int): diffraction order
            k (float): exponential factor, will be scale the slope of the curve between the min and max: range 0.01 to 500
            offset_max: clips the logit function at this maximum value to prevent diverging values to upto +infinity.
            offset_min: clips the logit function at this minimum value to prevent diverging values to upto -infinity.
            Constraint: The offset_min must be less than offset_max at all times.
            plot (boolean): boolean indicating whether to plot delv pattern
            flip (boolean): boolean indicating whether to flip direction of the ramp
        Returns:
            delv (numpy): array of delv values
        """
        channel_count = self.channel_count
        theta_i = self.theta_i
        delv_min = self.delv_min
        delv_max = self.delv_max
        theta_m = self.calculate_output_angle(m)  # calculate steer angle
        kprime = k
        if abs(theta_m) < abs(theta_i):
            pattern_period = abs(channel_count / m)  # calculate pattern periodicity
            L = (delv_max - delv_min)
            k = kprime / pattern_period
            gap_number = np.arange(0, channel_count, 1)  # gap list
            delv = np.zeros(len(gap_number))
            # calculate delta v
            for i, gap in enumerate(gap_number):
                p = ((gap % pattern_period) / pattern_period)
                # delv[i] = L / (1 + np.exp(-k * (gap % pattern_period - pattern_period / 2))) + delv_min
                temp = np.log(p / (1 - p ** k))
                if temp < offset_min:
                    delv[i] = offset_min
                elif temp > offset_max:
                    delv[i] = offset_max
                else:
                    delv[i] = temp

            # flip blaze if selected
            if flip:
                delv = np.flipud(delv)
            delv_0_1 = (delv - delv.min()) / (delv.max() - delv.min())
            delv = delv_min + (delv_max - delv_min) * delv_0_1

            # plot delv
            xplot = np.arange(0, gap_number[-1] + 0.01, 0.01)  # higher resolution in x, for plotting
            delv_plot = np.zeros(len(xplot))
            for i, gap in enumerate(xplot):
                p = (gap % pattern_period) / pattern_period
                # delv[i] = L / (1 + np.exp(-k * (gap % pattern_period - pattern_period / 2))) + delv_min
                temp = np.log(p / (1 - p ** k))
                if temp < offset_min:
                    delv_plot[i] = offset_min
                elif temp > offset_max:
                    delv_plot[i] = offset_max
                else:
                    delv_plot[i] = temp

                # delv_plot[i] = (L) / (1 + np.log(1+k/(gap % pattern_period ))) + pattern_period / offset
            if flip:
                delv_plot = np.flipud(delv_plot)
            delv_plot_0_1 = (delv_plot - delv_plot.min()) / (delv_plot.max() - delv_plot.min())
            delv_plot = delv_min + (delv_max - delv_min) * delv_plot_0_1
            if plot:
                plt.cla()
                plt.plot(xplot, delv_plot, 'r', alpha=0.8)
                plt.plot(gap_number, delv, 'ob', markersize=3)
                plt.xlabel('gap number')
                plt.ylabel('delta V')
                plt.title('m: ' + str(m) + '   pattern period: ' + "{:.2f}".format(pattern_period)
                          + '   steer angle: ' + "{:.2f}".format(theta_m))
                plt.xlim([0, channel_count])
                plt.ylim([delv_min, delv_max])
                plt.show()
                plt.pause(0.01)
            return delv, delv_plot
        else:
            print('no steering angle')

    def taylor_series(self,
                      m=-1,
                      x0=0,
                      an2=0,
                      an3=0,
                      a1=1,
                      a2=0,
                      y_max=1,
                      flip=False,
                      plot=False):
        """
        This function calculates the delta V profile as a function of rail position for different grating orders using
        the Taylor series expansion.
        for the grating order m:

            pattern_period = channel_count/m

            x = (rail%pattern_period)/pattern_period : hence x represents the position of the rail between 0 and 1

            if x > x0:

                y = an2*(x-x0)**(1/2) +an3*(x-x0)**(1/3) + a1*(x-x0)+ a2*(x-x0)**2 for 0<y<y_max and x0<x<1
            else:
                y = 0

        The coefficients of the Taylor series:

            an2 : 0<an2<1 : square root slope
            an3 : 0<an3<1 : cube root slope
            a1  : 0<a1<10 : linear slope
            a2  : 0<a2<10 : quadratic slope

        Clipping limits of the Series:

            x0 : rail position clipping offset : 0<x0<1
            y_max : max voltage clipping offset:  10> y_max > 0

        Handles:

            flip: inverts the voltage pattern between positive and negative grating orders
            plot: returns an additional finely sample delv vector for plotting

        Returns:
            delv: list of the delv voltages to be applied on the LCM clipped between the min and max

        """

        channel_count = self.channel_count
        delv_min = self.delv_min
        delv_max = self.delv_max
        pattern_period = channel_count / np.abs(m)  # calculate pattern periodicity
        L = (delv_max - delv_min)
        gap_number = np.arange(0, channel_count, 1)  # gap list
        delv = np.zeros(len(gap_number))

        # calculate delta v
        for i, gap in enumerate(gap_number):
            p = ((gap % pattern_period) / pattern_period)
            p_val = p - x0
            if p_val >= 0:
                temp = an2 * (p_val) ** (1 / 2) + an3 * (p_val) ** (1 / 3) + a1 * p_val + a2 * p_val ** 3
            else:
                temp = 0

            if temp <= 0:
                delv[i] = 0
            elif temp > y_max:
                delv[i] = y_max
            else:
                delv[i] = temp

        # flip blaze if selected
        if flip:
            delv = np.flipud(delv)
        delv_0_1 = (delv - delv.min()) / (delv.max() - delv.min())
        delv = delv_min + (delv_max - delv_min) * delv_0_1

        if plot:
            xplot = np.arange(0, gap_number[-1] + 0.001, 0.001)  # higher resolution in x, for plotting
            delv_plot = np.zeros(len(xplot))
            yy = np.zeros_like(xplot)
            xx = (xplot % pattern_period) / pattern_period
            for idx, val in enumerate(xx):

                x_val = (val - x0)
                if x_val >= 0:
                    temp = an2 * (x_val) ** (1 / 2) + an3 * (x_val) ** (1 / 3) + a1 * x_val + a2 * x_val ** 3
                else:
                    temp = 0
                if temp > y_max:
                    yy[idx] = y_max
                elif temp < 0:
                    yy[idx] = 0
                else:
                    yy[idx] = temp
            yy = (yy - yy.min()) / (yy.max() - yy.min())  # clip_0_1
            delv_plot = delv_min + (delv_max - delv_min) * (yy)
            if flip:
                delv_plot = np.flipud(delv_plot)

            return delv, delv_plot
        else:
            return delv

    def ramp(self,
             m=-1,
             plot=False,
             flip=False):
        """
        Calculates a ramp delv profile according to the calculated pattern periodicity

        Args:
            m (int): diffraction order
            plot (boolean): boolean indicating whether to plot delv pattern
            flip (boolean): boolean indicating whether to flip direction of the ramp

        Returns:
            delv (numpy): array of delv values

        """
        channel_count = self.channel_count
        theta_i = self.theta_i
        delv_min = self.delv_min
        delv_max = self.delv_max
        theta_m = self.calculate_output_angle(m)  # calculate steer angle

        if abs(theta_m) < abs(theta_i):
            pattern_period = abs(channel_count / m)  # calculate spatial frequency
            gap_number = np.arange(0, channel_count, 1)  # gap list
            delv = np.zeros(len(gap_number))

            # calculate delta v
            for i, gap in enumerate(gap_number):
                delv[i] = (gap % pattern_period) / pattern_period * (delv_max - delv_min) + delv_min
                if flip:
                    delv[i] = (1 - (gap % pattern_period) / pattern_period) * (delv_max - delv_min) + delv_min

            # plot delv
            xplot = np.arange(0, gap_number[-1] + 0.01, 0.01)  # higher resolution in x, for plotting
            delv_plot = np.zeros(len(xplot))
            for i, gap in enumerate(xplot):
                delv_plot[i] = (gap % pattern_period) / pattern_period * (delv_max - delv_min) + delv_min
                if flip:
                    delv_plot[i] = (1 - (gap % pattern_period) / pattern_period) * (delv_max - delv_min) + delv_min
            if plot:
                plt.cla()
                plt.plot(xplot, delv_plot, 'r', alpha=0.8)
                plt.plot(gap_number, delv, 'ob', markersize=3)
                plt.xlabel('gap number')
                plt.ylabel('delta V')
                plt.title('m: ' + str(m) + '   pattern period: ' + "{:.2f}".format(pattern_period)
                          + '   steer angle: ' + "{:.2f}".format(theta_m))
                plt.xlim([0, channel_count])
                plt.ylim([delv_min, delv_max])
                plt.show()
                plt.pause(0.01)
            return delv, delv_plot

        else:
            print('no steering angle')

    def clipped_ramp(self,
                     m=1,
                     ramp_fraction=0.5,
                     plot=False,
                     flip=False):
        """
        Calculates a clipped ramp delv profile according to the calculated pattern periodicity

        Args:
            m (int): diffraction order
            ramp_fraction (float): fraction (out of 1) of waveform which is a ramp, the rest will be clipped
            plot (boolean): boolean indicating whether to plot delv pattern
            flip (boolean): boolean indicating whether to flip direction of the ramp

        Returns:
            delv (numpy): array of delv values

        """
        channel_count = self.channel_count
        theta_i = self.theta_i
        delv_min = self.delv_min
        delv_max = self.delv_max
        theta_m = self.calculate_output_angle(m)  # calculate steer angle

        if abs(theta_m) < abs(theta_i):
            pattern_period = abs(channel_count / m)  # calculate spatial frequency
            gap_number = np.arange(0, channel_count, 1)  # gap list
            delv = np.zeros(len(gap_number))

            # calculate delta v
            for i, gap in enumerate(gap_number):
                if (gap % pattern_period) / pattern_period < 0.5:
                    delv[i] = delv_min
                if (gap % pattern_period) / pattern_period >= 0.5:
                    delv[i] = delv_max
                if 0.5 - ramp_fraction / 2 < (gap % pattern_period) / pattern_period < 0.5 + ramp_fraction / 2:
                    delv[i] = ((gap % pattern_period) / pattern_period - (0.5 - ramp_fraction / 2)) / ramp_fraction * \
                              (delv_max - delv_min) + delv_min

            # flip blaze if selected
            if flip:
                delv = np.flipud(delv)

            # plot delv
            xplot = np.arange(0, gap_number[-1] + 0.01, 0.01)  # higher resolution in x, for plotting
            delv_plot = np.zeros(len(xplot))
            for i, gap in enumerate(xplot):
                if (gap % pattern_period) / pattern_period < 0.5:
                    delv_plot[i] = delv_min
                if (gap % pattern_period) / pattern_period >= 0.5:
                    delv_plot[i] = delv_max
                if 0.5 - ramp_fraction / 2 < (gap % pattern_period) / pattern_period < 0.5 + ramp_fraction / 2:
                    delv_plot[i] = ((gap % pattern_period) / pattern_period - (0.5 - ramp_fraction / 2)) / ramp_fraction * \
                              (delv_max - delv_min) + delv_min
            if flip:
                delv_plot = np.flipud(delv_plot)

            if plot:
                plt.cla()
                plt.plot(xplot, delv_plot, 'r', alpha=0.8)
                plt.plot(gap_number, delv, 'ob', markersize=3)
                plt.xlabel('gap number')
                plt.ylabel('delta V')
                plt.title('m: ' + str(m) + '   pattern period: ' + "{:.2f}".format(pattern_period)
                          + '   steer angle: ' + "{:.2f}".format(theta_m))
                plt.xlim([0, channel_count])
                plt.ylim([delv_min, delv_max])
                plt.show()
                plt.pause(0.01)
            return delv, delv_plot

        else:
            print('no steering angle')

    def logistic(self,
                 m=-1,
                 k=7,
                 plot=False,
                 flip=False):
        """
        Calculates a logistic delv profile according to the calculated pattern periodicity

        Args:
            m (int): diffraction order
            k (float): exponential factor, will be scaled by pattern periodicity
            plot (boolean): boolean indicating whether to plot delv pattern
            flip (boolean): boolean indicating whether to flip direction of the ramp

        Returns:
            delv (numpy): array of delv values

        """
        channel_count = self.channel_count
        theta_i = self.theta_i
        delv_min = self.delv_min
        delv_max = self.delv_max
        theta_m = self.calculate_output_angle(m)  # calculate steer angle
        kprime = k

        if abs(theta_m) < abs(theta_i):
            pattern_period = abs(channel_count / m)  # calculate pattern periodicity
            L = (delv_max - delv_min)
            k = kprime / pattern_period
            gap_number = np.arange(0, channel_count, 1)  # gap list
            delv = np.zeros(len(gap_number))

            # calculate delta v
            for i, gap in enumerate(gap_number):
                delv[i] = L / (1 + np.exp(-k * (gap % pattern_period - pattern_period / 2))) + delv_min

            # flip blaze if selected
            if flip:
                delv = np.flipud(delv)

            # plot delv
            xplot = np.arange(0, gap_number[-1] + 0.01, 0.01)  # higher resolution in x, for plotting
            delv_plot = np.zeros(len(xplot))
            for i, gap in enumerate(xplot):
                delv_plot[i] = L / (1 + np.exp(-k * (gap % pattern_period - pattern_period / 2))) + delv_min
            if flip:
                delv_plot = np.flipud(delv_plot)

            if plot:
                plt.cla()
                plt.plot(xplot, delv_plot, 'r', alpha=0.8)
                plt.plot(gap_number, delv, 'ob', markersize=3)
                plt.xlabel('gap number')
                plt.ylabel('delta V')
                plt.title('m: ' + str(m) + '   pattern period: ' + "{:.2f}".format(pattern_period)
                          + '   steer angle: ' + "{:.2f}".format(theta_m))

                plt.xlim([0, channel_count])
                plt.ylim([delv_min, delv_max])
                plt.show()
                plt.pause(0.01)
            return delv, delv_plot

        else:
            print('no steering angle')

    def uniform(self,
                uniform_delv=0,
                plot=False):
        """
        Calculates a uniform (flat) delv profile

        Args:
            uniform_delv (float): single value in Vrms for the uniform (flat) delv pattern
            plot (boolean): boolean if you want the function to plot the delv pattern

        Returns:
            delv (numpy): array of delv values

        """
        if uniform_delv > self.delv_max:
            uniform_delv = self.delv_max
            print('values clipped to prevent damage')
        channel_count = self.channel_count

        gap_number = np.arange(0, channel_count, 1)  # gap list
        delv = np.zeros(len(gap_number)) + uniform_delv

        # plot delv
        xplot = np.arange(0, gap_number[-1] + 0.01, 0.01)  # higher resolution in x, for plotting
        delv_plot = np.zeros(len(xplot)) + uniform_delv

        if plot:
            plt.cla()
            plt.plot(gap_number, delv, 'r', alpha=0.8)
            plt.plot(gap_number, delv, 'ob', markersize=3)
            plt.xlabel('gap number')
            plt.ylabel('delta V')
            plt.xlim([0, channel_count])
            plt.ylim([0, self.delv_max])
            plt.show()
            plt.pause(0.01)

        return delv, delv_plot

    @property
    def maxm(self):
        """
        Finds the maximum diffraction order possible for the experimental configuration

        Returns:
            m (int): maximum diffraction order for the configuration

        """
        wavelength = self.wavelength
        theta_i = np.abs(self.theta_i)
        tile = self.tile
        m = 1
        theta_m = asind(m * wavelength / tile - sind(theta_i))
        while abs(theta_m) < abs(theta_i):
            m = m + 1
            theta_m = asind(m * wavelength / (tile) - sind(theta_i))
        m = m - 1
        return abs(m)

    def calculate_output_angle(self, m):
        """
        Calculates the output steering angle in degrees

        Args:
            m (int): the LCM diffraction order

        Returns:
            theta_m (float): output steering angle, in degrees

        """
        wavelength = self.wavelength
        theta_i = self.theta_i
        tile = self.tile
        if abs(self.minm) <= abs(m) <= abs(self.maxm):
            if theta_i <= 0:
                m = -1 * abs(m)
                theta_m = asind(m * wavelength / tile - sind(theta_i))
            if theta_i > 0:
                m = abs(m)
                theta_m = asind(m * wavelength / tile - sind(theta_i))
            theta_m = float(theta_m)
            return theta_m
        else:
            raise OutsideFOVException('Diffraction order outside FOV')


class VoltagePatternTranslator:

    def __init__(self,
                 v_min=0,
                 v_max=9):
        """
        This class provides methods for translating delv patterns to v patterns

        Args:
            v_min (float): minimum voltage in Vrms the Himax driver can supply
            v_max (float): maximum voltage in Vrms the Himax driver can supply

        Additional comments:
            delv is defined as abs(v[i] - v[i-1]) everywhere
            len(delv) = channels = len(v) = e.g. 204

        """

        self.v_min = v_min
        self.v_max = v_max

    def inrange(self, x):
        """
        Checks to see if voltage value x is within the min/max voltage range

        Args:
            x (float): voltage value in question

        Returns:
            result (boolean): boolean indicating whether x is in the available voltage range

        """

        if self.v_min <= x <= self.v_max:
            result = True
        else:
            result = False
        return result

    def rotate(self, delv):
        """
        Determines where to start the translation by finding two sequential non-zero delv values

        Args:
            delv (numpy): array of delv values

        Returns:
            delv_rotated (numpy): rotated array of delv values

        """

        channels = np.arange(0, len(delv), 1)
        idx_v_start = -1
        for i, ch in enumerate(channels):
            if delv[ch - 1] > 0 and delv[ch - 2] > 0:
                idx_v_start = ch
                break
        if idx_v_start == -1:
            raise RuntimeError('No two consecutive rails are open!  Unable to generate safe voltage pattern.')
        delv_rotated = np.roll(delv, -idx_v_start)
        return delv_rotated, idx_v_start

    def delvtov_anchor(self, delv):
        """
        Converts delv to v, aiming to anchor the voltage at the 0V rail as much as possible

        Args:
            delv (numpy): array of delv values

        Returns:
            v (numpy): array of v values

        """

        channels = np.arange(0, len(delv), 1)
        v_rotated = np.zeros(len(channels))
        clipping_cases = 0
        delv_rotated, idx_v_start = self.rotate(delv)
        for i, ch in enumerate(channels):
            if i == channels[0]:
                v_rotated[i] = 0
            elif i == channels[-1]:
                v_rotated[i] = self.v_max / 2
            else:
                choice1 = v_rotated[i - 1] + delv_rotated[i]
                choice2 = v_rotated[i - 1] - delv_rotated[i]
                if self.inrange(choice1) == False and self.inrange(choice2) == True:
                    v_rotated[i] = choice2
                if self.inrange(choice1) == True and self.inrange(choice2) == False:
                    v_rotated[i] = choice1
                if self.inrange(choice1) == True and self.inrange(choice2) == True:
                    diff1 = abs(choice1 - self.v_min)
                    diff2 = abs(choice2 - self.v_min)
                    if diff1 < diff2:
                        v_rotated[i] = choice1
                    else:
                        v_rotated[i] = choice2
                if self.inrange(choice1) == False and self.inrange(choice2) == False:
                    clipping_cases = clipping_cases + 1
                    diff1 = abs(v_rotated[i - 1] - self.v_max)
                    diff2 = abs(v_rotated[i - 1] - self.v_min)
                    if diff1 > diff2:
                        v_rotated[i] = self.v_max
                    else:
                        v_rotated[i] = self.v_min
        # v_rotated = self.v_max - v_rotated
        v = np.roll(v_rotated, idx_v_start)
        return v, clipping_cases

    def delvtov_edges(self, delv):
        """
        Converts delv to v, aiming to keep the voltage to the edges of the range as much as possible

        Args:
            delv (numpy): array of delv values

        Returns:
            v (numpy): array of v values

        """
        channels = np.arange(0, len(delv), 1)
        v_rotated = np.zeros(len(channels))
        clipping_cases = 0
        delv_rotated, idx_v_start = self.rotate(delv)
        for i, ch in enumerate(channels):
            if i == channels[0]:
                v_rotated[i] = 0
            elif i == channels[-1]:
                v_rotated[i] = self.v_max / 2
            else:
                choice1 = v_rotated[i - 1] + delv_rotated[i]
                choice2 = v_rotated[i - 1] - delv_rotated[i]
                if self.inrange(choice1) == False and self.inrange(choice2) == True:
                    v_rotated[i] = choice2
                if self.inrange(choice1) == True and self.inrange(choice2) == False:
                    v_rotated[i] = choice1
                if self.inrange(choice1) == True and self.inrange(choice2) == True:
                    diff1min = abs(choice1 - self.v_min)
                    diff2min = abs(choice2 - self.v_min)
                    diff1max = abs(choice1 - self.v_max)
                    diff2max = abs(choice2 - self.v_max)
                    if diff1min > diff1max:
                        diff1 = diff1max
                    else:
                        diff1 = diff1min
                    if diff2min > diff2max:
                        diff2 = diff2max
                    else:
                        diff2 = diff2min

                    if diff1 < diff2:
                        v_rotated[i] = choice1
                    else:
                        v_rotated[i] = choice2
                if self.inrange(choice1) == False and self.inrange(choice2) == False:
                    clipping_cases = clipping_cases + 1
                    diff1 = abs(v_rotated[i - 1] - self.v_max)
                    diff2 = abs(v_rotated[i - 1] - self.v_min)
                    if diff1 > diff2:
                        v_rotated[i] = self.v_max
                    else:
                        v_rotated[i] = self.v_min
        v = np.roll(v_rotated, idx_v_start)
        return v, clipping_cases

    def delvtov_center(self, delv):
        """
        Converts delv to v, aiming to keep the voltage in the center of two rails

        Args:
            delv (numpy): array of delv values

        Returns:
            v (numpy): array of v values

        """

        channels = np.arange(0, len(delv), 1)
        v_rotated = np.zeros(len(channels))
        clipping_cases = 0
        delv_rotated, idx_v_start = self.rotate(delv)
        avrange = (self.v_max + self.v_min) / 2
        for i, ch in enumerate(channels):
            if i == channels[0]:
                v_rotated[i] = self.v_max / 2
            elif i == channels[-1]:
                v_rotated[i] = self.v_max / 2
            else:
                choice1 = v_rotated[i - 1] + delv_rotated[i]
                choice2 = v_rotated[i - 1] - delv_rotated[i]
                if self.inrange(choice1) == False and self.inrange(choice2) == True:
                    v_rotated[i] = choice2
                if self.inrange(choice1) == True and self.inrange(choice2) == False:
                    v_rotated[i] = choice1
                if self.inrange(choice1) == True and self.inrange(choice2) == True:
                    diff1 = abs(choice1 - avrange)
                    diff2 = abs(choice2 - avrange)
                    if diff1 < diff2:
                        v_rotated[i] = choice1
                    else:
                        v_rotated[i] = choice2
                if self.inrange(choice1) == False and self.inrange(choice2) == False:
                    clipping_cases = clipping_cases + 1
                    diff1 = abs(v_rotated[i - 1] - self.v_max)
                    diff2 = abs(v_rotated[i - 1] - self.v_min)
                    if diff1 > diff2:
                        v_rotated[i] = self.v_max
                    else:
                        v_rotated[i] = self.v_min
        v = np.roll(v_rotated, idx_v_start)
        return v, clipping_cases

    def vtodelv(self, v):
        """
        Converts v to delv. delv is defined as abs(v[i] - v[i-1])

        Args:
            v: numpy array of v values

        Returns:
            delv: numpy array of delv values

        """
        delv = np.zeros(len(v))
        for i, val in enumerate(delv):
            delv[i] = abs(v[i] - v[i - 1])
        return delv

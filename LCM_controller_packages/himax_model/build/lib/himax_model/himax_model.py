# import matplotlib.pyplot as plt
import numpy as np
import os
import yaml


_YAML_FILE = os.path.join(os.path.dirname(__file__),
                          'yaml',
                          'hx8175-k10_model.yml')


class HimaxModel:
    def __init__(self, vgma_bias, vdd=18.):
        '''vgma_bias is a dictionary with keys "vgma1" to "vgma20". Only keys
        for vgma1, 2, 9, 10, 11, 12, 19, and 20 are required. All values must
        be ints or floats. Also, values be monotonically decreasing from vgma1
        to vgma20.
        '''
        self.vdd = vdd
        self._yml_data = self._parse_yaml()
        # get default voltages using mandatory entries from vgma_bias
        self._vgma = self._format_and_eval_yaml(self._yml_data['vgma_default'],
                                                vgma_bias)
        # update voltages with user overrides and compute transfer curves
        self.update_vgma(vgma_bias)


    def _parse_yaml(self, file_path=_YAML_FILE):
        '''Parses a yaml file that describes the driver's transfer function.
        Currently, three fields are supported: vgma_default, dac_hi, and dac_lo.
        '''
        with open(file_path, 'r') as f:
            y = yaml.safe_load(f)
        return y


    def _format_and_eval_yaml(self, items, fields):
        '''Interprets 'items' as a list of strings containing mathematical
        expressions to be evaluated with eval(). Each string may contain
        fields to be populated using the format() command prior to evaluation.
        The dict 'fields' provides the values for the fields referenced in the
        strings. This function will raise a 'KeyError' exception if a format
        field in one of the strings is missing from the dict 'fields'.
        '''
        return [eval(s.format(**fields)) for s in items]


    def _validate_vgma(self):
        '''Check that 1) there are 20 references, 2) they are monotonically
        decreasing from vgma1 to vgma20, and 3) the values are either floats
        or ints.
        '''
        if len(self._vgma) != 20:
            msg = "self._vgma has length {}.".format(len(self._vgma))
            raise RuntimeError(msg)

        for i, vgma in enumerate(self._vgma):
            if not isinstance(vgma, float) and not isinstance(vgma, int):
                msg = "Invalid gamma voltage: vgma{} = {}".format(i+1, vgma)
                raise RuntimeError(msg)

        for i, (hi, lo) in enumerate(zip(self._vgma[:-1], self._vgma[1:])):
            if hi < lo:
                first = "vgma{} = {}".format(i+1, hi)
                second = "vgma{} = {}".format(i+2, lo)
                raise RuntimeError("{} must be >= {}".format(first, second))


    def get_vgma(self):
        '''Return the gamma reference voltages as a dictionary with keys
        "vgma1" to "vgma20".
        '''
        return {'vgma{}'.format(i+1): vgma for i, vgma in enumerate(self._vgma)}


    def update_vgma(self, dikt):
        '''Update the gamma references using the provided dict. The dict may
        be partial, in which case, the value for any omitted keys will remain
        unchanged. After updating the gamma voltages, the transfer curves are
        also updated.
        '''
        for i in range(20):
            key = 'vgma{}'.format(i+1)
            vgma = dikt.get(key, None)
            if vgma is not None:
                self._vgma[i] = vgma
        self._validate_vgma()
        self._dac_hi = self._format_and_eval_yaml(self._yml_data['dac_hi'],
                                                  self.get_vgma())
        self._dac_lo = self._format_and_eval_yaml(self._yml_data['dac_lo'],
                                                  self.get_vgma())

    @property
    def N_CODES(self):
        return len(self._dac_lo)

    @property
    def MAX_CODE(self):
        return len(self._dac_lo) - 1

    @property
    def MIN_CODE(self):
        return 0

    def get_voltages(self, codes, high_vec=False):
        '''Returns a list of voltages for the provided list of digital codes.
        There are two possible transfer functions for the code domain. Each
        element in the list 'high_vec' indicates which transfer function to
        use for the corresponding code in 'codes'. Set the high_vec element to
        True to access the function with range [VDD/2, VDD] and set it to False
        to access the function with range [0, VDD/2], which is the default. If
        high_vec is a bool then its value will be automatically broadcasted and
        used for all codes.
        '''
        if isinstance(high_vec, bool):
            high_vec = [high_vec for _ in codes]
        return [self._dac_hi[code] if high else self._dac_lo[code]
                for code, high in zip(codes, high_vec)]


    def get_codes(self, voltages):
        '''Returns a list of codes for the provided list of voltages. Each code
        is generated through linear interpolation and then rounding to the
        nearest integer. There are two possible transfer functions for the
        code domain: one with range [0, VDD/2] and one with range [VDD/2, VDD].
        This function converts any value in the range [VDD/2, VDD] to its corresponding
        voltage in [0, VDD/2], which results in the same code from the transfer function.
        An exception is raised if the supplied voltages are outside of [0, VDD].
        '''
        max_v = max(voltages)
        min_v = min(voltages)
        voltages = np.asarray(voltages)
        voltages[voltages > self.vdd/2] = abs(self.vdd-voltages[voltages > self.vdd/2])
        if max_v <= self.vdd and min_v >= 0:
            codes = np.interp(voltages,
                              self._dac_lo[::-1],
                              list(reversed(range(len(self._dac_lo)))))
        else:
            msg = "The provided voltages are out of range "
            msg += "(min given = {}, max given = {})."
            msg = msg.format(min_v, max_v)
            raise RuntimeError(msg)
        return [int(i) for i in np.round(codes)]


    # def plot(self, show=True):
    #     '''Plot the transfer functions.
    #     '''
    #     gamma_codes = (0xFF, 0xFE, 0xDF, 0xBF, 0x7F, 0x3F, 0x1F, 0x02, 0x01, 0x00,
    #                    0x00, 0x01, 0x02, 0x1F, 0x3F, 0x7F, 0xBF, 0xDF, 0xFE, 0xFF)
    #     plt.plot(gamma_codes, self._vgma, 'ro', label='Gamma Voltages')
    #
    #     codes = list(range(len(self._dac_lo)))
    #     plt.plot(codes, self._dac_lo)
    #     plt.plot(codes, self._dac_hi)
    #
    #     voltages_lo = [v for v in range(int(self.vdd+1)) if v <= self.vdd/2]
    #     voltages_hi = [v for v in range(int(self.vdd+1)) if v >= self.vdd/2]
    #     codes_lo = self.get_codes(voltages_lo)
    #     codes_hi = self.get_codes(voltages_hi)
    #     plt.plot(codes_lo, voltages_lo, 'g.', label='Integral Voltages (low)')
    #     plt.plot(codes_hi, voltages_hi, 'b.', label='Integral Voltages (high)')
    #
    #     plt.xlabel('code')
    #     plt.ylabel('Voltage (V)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.draw()
    #     if show:
    #         plt.show()


def main():
    vgma = {'vgma1': 18,
            'vgma2': 18,
            'vgma9': 9,
            'vgma10': 9,
            'vgma11': 9,
            'vgma12': 9,
            'vgma19': 0,
            'vgma20': 0}
    h = HimaxModel(vgma)
    # #h.plot(show=False)
    # h.plot()
    #
    # v = list(range(1,9))
    # c = h.get_codes(v)
    # for i, j in zip(v,c):
    #     print(i, j, hex(j))
    #
    # v = list(range(10,18))
    # c = h.get_codes(v)
    # for i, j in zip(v,c):
    #     print(i, j, hex(j))
    #
    # vgma['vgma3'] = 16
    # h.update_vgma(vgma)
    # h.plot()


if __name__ == '__main__':
    main()

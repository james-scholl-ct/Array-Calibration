import numpy as np


class VoltageTranslatorFFT:
    """Converts between Delta V and V patterns, optimizing for minimum error, voltage, and/or Fourier component."""

    def __init__(self,
                 min_voltage,
                 max_voltage,
                 max_del_v,
                 error_factor=1,
                 voltage_factor=0,
                 fft_factor=1,
                 sidelobe_factor=0,
                 num_rails=204):
        """Initializes the translator object with specific voltage limits and algorithm parameters

        Args:
            min_voltage: minimum allowed voltage
            max_voltage: maximum allowed voltage
            max_del_v: maximum delta V (absolute value)
            error_factor: how much to weight voltage errors
            voltage_factor: how much to weight mean voltage
            fft_factor: how much to weight maximizing the fourier component at diff_order
            sidelobe_factor: how much to weight minimizing the next-highest fourier component
            num_rails: number of LCM rails"""
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.max_del_v = max_del_v
        self.num_rails = num_rails
        self.max_depth = 4
        self.num_bests = 4
        self.error_factor = error_factor
        self.voltage_factor = voltage_factor
        self.fft_factor = fft_factor
        self.sidelobe_factor = sidelobe_factor

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
            delv[i] = abs(v[i] - v[i-1])
        return delv

    def delvtov(self, delv,
                diff_order=0,
                max_depth=4,
                num_bests=4,
                error_factor=1,
                voltage_factor=0.01,
                target_v_avg=0,
                fft_factor=1,
                sidelobe_factor=1,
                max_voltage=None):
        """Converts a delta v pattern into a v pattern.

        Considers repeated trees of options to find an optimal result based on cost factors.
        Args:
            delv: desired delta V pattern
            diff_order: desired diff_order for fourier optimization
            max_depth: depth of each tree
            num_bests: number of best results to keep from each tree
            error_factor: how much to weight voltage errors
            voltage_factor: how much to weight mean voltage
            target_v_avg: what voltage to aim for in weighting algorithm
            fft_factor: how much to weight maximizing the fourier component at diff_order
            sidelobe_factor: how much to weight minimizing the next-highest fourier component
            max_voltage: voltage limit of the driver

        Returns:
            v_pattern: a list of rail voltages that best approximates the target delta_v
            """
        if not max_voltage is None:
            self.max_voltage = max_voltage
        self.diff_order = diff_order
        self.max_depth = max_depth
        self.num_bests = num_bests
        self.error_factor = error_factor
        self.voltage_factor = voltage_factor
        self.target_v_avg = target_v_avg
        self.fft_factor = fft_factor
        self.sidelobe_factor = sidelobe_factor
        self.target_del_v_pattern = delv
        self.last_voltage = 4.5
        initial_v_guess = [self.last_voltage] * self.num_rails
        v_pattern = self.optimize_pattern(initial_v_guess)
        self.check_result(v_pattern, self.target_del_v_pattern)
        return v_pattern

    def find_voltages(self, current_voltage, target_del_v):
        """Finds all allowed ways to generate the next delta V.
        Returns a list of allowed voltages, clipped to allowed limits."""
        # plus option:
        plus_voltage = current_voltage + target_del_v
        plus_voltage = min(plus_voltage, self.max_voltage)
        # minus options:
        minus_voltage = current_voltage - target_del_v
        minus_voltage = max(minus_voltage, self.min_voltage)
        return [plus_voltage, minus_voltage]

    def find_constrained_voltages(self, v1, del_v2, del_v3, v3):
        """Finds all allowed ways to pick a given voltage that is constrained on both sides"""
        # Pattern is: v1 - del_v2 - v2 - del_v3 - v3
        # Solve for allowed v2. Use average of two del_v results, to put equal error on both.
        plus_minus = ((v1 + del_v2) + (v3 - del_v3))/2
        plus_plus = ((v1 + del_v2) + (v3 + del_v3))/2
        minus_plus = ((v1 - del_v2) + (v3 + del_v3))/2
        minus_minus = ((v1 - del_v2) + (v3 - del_v3))/2
        options = [plus_minus, plus_plus, minus_plus, minus_minus]
        allowed_options = []
        # example error case: 9V - 5.5V - VN - 0.1V - 0V
        # averaging will pull VN out of delta_v range
        options += [(v1 + v3) / 2]
        for option in options:
            # coerce values into voltage limits
            coerced_option = min(max(option, self.min_voltage), self.max_voltage)
            # compute actual del_vs, then don't allow voltages that would exceed either max_del_v
            actual_del_v2 = abs(coerced_option - v1)
            actual_del_v3 = abs(coerced_option - v3)
            if actual_del_v3 < self.max_del_v and actual_del_v2 < self.max_del_v:
                allowed_options += [coerced_option]
        if len(allowed_options) == 0:
            # I think you can't reach this error?
            raise RuntimeError(f'Could not find an allowed solution for last voltage.')
        return allowed_options

    def weight_result(self, v_pattern, del_v_start_index, depth):
        """Determines the weight for a given result"""
        weight = 0
        for i in range(del_v_start_index, del_v_start_index + depth):
            # Only calculate weight for the parts you changed.
            v1 = v_pattern[i-1]
            v2 = v_pattern[i]
            actual_del_v = abs(v2 - v1)
            target_del_v = self.target_del_v_pattern[i]
            # --weight by voltage error--
            del_v_error = abs(actual_del_v - target_del_v)
            weight += del_v_error * self.error_factor
            # to instead weight by phase error, change to this:
            # phase_error = abs(self.get_phase(actual_del_v) - self.get_phase(target_del_v))
            # weight += phase_error * self.error_factor
            # --weight by absolute voltage--
            weight += abs(v2 - self.target_v_avg) * self.voltage_factor
        # --weight by fft--
        if not self.fft_factor == 0:
            # What matters most? I think: 1, 2, not 3.
            # 1) Maximize peak amplitude
            fft_list = np.fft.fft(v_pattern)
            peak_amplitude = np.absolute(fft_list[self.diff_order] / len(v_pattern))
            weight -= peak_amplitude * self.fft_factor
            # 2) Minimize next-highest single sidelobe amplitude
            if not self.sidelobe_factor == 0:
                side_lobe_list = np.delete(np.absolute(fft_list / len(v_pattern)), [0, self.diff_order, self.num_rails - self.diff_order])
                weight += self.sidelobe_factor*side_lobe_list.max()
            # 3) Minimize average of all side lobes
            # fft_amplitude = fft_list[self.diff_order] / fft_list.sum()
        return weight

    @staticmethod
    def get_phase(del_v):
        """A model function that maps applied delta V to resulting phase

        Args:
            del_v: Applied Delta V"""
        # Current model: no phase change between 0 and 1V, then linear ramp afterwards.
        # We can update this however we want.
        slope = 1
        offset = -1
        if abs(del_v) < 1:
            return 0
        else:
            return slope*abs(del_v) + offset

    @staticmethod
    def read_upwards_from_node(node):
        """Given a node, read up the tree and return the list of voltages that leads to this node.
        Doesn't return a voltage for the root node to avoid duplication."""
        voltage_pattern = []
        while node.parent is not None:
            voltage_pattern.append(node.voltage)
            node = node.parent
        voltage_pattern.reverse()
        return voltage_pattern

    def generate_node_list(self, root, depth, del_v_index):
        """Recursively generate a full list of all options, up to depth"""

        if depth == 0:
            return [root]
        else:
            if self.target_del_v_pattern[del_v_index] == 0:
                # Never drive a short - fix delta V of 0
                allowed_voltages = [root.voltage]
            elif del_v_index == len(self.target_del_v_pattern) - 2:
                # At end of pattern, can't select last voltage, so different voltages allowed.
                v202 = root.voltage
                delv_203 = self.target_del_v_pattern[del_v_index]
                delv_204 = self.target_del_v_pattern[del_v_index + 1]
                v_204 = self.last_voltage
                allowed_voltages = self.find_constrained_voltages(v202, delv_203, delv_204, v_204)
            else:
                allowed_voltages = self.find_voltages(root.voltage, self.target_del_v_pattern[del_v_index])
            children = []
            for voltage in allowed_voltages:
                new_root = Node(voltage, root)
                children += self.generate_node_list(new_root, depth - 1, del_v_index + 1)
            return children

    def assess_one_chunk(self, prev_result, del_v_start_index, depth):
        """Computes the best allowed pattern for the next chunk of pattern, up to depth.

        Args:
            prev_result: The previous chunk's results.
            del_v_start_index: First index in the del_v data to be calculated
            depth: How far to calculate for this chunk."""
        # Assume that the function above this one handles matching the list end condition.
        # --Set starting conditions--
        start_v = prev_result.v_list[del_v_start_index - 1]
        start_node = Node(start_v, None)
        results = []
        # --Get all possible results and their weights--
        possible_nodes = self.generate_node_list(start_node, depth, del_v_start_index)
        for node in possible_nodes:
            new_v_list_subset = self.read_upwards_from_node(node)
            new_v_list = prev_result.v_list.copy()
            new_v_list[del_v_start_index : del_v_start_index + depth] = new_v_list_subset
            weight = prev_result.weight + self.weight_result(new_v_list, del_v_start_index, depth)
            results += [Result(weight, new_v_list)]
        # --Pre-sort results and return the best num_bests of them--
        best_results = sorted(results, key=lambda x: x.weight)
        return best_results[0:self.num_bests]

    def optimize_pattern(self, initial_v_guess):
        """Runs the full algorithm to compute delta V.

        Args:
            initial_v_guess: A guess at the best pattern.  Mostly ignored, only affects fourier weighting.
            """
        # If rail 204 is shorted to rail 1, can't apply any voltage.
        # Iterate backwards until you find the first open channel, then set that as the end condition.
        end_index = len(self.target_del_v_pattern)-1
        while self.target_del_v_pattern[end_index] == 0:
            end_index -= 1
        # set up initial values
        best_results = [Result(0, initial_v_guess)]
        current_index = 0
        # after first run, tree_depth changes to self.max_depth
        tree_depth = end_index % self.max_depth
        while current_index < end_index:
            new_bests = []
            for result in best_results:
               new_bests += self.assess_one_chunk(result, current_index, tree_depth)
            best_results = sorted(new_bests, key=lambda x: x.weight)
            best_results = best_results[0:self.num_bests]
            current_index += tree_depth
            tree_depth = self.max_depth
        return best_results[0].v_list

    def multipass(self, initial_v_guess, num_passes):
        """Repeatedly run optimize_pattern, to see if multiple passes improve the result"""
        # Haven't tested this function and I'm not sure if it matters
        v_result = initial_v_guess
        for i in range(0,num_passes):
            v_result = self.optimize_pattern(v_result)
        return v_result

    def assess_results(self, target_del_v_pattern, v_pattern):
        """Calculates some details about the algorithm results

        Not activated during normal runs, only during assessment"""
        actual_del_v = self.vtodelv(v_pattern)
        error = -1*np.ones(len(target_del_v_pattern))
        for i in range(len(v_pattern)):
            error[i] = target_del_v_pattern[i] - actual_del_v[i]
        fft_results = np.absolute(np.fft.fft(v_pattern)/len(v_pattern))
        return actual_del_v, error, fft_results

    def check_result(self, v_pattern, target_del_v_pattern):
        """Double checks that a given output voltage pattern has the right min voltage, max voltage, and max delv

         @Erik thinks this will never raise an error, so come ask me if this happens."""
        actual_del_v = self.vtodelv(v_pattern)
        for i in range(0, len(v_pattern)):
            if not self.min_voltage <= v_pattern[i] <= self.max_voltage:
                raise RuntimeError(f'Voltage on rail {i} out of range, at {v_pattern[i]}! Ask Erik.')
            if not abs(actual_del_v[i]) <= self.max_del_v:
                raise RuntimeError(f'Delta V {i} out of range, at {actual_del_v[i]}! Ask Erik.')
            if target_del_v_pattern[i] == 0 and not actual_del_v[i] == 0:
                start_rail = i + 1
                end_rail = i % 204 + 1
                raise RuntimeError(f'Optimizer tried to drive {start_rail} to {end_rail}, but this is shorted!')


class Node:
    """Represents a single possible choice of rail voltage at some depth"""
    def __init__(self, voltage, parent):
        self.voltage = voltage
        self.parent = parent

    def __str__(self):
        # Define how this object is converted to a string, useful for debugging
        return f'Node: V={self.voltage}'


class Result:
    """Represents a list of voltages that solve the problem, weighted by quality.  Lower weight is better."""
    def __init__(self, weight, v_list):
        self.weight = weight
        self.v_list = v_list

    def __str__(self):
        return f'Result: w={self.weight}, v_list={self.v_list}'


if __name__ == '__main__':
    """Test code to see if it's working"""
    order = 3
    sample_v = [0, 1.1, 4.5, 0, 5.9, 4.1, 9, 4.3, 9, 4.3, 9, 4.4, 9, 4.4, 9, 3.5, 7.5, 4.5,
                4, 2, 1, 5, 3, 1, 6, 4, 2, 4, 3, 1]
    # sample_v = [0, 1.1, 4.5, 0, 5.9, 4.1, 9, 9, 4.3, 9, 4.4, 9, 4.4, 9, 3.5, 7.5, 4.5]
    # sample_v = [0, 1.1, 4.5, 3]
    vt = VoltageTranslatorFFT(min_voltage=0, max_voltage=9, max_del_v=6.5, num_rails=len(sample_v))
    sample_del_v = vt.vtodelv(sample_v)
    v_pattern_out = vt.delvtov(sample_del_v, diff_order=order, fft_factor=.5)
    actual_del_v_results, error_results, fft_sample_results = vt.assess_results(sample_del_v, v_pattern_out)
    np.set_printoptions(precision=3, suppress=True)
    print('results:')
    print(f'target del v: {np.array(sample_del_v)}')
    print(f'actual del v: {actual_del_v_results}')
    print(f'error avg:        {error_results.mean()}')
    print(f'old v: {np.array(sample_v)}')
    print(f'new v: {np.array(v_pattern_out)}')
    print(f'old v avg: {np.array(sample_v).mean()}')
    print(f'new v avg: {np.array(v_pattern_out).mean()}')
    fft_before = np.absolute(np.fft.fft(sample_v)[order]/len(sample_v))
    fft_after = np.absolute(np.fft.fft(v_pattern_out)[order]/len(sample_v))
    print(f'fft amplitude before: {fft_before:0.2f}')
    print(f'fft amplitude after: {fft_after:0.2f}')


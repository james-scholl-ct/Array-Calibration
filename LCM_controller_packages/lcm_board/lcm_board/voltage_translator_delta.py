import numpy as np
from python_tools import himax_model
from bisect import bisect_left
import math


class VoltageTranslator:
    """Converts between Delta V and V patterns, optimizing for minimum error, voltage, and/or Fourier component."""

    def __init__(self,
                 max_del_v=6,
                 error_factor=1,
                 voltage_factor=0,
                 max_depth=2,
                 num_bests=2,
                 aggressive=False,
                 soft_seam_factor=1.0):
        """Initializes the translator object with specific voltage limits and algorithm parameters

        Args:
            max_del_v: maximum delta V (absolute value)
            error_factor: how much to weight voltage errors
            voltage_factor: how much to weight mean voltage
            max_depth: depth of search tree.  Typ. 2 for speed, bump to 3 or 4 to get more accuracy but slower run
            num_bests: number of parallel search trees to keep.  Also typ. 2.
            soft_seam_factor: a limit on max delta V near the odd/even domain boundaries. Expressed as percent of nominal delta V.
            """
        self.min_voltage = 0
        self.middle_voltage = 9
        self.max_voltage = 18
        self.max_del_v = max_del_v
        self.num_rails = 1021
        self.max_depth = max_depth
        self.num_bests = num_bests
        self.aggressive = aggressive
        if aggressive:
            print('WARNING: Voltage Translator is running in aggressive mode!')
            print('This will ignore max_del_v and shorting errors, and could damage the LCM')
        self.error_factor = error_factor
        self.voltage_factor = voltage_factor
        vgma = {'vgma1': 18,
                'vgma2': 18,
                'vgma9': 9,
                'vgma10': 9,
                'vgma11': 9,
                'vgma12': 9,
                'vgma19': 0,
                'vgma20': 0}
        self.h = himax_model.HimaxModel(vgma)
        self.low_voltages = self.h._dac_lo[1:-1]
        self.low_voltages.reverse()
        self.high_voltages = [18 - v for v in self.h._dac_lo[1:-1]]
        # need to know the maximum digitization error
        v_steps = [self.low_voltages[i+1] - self.low_voltages[i] for i in range(len(self.low_voltages)-1)]
        biggest_step = max(v_steps)
        self.max_digitization_error = biggest_step/2 + 0.01
        # 0.01 is to avoid weird floating point math errors
        # for default himax values, max_digitization_error is 0.4743
        # at the end of the region, you need at least the following number of steps to get from 0 to 9V
        # not totally accurate to non-linear digitization, but not worth the complexity to add that.
        # this assumes min_voltage and max_voltage always symmetric about middle voltage.
        self.end_step_size = (self.max_del_v*soft_seam_factor - self.max_digitization_error)
        self.end_steps_count = math.ceil((self.middle_voltage - self.min_voltage) / self.end_step_size)
        if self.max_del_v < 3:
            print('Warning: small values of max_del_v are more sensitive to shorting')
            print(f'One short just before 511 or 1021 could knock out {self.end_steps_count} channels')
        if self.end_step_size < 0:
            raise RuntimeError(f'Delta V too small, at {self.max_del_v}.  Cannot solve due to digitization.')

    @staticmethod
    def vtodelv(v):
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

    def delvtov(self, delv, short_filter):
        """Converts a delta v pattern into a v pattern.

        Considers repeated trees of options to find an optimal result based on cost factors.
        Args:
            delv: desired delta V pattern
            short_filter: list of shorted channels.  If short_filter[i] == 0, delta_v[i] is shorted.

        Returns:
            v_pattern: a list of rail voltages that semi-optimally approximates the target delta_v
            """
        self.target_del_v_pattern = delv
        self.short_filter = short_filter
        init_results = [Result(0, [self.middle_voltage] * self.num_rails)]
        odd_results = self.solve_one_half(1, 511, init_results)
        even_results = self.solve_one_half(512, 1021, odd_results)
        v_pattern = even_results[0].v_list
        self.check_result(v_pattern, self.short_filter)
        return v_pattern

    def find_voltages(self, current_v, target_del_v, del_v_index):
        """Find options for the next voltage that satisfy delta V. """
        plus_voltage = self.digitize_to_del_v_limit(current_v, 0 + target_del_v, del_v_index)
        minus_voltage = self.digitize_to_del_v_limit(current_v, 0 - target_del_v, del_v_index)
        return [plus_voltage, minus_voltage]

    def digitize_to_del_v_limit(self, current_v, target_del_v, del_v_index):
        """Finds the nearest Himax output voltage that does not violate delta V limits"""
        # as you get closer to boundary condition, start to limit voltage range
        # to make sure you can always find a solution for the last step.
        ideal_voltage = current_v + target_del_v
        steps_to_end = self.stop_index - del_v_index
        step_size = self.end_step_size
        # something here:
        if self.current_min_v == 0:  # low range
            allowed_voltages = self.low_voltages
            min_v = max(0, 9 - step_size * steps_to_end)
            max_v = 9
        else:  # high range
            allowed_voltages = self.high_voltages
            min_v = 9
            max_v = min(18, 9 + step_size * steps_to_end)
        voltage = min(max(ideal_voltage, min_v), max_v)
        pos = bisect_left(allowed_voltages, voltage)
        if pos < 1:
            pos = 1
        if pos >= len(allowed_voltages):
            pos = len(allowed_voltages) - 1
        before = allowed_voltages[pos - 1]
        after = allowed_voltages[pos]
        before_del_v = abs(before - current_v)
        after_del_v = abs(after - current_v)
        # TODO: if we're satisfied that these RuntimeErrors will never happen, we can cut them for speed?
        if before_del_v > self.max_del_v:
            if after_del_v > self.max_del_v:
                msg = f'Max del V error\n current_v:{current_v} \nbefore:{before} \nafter:{after}'
                msg += f'\ntarget_del_v:{target_del_v}'
                if not self.aggressive:
                    raise RuntimeError(msg)
                else:
                    print(msg)
            return after
        elif after_del_v > self.max_del_v:
            if before_del_v > self.max_del_v:
                msg = f'Max del V error\n current_v:{current_v} \nbefore:{before} \nafter:{after}'
                msg += f'\ntarget_del_v:{target_del_v}'
                if not self.aggressive:
                    raise RuntimeError(msg)
                else:
                    print(msg)
            return before
        else:
            if abs(after_del_v - target_del_v) < abs(before_del_v - target_del_v):
                return after
            else:
                return before

    def digitize_to_nearest(self, voltage):
        """Finds the nearest Himax output voltage, ignoring delta V limits"""
        if self.current_min_v == 0:  # low range
            allowed_voltages = self.low_voltages
        else:  # high range
            allowed_voltages = self.high_voltages
        pos = bisect_left(allowed_voltages, voltage)
        if pos < 1:
            pos = 1
        if pos >= len(allowed_voltages):
            pos = len(allowed_voltages) - 1
        before = allowed_voltages[pos - 1]
        return before

    def find_constrained_voltages(self, v1, del_v2, del_v3, v3, del_v_index):
        """Finds all allowed ways to pick a given voltage that is constrained on both sides"""
        # Pattern is: v1 - del_v2 - v2 - del_v3 - v3
        # Solve for allowed v2. Use average of two del_v results, to put equal error on both.
        if 0 < del_v_index < 512:
            plus_minus = ((v1 + del_v2) + (v3 - del_v3))/2
            minus_minus = ((v1 - del_v2) + (v3 - del_v3))/2
            options = [plus_minus, minus_minus]
        else:
            plus_plus = ((v1 + del_v2) + (v3 + del_v3)) / 2
            minus_plus = ((v1 - del_v2) + (v3 + del_v3)) / 2
            options = [plus_plus, minus_plus]
        # example error case: solve 9V - VN - 0V for delta Vs 5.5V, 0.1V.
        # averaging delta Vs will pull VN out of delta_v range
        options += [(v1 + v3) / 2]
        allowed_options = []
        for option in options:
            # coerce values into voltage limits
            coerced_option = min(max(option, self.current_min_v), self.current_max_v)
            # compute actual del_vs, then don't allow voltages that would exceed either max_del_v
            # check max_del_v both before digitization (to avoid a bug) and after digitization (to verify result)
            # hacky :(
            actual_del_v2 = abs(coerced_option - v1)
            actual_del_v3 = abs(coerced_option - v3)
            if actual_del_v3 < self.max_del_v and actual_del_v2 < self.max_del_v:
                digitized_option = self.digitize_to_del_v_limit(v1, coerced_option - v1, del_v_index)
                digitized_del_v2 = abs(digitized_option - v1)
                digitized_del_v3 = abs(digitized_option - v3)
                if digitized_del_v3 < self.max_del_v and digitized_del_v2 < self.max_del_v:
                    allowed_options += [digitized_option]
        if len(allowed_options) == 0:
            # I think you can't reach this error?
            print(f'v1 = {v1}, del_v2 = {del_v2}, del_v3 = {del_v3}, v3 = {v3}')
            raise RuntimeError(f'Could not find an allowed solution for last voltage.')
        return allowed_options

    def weight_result(self, v_pattern, del_v_start_index, depth):
        """Determines the weight for a given result

        Tentatively deprecated by switch to node weights"""
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
            # --weight by absolute voltage--
            weight += abs(v2 - 9) * self.voltage_factor
        return weight

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
            del_v_index = del_v_index % self.num_rails
            # del_v[1021] = del_v[0], so % num_rails handles rollover case
            if self.short_filter[del_v_index] == 0:
                # Never drive a short - fix delta V of 0
                allowed_voltages = [root.voltage]
            elif del_v_index == self.stop_index - 1:
                # At boundaries, can't select last voltage, so different voltages allowed.
                v_prev = root.voltage
                delv_next = self.target_del_v_pattern[del_v_index]
                delv_last = self.target_del_v_pattern[(del_v_index + 1) % self.num_rails]
                v_last = self.middle_voltage
                allowed_voltages = self.find_constrained_voltages(v_prev, delv_next, delv_last, v_last, del_v_index)
            else:
                allowed_voltages = self.find_voltages(root.voltage, self.target_del_v_pattern[del_v_index], del_v_index)
            children = []
            for voltage in allowed_voltages:
                node_weight = self.weight_node(root, voltage, del_v_index)
                new_root = Node(voltage, root, node_weight)
                children += self.generate_node_list(new_root, depth - 1, del_v_index + 1)
            return children

    def weight_node(self, parent, voltage, del_v_index):
        weight = parent.weight
        actual_del_v = abs(voltage - parent.voltage)
        target_del_v = self.target_del_v_pattern[del_v_index]
        weight += self.error_factor * abs(target_del_v - actual_del_v)
        weight += self.voltage_factor*abs(self.middle_voltage - voltage)
        if del_v_index == self.stop_index - 1:
            wraparound_actual_del_v = abs(voltage - self.middle_voltage)
            wraparound_target_del_v = self.target_del_v_pattern[(del_v_index + 1)%self.num_rails]
            weight += self.error_factor*abs(wraparound_target_del_v - wraparound_actual_del_v)
        return weight

    def solve_one_chunk(self, prev_result, del_v_start_index, depth):
        """Computes the best allowed pattern for the next chunk of pattern, up to depth.

        Assumes that there is not a domain boundary within the chunk, only possibly at the end of the chunk.
        Args:
            prev_result: The previous chunk's results.
            del_v_start_index: First index in the del_v data to be calculated
            depth: How far to calculate for this chunk."""
        # --Set starting conditions--
        start_v = prev_result.v_list[del_v_start_index - 1]
        start_node = Node(start_v, None, prev_result.weight)
        results = []
        # --Get all possible ways to solve the next 'depth' voltages and their weights--
        possible_nodes = self.generate_node_list(start_node, depth, del_v_start_index)
        best_nodes = sorted(possible_nodes, key=lambda x: x.weight)
        # For the num_bests best nodes, track out the solutions as new results
        for node in best_nodes[0:self.num_bests]:
            new_v_list_subset = self.read_upwards_from_node(node)
            new_v_list = prev_result.v_list.copy()
            new_v_list[del_v_start_index: del_v_start_index + depth] = new_v_list_subset
            # If we need to weight by FFT, that code would go here.
            weight = node.weight
            results += [Result(weight, new_v_list)]
        # --Pre-sort results and return the best num_bests of them--
        # best_results = sorted(results, key=lambda x: x.weight)
        # return best_results[0:self.num_bests]
        return results

    def solve_one_half(self, start_index, stop_index, prev_results):
        """Solves a portion of the pattern (either odd or even)"""
        best_results = prev_results
        current_del_v_index = start_index
        if 1 <= start_index < 511:
            # Set voltage limits based on ODD or EVEN domain
            self.current_min_v = self.min_voltage
            self.current_max_v = self.middle_voltage
        else:
            self.current_min_v = self.middle_voltage
            self.current_max_v = self.max_voltage
        # If rail 1021 is shorted to rail 1, can't apply any voltage.
        # need at least self.end_steps_count non-shorted delta Vs in a row.
        # Iterate backwards until you find a group of delta Vs at least this length.
        # while self.short_filter[self.stop_index % self.num_rails] == 0 or self.short_filter[self.stop_index - 1] == 0:
        self.stop_index = stop_index
        # a one-time construct to aid in solving wraparound issue at index 1021. used so index 1022 = index 0.
        doubled_list = list(self.short_filter) + list(self.short_filter)
        while 0 in doubled_list[self.stop_index - self.end_steps_count + 1:self.stop_index + 1]:
            self.stop_index -= 1
            if self.stop_index - start_index < self.end_steps_count:
                num_shorts = self.short_filter.count(0)
                msg = f'Shorting is terminal. With {num_shorts} shorts, cannot find a block of '
                msg += f'{self.end_steps_count} non-shorted channels.\nHe\'s dead, Jim. :('
                if not self.aggressive:
                    raise RuntimeError(msg)
                else:
                    print('DANGER: Aggressive mode is now driving shorted channels. Too many shorts for clean solution.')
                    self.stop_index = stop_index
                    self.short_filter = [1]*len(self.short_filter)
                    break
        # after first loop, tree_depth changes to self.max_depth
        tree_depth = (self.stop_index - current_del_v_index) % self.max_depth
        while current_del_v_index < self.stop_index - 1:
            new_bests = []
            for result in best_results:
                new_bests += self.solve_one_chunk(result, current_del_v_index, tree_depth)
            best_results = sorted(new_bests, key=lambda x: x.weight)
            best_results = best_results[0:self.num_bests]
            current_del_v_index += tree_depth
            tree_depth = self.max_depth
        return best_results

    def check_result(self, v_pattern, short_filter):
        """Double checks that a given output voltage pattern has the right min voltage, max voltage, and max delv

         @Erik thinks this will never raise an error, so come ask me if this happens."""
        codes = self.h.get_codes(v_pattern)
        driver_pins = [2 * r - 1 if r <= 511 else
                       2 * r - 1022
                       for r in range(1, 1 + self.num_rails)]
        high_vec = [1 - (d % 2) for d in driver_pins]
        actual_v_pattern = np.asarray(self.h.get_voltages(codes=codes,
                                                          high_vec=high_vec))
        actual_del_v = self.vtodelv(actual_v_pattern)
        for i in range(0, len(actual_v_pattern)):
            if 0 < i < 511:
                # ODD channels, 0 to 9V
                min_v = self.min_voltage
                max_v = self.middle_voltage
            else:
                # EVEN channels, 9 to 18V
                min_v = self.middle_voltage
                max_v = self.max_voltage
            if not min_v <= actual_v_pattern[i] <= max_v:
                raise RuntimeError(f'Voltage on rail {i+1} out of range, at {actual_v_pattern[i]}! Ask Erik.')
            if not abs(actual_del_v[i]) <= self.max_del_v:
                msg = f'Delta V {i} out of range, at {actual_del_v[i]}! Ask Erik.'
                if not self.aggressive:
                    raise RuntimeError(msg)
                else:
                    print(msg)
            if short_filter[i] == 0 and not actual_del_v[i] == 0:
                start_rail = (i - 1) % self.num_rails + 1
                end_rail = i+1
                msg = f'Optimizer tried to drive {start_rail} to {end_rail}, but this is shorted!'
                if not self.aggressive:
                    raise RuntimeError(msg)
                else:
                    print(msg)


class Node:
    """Represents a single possible choice of rail voltage at some depth"""
    def __init__(self, voltage, parent, weight):
        self.voltage = voltage
        self.parent = parent
        self.weight = weight

    def __str__(self):
        # Define how this object is converted to a string, useful for debugging
        return f'Node: V={self.voltage:0.2f}, w={self.weight}'


class Result:
    """Represents a list of voltages that solve the problem, weighted by quality.  Lower weight is better."""
    def __init__(self, weight, v_list):
        self.weight = weight
        self.v_list = v_list

    def __str__(self):
        return f'Result: w={self.weight}, v_list={self.v_list}'

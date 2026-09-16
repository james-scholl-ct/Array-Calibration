import numpy as np
from lcm_board.voltage_pattern import VoltagePatternGenerator


class VoltageTranslator:
    def __init__(self,
                 min_voltage,
                 max_voltage,
                 max_del_v,
                 error_factor=1,
                 voltage_factor=0.01):
        """Initializes and runs an algorithm to get a specific v pattern for a target delta v.

        Args:
            min_voltage: minimum driving voltage
            max_voltage: maximum driving voltage
            max_del_v: maximum allowed delta v
            error_factor: how much weight to assign voltage errors
            voltage_factor: how much weight to assign to mean voltage"""
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.max_del_v = max_del_v
        self.last_voltage = 4.5
        self.error_factor = error_factor
        self.voltage_factor = voltage_factor

    def vtodelv(self, v):
        """
        Converts v to delv. delv[i] is defined as abs(v[i] - v[i-1])

        Args:
            v: numpy array of v values

        Returns:
            delv: numpy array of delv values

        """
        delv = np.zeros(len(v))
        for i, val in enumerate(delv):
            delv[i] = abs(v[i] - v[i-1])
        return delv

    def delvtov(self, delv, max_depth=3, num_bests=3, error_factor=1, voltage_factor=0.01):
        """Converts a delta v pattern into a v pattern.

        Considers repeated trees of options to find an optimal result based on cost factors.
        Args:
            delv: desired delta V pattern
            max_depth: depth of each tree
            num_bests: number of best results to keep from each tree
            error_factor: how much weight to assign to delta_v errors
            voltage_factor: how much weight to assign to minimizing the voltage

        Returns:
            v_pattern: a list of rail voltages that best approximates the target delta_v
            """
        self.max_depth = max_depth
        self.num_bests = num_bests
        self.error_factor = error_factor
        self.voltage_factor = voltage_factor
        self.target_pattern = delv
        v_pattern = self.run_algorithm()
        self.check_result(v_pattern, self.target_pattern)
        return v_pattern

    def find_voltages(self, current_voltage, target_del_v):
        """Finds all allowed ways to generate the next delta V.
        Returns a list of allowed voltages, clipped to allowed limits."""
        # clip located voltages to the allowed limits.
        # when I get around to adding perturbations, do that here.

        # plus option:
        plus_voltage = current_voltage + target_del_v
        plus_voltage = min(plus_voltage, self.max_voltage)
        # minus options:
        minus_voltage = current_voltage - target_del_v
        minus_voltage = max(minus_voltage, self.min_voltage)
        return [plus_voltage, minus_voltage]

    def weight_result(self, node_voltage, target_del_v, actual_del_v):
        # assigns a weight to a node
        # weight by clipping.  Later, this will be a function : f(V, error)
        del_v_error = abs(actual_del_v - target_del_v)
        weight = del_v_error * self.error_factor
        if target_del_v == 0:
            # If target_del_v is 0, might be a short.
            # we really don't want to drive those hard, so weight like crazy.
            weight += del_v_error*1000
        # also weight by absolute voltage
        weight += node_voltage * self.voltage_factor
        return weight

    def find_last_voltages(self, voltage):
        # The choices for the last voltage are constrained by the wraparound condition.
        # v202 - delv203 - v203 - delv204 - v204
        v_204 = self.last_voltage
        v_202 = voltage
        delv_203 = self.target_pattern[-2]
        delv_204 = self.target_pattern[-1]

        # average of two del_v results.  Puts equal error on both.
        # Assumes that v1 is 0
        plus_minus = ((v_204 + delv_204) + (v_202 - delv_203))/2
        plus_plus = ((v_204 + delv_204) + (v_202 + delv_203))/2
        minus_plus = ((v_204 - delv_204) + (v_202 + delv_203))/2
        minus_minus = ((v_204 - delv_204) + (v_202 - delv_203))/2
        # When I start in on perturbations, also try two some other options:
        # only matters if weight is not linear with error
        options = [plus_minus, plus_plus, minus_plus, minus_minus]
        # error case: 9V - 5.5V - V_203 - 0.1V - 0V
        # averaging will pull VN out of delta_v range
        allowed_options = []
        options += [(v_202 + v_204)/2]
        for option in options:
            # coerce values into voltage limits
            coerced_option = min(max(option, self.min_voltage), self.max_voltage)
            # coerce values into del_v limits?
            actual_del_v_203 = abs(coerced_option - v_202)
            actual_del_v_204 = abs(coerced_option - v_204)
            if actual_del_v_204 < self.max_del_v and actual_del_v_203 < self.max_del_v:
                allowed_options += [coerced_option]
        if len(allowed_options) == 0:
            # I think you can't reach this error?
            raise RuntimeError(f'Could not find an allowed solution for last voltage.')
        return allowed_options

    def generate_node_list(self, root, depth, del_v_index):
        # Recursively generate a full list of all options, up to depth
        if depth == 0:
            return [root]
        else:
            target_del_v = self.target_pattern[del_v_index]
            if target_del_v == 0:
                # Never drive a short - force voltage to be same as previous rail
                allowed_voltages = [root.voltage]
            elif del_v_index == len(self.target_pattern) - 2:
                # This is the wraparound condition. Different voltages allowed.
                allowed_voltages = self.find_last_voltages(root.voltage)
            else:
                allowed_voltages = self.find_voltages(root.voltage, target_del_v)
            children = []
            for voltage in allowed_voltages:
                actual_del_v = abs(voltage - root.voltage)
                weight = root.weight + self.weight_result(root.voltage, target_del_v, actual_del_v)
                if del_v_index == len(self.target_pattern) - 2:
                    wraparound_del_v = abs(voltage - self.last_voltage)
                    wrap_error = abs(wraparound_del_v - self.target_pattern[-1])
                    weight += wrap_error * self.error_factor
                new_root = Node(weight, voltage, root)
                children += self.generate_node_list(new_root, depth - 1, del_v_index + 1)
            return children

    @staticmethod
    def get_min_items(node_list, num_items):
        """Find the {num_items} nodes in {node_list} that have the lowest weight"""
        # todo: work on sorting this list in the fastest way.  Maybe 'operator'?
        sorted_list = sorted(node_list, key=lambda x: x.weight)
        return sorted_list[0:num_items]

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

    def assess_one_tree(self, prev_result, depth, del_v_index):
        """ Assesses all nodes below root to given depth
        Returns a list of the best (self.num_bests) results"""
        distance_to_list_end = len(self.target_pattern) - del_v_index - 1
        limited_depth = min(depth, distance_to_list_end)
        node_list = self.generate_node_list(prev_result.node, limited_depth, del_v_index)
        best_nodes = self.get_min_items(node_list, self.num_bests)
        new_results = []
        for node in best_nodes:
            v_list = prev_result.v_list + self.read_upwards_from_node(node)
            node.parent = None
            new_results.append(Result(node, v_list))
        return new_results

    def run_algorithm(self):
        initial_root = Node(0, self.last_voltage, None)
        end_index = len(self.target_pattern) - 1
        end_length = 1
        while self.target_pattern[end_index] == 0:
            end_index -= 1
            end_length += 1
        best_results = [Result(initial_root, [])]
        del_v_index = 0
        # Solving a fencepost problem: first tree is shortened.
        # for example: tree of total depth 14 and max depth 6 will run three times: depth 2, 6, 6.
        # I want the '2' value at the front so the algorithm considers more options during wraparound condition.
        tree_depth = end_index % self.max_depth
        while del_v_index < end_index - 1:
            new_bests = []
            for result in best_results:
                new_bests += self.assess_one_tree(result, tree_depth, del_v_index)
            best_results = sorted(new_bests, key=lambda x: x.node.weight)
            best_results = best_results[0:self.num_bests]
            del_v_index += tree_depth
            tree_depth = self.max_depth
        return best_results[0].v_list + [self.last_voltage] * end_length

    @staticmethod
    def assess_results(target_del_v_pattern, v_pattern):
        """Calculates some details about the algorithm results"""
        actual_del_v = -1*np.ones(len(v_pattern))
        error = -1*np.ones(len(v_pattern))
        for i in range(len(v_pattern)):
            actual_del_v[i] = abs(v_pattern[i] - v_pattern[i-1])
            error[i] = target_del_v_pattern[i] - actual_del_v[i]

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
                raise RuntimeError(f'Error: Optimizer tried to drive delta V {i} but it is shorted')


class Node:
    def __init__(self, weight, voltage, parent):
        self.weight = weight
        self.voltage = voltage
        self.parent = parent

    def __str__(self):
        return f'Node: w={self.weight}, V={self.voltage}'


class Result:
    def __init__(self, node, v_list):
        self.node = node
        self.v_list = v_list


if __name__ == '__main__':
    """Test code to see if it's working"""
    # sample_v = [0, 1.1, 4.5, 0, 5.9, 4.1, 9, 4.3, 9, 4.3, 9, 4.4, 9, 4.4, 9, 3.5, 7.5, 4.5]
    vpg = VoltagePatternGenerator()
    vt = VoltageTranslator(min_voltage=0, max_voltage=9, max_del_v=6.1)
    sample_v = vpg.logit(m=8, )[0]
    # sample_v = [0, 1.1, 4.5, 0, 5.9, 4.1, 9, 9, 4.3, 9, 4.4, 9, 4.4, 9, 3.5, 7.5, 4.5]
    # sample_v = [0, 1.1, 4.5, 3]
    print(f'Mean Input Voltage: {np.array(sample_v).mean()}')
    sample_del_v = vt.vtodelv(sample_v)
    sample_del_v[-1]=0
    sample_del_v[-2] = 0
    sample_del_v[-3] = 0
    output = vt.delvtov(sample_del_v)
    print(f'Mean Output Voltage: {np.array(output).mean()}')
    # vt.assess_results()
    # print('results:')
    # print(f'target del v: {np.array(sample_del_v)}')
    # print(f'actual del v: {vt.actual_del_v}')
    # print(f'error:        {vt.error}')
    # print(f'target v: {np.array(sample_v)}')
    # print(f'actual v: {np.array(vt.v_pattern)}')

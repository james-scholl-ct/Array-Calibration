'''This module generates a Verilog file for the coefficient ROM in Lotus.
'''

import csv
import jinja2
import math
import numpy as np
import os
import random
import sys
import textwrap

from python_tools.himax_model import HimaxModel

_N_COEFFS_PER_ANGLE = 204

_N_ANGLES = 128

_N_WORDS = _N_ANGLES * _N_COEFFS_PER_ANGLE * 8 // 32

_ADDR_WIDTH = math.ceil(math.log(_N_WORDS, 2))

_FILENAME = 'lotus_rom.gen.v'

_DEFAULT_VOLTAGE = 9.

_VPATTERN_FILEPATH = os.path.join(os.path.dirname(__file__),
                                  'data',
                                  '')

_JINJA2_TEMPLATE_ROM = textwrap.dedent('''
    `include "lotus_defines.v"
    `timescale 1 ns / 1 ps

    module lotus_rom #(
        parameter integer DEPTH = {{n_words}},
        parameter integer WIDTH = 32
    )
    (
        input   wire                                        CLK,
        input   wire                                        EN,
        input   wire    [$clog2(DEPTH) - 1 : 0]             ADDR,

        output  wire    [WIDTH - 1 : 0]                     DATA
    );


        (* rom_style = "block" *) reg [WIDTH - 1 : 0] data_pp;

        assign DATA = data_pp;

        always @(posedge CLK)
        begin
            if( EN )
            begin
                case(ADDR)
                {%- for word in mem %}
                    {{addr_width}}'d{{loop.index0}}: data_pp <= 32'h{{word}};
                    {%- if loop.index0 % table_size == 0 -%}
                        {{ ' // Table %d'|format(loop.index0 / table_size) }}
                    {%- endif -%}
                {%- endfor %}
                    default: data_pp <= 32'h00_00_00_00;
                endcase
            end
        end

    endmodule

''')


def fold_array(arr, width):
    # With width = 4, list(range(7)) maps to [[0, 1, 2, 3], [4, 5, 6]]
    depth = (len(arr) - 1) // width + 1
    return [arr[width*i : width*(i+1)] for i in range(depth)]

def angles_to_memh(struc, filename=_FILENAME):
    with open(filename, 'w') as f:
        for angle in struc:
            if len(angle) != _N_COEFFS_PER_ANGLE:
                raise RuntimeError("Invalid length {}.".format(angle))
            folded = fold_array(angle, 4)
            for word in folded:
                s = '_'.join(["{:02x}".format(b) for b in reversed(word)])
                f.write(s + '\n')

def angles_to_verilog(struc, filename=_FILENAME):
    hex_strs = []
    for angle in struc:
        if len(angle) != _N_COEFFS_PER_ANGLE:
            raise RuntimeError("Invalid length {}.".format(angle))
        folded = fold_array(angle, 4)
        for word in folded:
            s = '_'.join(["{:02x}".format(b) for b in reversed(word)])
            hex_strs.append(s)

    t = jinja2.Template(_JINJA2_TEMPLATE_ROM)
    out = t.render(mem=hex_strs,
                   addr_width=_ADDR_WIDTH,
                   table_size=(_N_WORDS / _N_ANGLES),
                   n_words=_N_WORDS)
    with open(filename, 'w') as f:
        f.write(out)

def parse_csv(filename, filepath=_VPATTERN_FILEPATH):
    headers = []
    voltages = _DEFAULT_VOLTAGE * np.ones((_N_ANGLES, _N_COEFFS_PER_ANGLE))
    path = filepath + filename
    with open(path, 'r') as f:
        parser = csv.reader(f)
        for i_row, row in enumerate(parser):
            if i_row == 0:
                # header row
                headers = row[:]
                for i_col, col in enumerate(headers):
                    table_idx = int(col)
                    if not 0 <= table_idx < _N_ANGLES:
                        msg = "Invalid table index {} in the {}-th column."
                        msg = msg.format(table_idx, i_col+1)
                        raise RuntimeError(msg)
            else:
                # data row
                for i_col, col in enumerate(row):
                    table_idx = int(headers[i_col])
                    coeff_idx = i_row - 1 # subtract header row
                    voltages[table_idx][coeff_idx] = float(col)
    return voltages

if __name__ == '__main__':
    #codes = [[random.randint(0, 0xff) for _ in range(_N_COEFFS_PER_ANGLE)]
    #         for _ in range(_N_ANGLES)]
    #angles_to_memh(codes)
    #angles_to_verilog(codes)

    voltages = parse_csv(sys.argv[1])
    vgma = {'vgma1': 18, 'vgma2': 18, 'vgma9':  9, 'vgma10': 9,
            'vgma11': 9, 'vgma12': 9, 'vgma19': 0, 'vgma20': 0}
    h = HimaxModel(vgma)
    codes = [h.get_codes(v_pat) for v_pat in voltages]
    angles_to_verilog(codes)
    #for i, j in zip(voltages[10], codes[10]):
    #    print(i, j, hex(j))

'''build_sim_sources.py

This module processes the output from vivado's export_simulation command and
then runs compile and elaboration for subsequent simulation.
'''

import argparse
import jinja2
import os
import shutil
import subprocess
import sys
import textwrap


_JINJA2_TEMPLATE_PRJ_VERILOG = textwrap.dedent('''
    {% for lib in verilog_files -%}
    {% for inc_dir in verilog_files[lib] -%}
    verilog {{lib}} \\
        {%- if inc_dir %}
        --include {{inc_dir}} \\
        {%- endif %}
        {% for vlog_filepath in verilog_files[lib][inc_dir] -%}
            {{vlog_filepath}} {% if not loop.last %}\\{% endif %}
        {% endfor %}
    {% endfor -%}
    {% endfor -%}
    nosort

''')

_JINJA2_TEMPLATE_PRJ_VHDL = textwrap.dedent('''
    {% for lib in vhdl_files -%}
    vhdl {{lib}} \\
        {% for vhdl_filepath in vhdl_files[lib] -%}
            {{vhdl_filepath}} {% if not loop.last %}\\{% endif %}
        {% endfor %}
    {% endfor -%}
    nosort

''')

_XVLOG_LIBS = ('axi_vip_v1_1_5',
               'processing_system7_vip_v1_0_7',
               'xilinx_vip')

_XELAB_LIBS = ('axi_bram_ctrl_v4_1_1',
               'xil_defaultlib',
               'lib_pkg_v1_0_2',
               'lib_srl_fifo_v1_0_2',
               'lib_fifo_v1_0_13',
               'axi_datamover_v5_1_21',
               'axi_sg_v4_1_12',
               'axi_cdma_v4_1_19',
               'axi_infrastructure_v1_1_0',
               'axi_vip_v1_1_5',
               'processing_system7_vip_v1_0_7',
               'generic_baseblocks_v2_1_0',
               'axi_register_slice_v2_1_19',
               'fifo_generator_v13_2_4',
               'axi_data_fifo_v2_1_18',
               'axi_crossbar_v2_1_20',
               'lib_cdc_v1_0_2',
               'proc_sys_reset_v5_0_13',
               'axi_protocol_converter_v2_1_19',
               'xilinx_vip',
               'unisims_ver',
               'unimacro_ver',
               'secureip',
               'xpm')

_XVLOG_DEFINES = ('NO_DEFAULT_NETTYPE_NONE',)


def _subprocess_lines(command):
    '''
    Helper to split multiline output into a list of lines.
    '''
    output = subprocess.getoutput(command).strip()
    return output.split('\n')


def get_git_root():
    """
    Returns the path of  root of the git repository containing the current
    working directory
    """
    cmd = 'git rev-parse --show-toplevel'
    output = _subprocess_lines(cmd)
    return os.path.abspath(output[0])


def xcompile(**kwargs):
    # verilog
    defines = ' '.join(["-d {}".format(define) for define in _XVLOG_DEFINES])
    libs = ' '.join(["-L {}".format(lib) for lib in _XVLOG_LIBS])
    options = ' '.join(["-{} {}".format(k, v or '') for k, v in kwargs.items()])
    cmd = ' '.join(['xvlog', '-relax', defines, libs, options, '-prj vlog.prj'])
    subprocess.run(cmd, shell=True)

    # vhdl
    cmd = ' '.join(['xvhdl', '-relax', '-prj vhdl.prj'])
    subprocess.run(cmd, shell=True)


def xelab(module, **kwargs):
    libs = ' '.join(["-L {}".format(lib) for lib in _XELAB_LIBS])
    options = ' '.join(["-{} {}".format(k, v or '') for k, v in kwargs.items()])
    cmd = ' '.join(['xelab',
                    '-relax',
                    '-debug typical',
                    libs,
                    options,
                    '-snapshot tb_{}'.format(module),
                    'xil_defaultlib.tb_{}'.format(module),
                    'xil_defaultlib.glbl'])
    subprocess.run(cmd, shell=True)


def xsim(module):
    cmd = 'xsim -tclbatch sim.tcl tb_{}'.format(module)
    subprocess.run(cmd, shell=True)


def open_sim(module):
    wavefile_name = 'tb_{}_behav.wcfg'.format(module)
    wavefile = os.path.join(get_git_root(), 'sim_src', wavefile_name)
    cmd = 'xsim -gui -view {} tb_{}'.format(wavefile, module)
    subprocess.run(cmd, shll=True)


def move_sim_files(src, dst):
    shutil.copytree(os.path.join(src, 'xsim', 'srcs'),
                    os.path.join(dst, 'srcs'))
    shutil.copy(os.path.join(src, 'xsim', 'cmd.tcl'),
                os.path.join(dst, 'sim.tcl'))
    shutil.copy(os.path.join(src, 'xsim', 'glbl.v'), dst)
    shutil.copy(os.path.join(src, 'xsim', 'xsim.ini'), dst)
    shutil.copy(os.path.join(src, 'vlog.prj'), dst)
    shutil.copy(os.path.join(src, 'vhdl.prj'), dst)


def make_fresh_directory(dir_name):
    """
    Creates the specified folder, deleting it first if it already exists.
    The folder is created in the current working directory.
    """
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)


def generate_prj_files(sim_export_dir):
    filepath = os.path.join(sim_export_dir,
                            'xsim',
                            'file_info.txt')
    with open(filepath, 'r') as f:
        lines = f.readlines()

    libs = []
    verilog_files = {}
    vhdl_files = {}
    map = ['file', 'language', 'lib', 'rel_path', 'include_dir']
    for line in lines:
        dikt = {map[i]:token
                for i, token in enumerate(line.strip().split(','))}
        lib = dikt['lib']
        include_dir = dikt.get('include_dir', '=').split('=')[1]
        if 'verilog' in dikt['language'].lower():
            if not verilog_files.get(lib, None):
                verilog_files[lib] = {}
            if not verilog_files[lib].get(include_dir, None):
                verilog_files[lib][include_dir] = []
            verilog_files[lib][include_dir].append(dikt['rel_path'])
        elif 'vhdl' in dikt['language'].lower():
            if not vhdl_files.get(lib, None):
                vhdl_files[lib] = []
            vhdl_files[lib].append(dikt['rel_path'])
        else:
            raise Exception("unrecognized file type for <{}>".format(line))

    t = jinja2.Template(_JINJA2_TEMPLATE_PRJ_VERILOG)
    out = t.render(verilog_files=verilog_files)
    outfile = os.path.join(sim_export_dir, 'vlog.prj')
    with open(outfile, 'w') as f:
        f.write(out)

    t = jinja2.Template(_JINJA2_TEMPLATE_PRJ_VHDL)
    out = t.render(vhdl_files=vhdl_files)
    outfile = os.path.join(sim_export_dir, 'vhdl.prj')
    with open(outfile, 'w') as f:
        f.write(out)


def get_sim_export_dir(module):
    return os.path.join(get_git_root(),
                        '_build_{}'.format(module),
                        '_sim_export_{}'.format(module))


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-m', '--module',
                        required=True,
                        help='e.g. orchid')

    parser.add_argument('-d', '--dir',
                        default='sim',
                        help='directory in which to run simulations')

    return parser.parse_args(argv)


def build(module, where=None, xcompile_opts={}, xelab_opts={}):
    sim_export_dir = get_sim_export_dir(module)
    generate_prj_files(sim_export_dir)
    cwd = os.getcwd()
    if where is None:
        move_sim_files(sim_export_dir, cwd)
    else:
        make_fresh_directory(where)
        move_sim_files(sim_export_dir, where)
        os.chdir(where)
    xcompile(**xcompile_opts)
    xelab(module, **xelab_opts)
    return cwd


def main():
    args = _parse_args(sys.argv[1:])
    return_to_here = build(args.module, where=args.dir)
    xsim(args.module)
    os.chdir(return_to_here)


if __name__ == '__main__':
    main()

import os
import subprocess


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

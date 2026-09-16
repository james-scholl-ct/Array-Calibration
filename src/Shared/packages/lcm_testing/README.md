lcm-testing
===========
`lcm-testing` is a repository of various files used during testing the LCM chips used in the Carillon project.

Prerequisites
-------------
### External
* Python 3.7
* `git`


Installing
----------
This package has only been tested on Windows and it is unknown whether LCM modules or thorlabs_kinesis modules will work on other operating systems, such as Linux.
To install a local copy of this tool:
1. `pip install git+https://bitbucket.geost.com/scm/car/lcm-testing.git`
2. Installation will install the following packages:
* `numpy`
* `scipy`
* `matplotlib`
* `tkinter`
* `functools`
* `pandas`
* `imutils`
* `cv2`
* `zernike`
3. To use thorlabs_kinesis, the `pythonnet` package must be installed from conda-forge:
* `conda install -c conda-forge pythonnet`

### Updating
To update this tool:
1. Uninstall the package using: `pip uninstall lcm-testing`
2. Reinstall the package using: `pip install git+https://bitbucket.geost.com/scm/car/lcm-testing.git`

### Usage

Examples still to come...
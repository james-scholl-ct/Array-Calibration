import setuptools
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
exec(open('./version.py').read())
setuptools.setup(
    name='lcm-testing',
    version=__version__,
    author='Rick Patton',
    author_email='rick.patton@geost.com',
    description='Carillon LCM testing and analysis utilities',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://bitbucket.geost.com/projects/CAR/repos/lcm-testing/',
    project_urls = {
        
    },
    license='GEOST Proprietary',
    packages=setuptools.find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'pandas',
                      'imutils',
                      'opencv-python',
                      'zernike',
                      'h5py',
                      # Private Repositories
                      'lcm_board @ git+https://bitbucket.geost.com/scm/car/lcm_board.git',
                      'lcm_voltage_patterns @ git+https://bitbucket.geost.com/scm/car/lcm_voltage_patterns.git',
                      'himax_model @ git+https://bitbucket.geost.com/scm/car/himax_model.git',
                      'zynq_dev @ git+https://bitbucket.geost.com/scm/car/zynq_dev.git'
                    ],
    python_requires="<3.8, >3.0"
   )
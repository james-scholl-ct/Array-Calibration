from setuptools import setup, find_packages


# https://setuptools.readthedocs.io/en/latest/index.html
setup(name='zynq_dev',
      version='0.1.0',
      description='Python libraries to interact with Lumotive\'s Zynq FPGA hardware',
      url='bitbucket.org/lumotive/zynq_dev',
      author='Tyler Williamson',
      author_email='tyler.williamson@lumotive.com',
      packages=find_packages(),
      package_data={'python_tools': ['yaml/*.yml']},
      install_requires=['pyyaml', 'paramiko', 'scp'],
      zip_safe=False)

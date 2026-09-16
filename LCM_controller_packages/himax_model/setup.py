from setuptools import setup, find_packages


# https://setuptools.readthedocs.io/en/latest/index.html
setup(name='himax_model',
      version='0.0.1',
      description='Python libraries to model HX8175 driver chip output',
      url='bitbucket.org/lumotive/himax_model',
      author='Tyler Williamson',
      author_email='tyler.williamson@lumotive.com',
      packages=find_packages(),
      package_data={'himax_model': ['yaml/*.yml']},
      install_requires=['pyyaml', 'numpy'],
      zip_safe=False)

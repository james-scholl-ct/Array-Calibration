from setuptools import setup, find_packages

setup(
    name='lcm_board',
    version='0.6.2',
    packages=find_packages(),
    url='https://bitbucket.org/lumotive/lcm_board',
    install_requires = [
        'scipy',
        'matplotlib',
        "paramiko",
        "scp",
        "pyyaml",
        "numpy",
        "jinja2",
        "pandas"
    ],
    license='',
    author='',
    author_email='',
    description='Classes and methods for interfacing with the Lotus board',
    include_package_data=True
)

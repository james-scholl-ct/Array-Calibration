from setuptools import setup, find_packages

setup(
    name='lcm_voltage_patterns',
    version='0.0.4',
    packages=find_packages(),
    url='https://bitbucket.org/lumotive/lcm_voltage_patterns',
    python_requires=">=3.5",
    install_requires=[
        'pandas',
    ],

    license='',
    author='',
    author_email='',
    description='',
    include_package_data=True,
    package_data={
        "": ["*.csv"]
    }
)

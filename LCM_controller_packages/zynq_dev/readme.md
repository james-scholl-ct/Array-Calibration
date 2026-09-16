# Preliminaries

## Setting up your computer

### Downloading PuTTY
Download the puTTY executable `putty.exe` from [here](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html). The exe file is preferred over the installer because it is a standalone application that can be relocated.

### Installing Silicon Labs USB-to-UART driver
Download the Windows 10 universal virtual COM port (VCP) driver [here](https://www.silabs.com/products/development-tools/software/usb-to-uart-bridge-vcp-drivers). The version at the time of writing is v10.1.3. Unzip the folder and run `CP210xVCPInstaller_x64.exe`.

### Installing Vivado
1. Download the full product installation of Vivado Design Suite (version 2019.1 at the time of writing) from https://www.xilinx.com/support/download.html. The link is called "Vivado HLx 2019.1: WebPACK and Editions - Windows Self Extracting Web Installer". The installation will take about 90 minutes to complete.
2. Install licenses: TODO

### Installing Microzed board definition files for Vivado
The board definition files can be downloaded here: http://www.zedboard.org/support/documentation/1519. At the time of writing, the most recent version was "MicroZed Board Definition Install for Vivado 2015.3 through 2017.4". Follow the instructions contained in the downloaded zip folder.

### Setting up the environment
Please follow the instructions in the readme markdown file for the [git_bash_setup](https://bitbucket.org/lumotive/git_bash_setup) repo. After following those instructions, you should have a clone of that repo, which is currently hosted at `git@bitbucket.org:lumotive/git_bash_setup.git`, in the directory `~/git_repos/git_bash_setup`. If you haven't done so already, you will need to create a personalized `.bashrc` file by copying `default/.bashrc` into `<git_username>/.bashrc` (i.e. substituting your git username). Assuming you have a personalized `.bashrc` file, add the following lines to that file:

```bash
# For 2018.2, see /c/Xilinx/Vivado/2018.2/env.sh
#export PATH='/c/Xilinx/Vivado/2018.2/lib/win64.o':$PATH
#export PATH='/c/Xilinx/Vivado/2018.2/bin':$PATH
#export XILINX_VIVADO='/c/Xilinx/Vivado/2018.2'

# For 2019.1, see /c/Xilinx/Vivado/2019.1/.settings64-Vivado.sh
export PATH='/c/Xilinx/Vivado/2019.1/lib/win64.o':$PATH
export PATH='/c/Xilinx/Vivado/2019.1/bin':$PATH
export XILINX_VIVADO='/c/Xilinx/Vivado/2019.1'
```

Source the setup script to see the changes.
```bash
. ./setup.sh
```

### Downloading sources
Clone the `zynq_dev` repository by running the following in the Git Bash terminal:

```bash
cd ~/git_repos
git clone git@bitbucket.org:lumotive/zynq_dev.git
```

## Setting up Microzed

Most of the documentation for Microzed can be found on Avnet's website [here](http://zedboard.org/support/documentation/1519).

### Unboxing

1. Setup the hardware
    1. Set jumpers JP3 / JP2 / JP1 in the DOWN / DOWN / UP positions to enable booting from the SD card. DOWN connects pins 2 and 3 while UP connects pins 1 and 2 (see the silkscreen and/or schematic if needed).
    2. \[Optional\] Your Microzed may not have shipped with Wind River Pulsar Linux loaded onto the included MicroSD card. If so, go to the Google Drive folder "Team Drives/Engineering/EE/Lotus/microzed/wind_river_pulsar" and follow the instructions in readme.txt.
    3. Insert the MicroSD card (included with Microzed) into the slot on the back of the Microzed board.
    4. Connect Microzed to your LAN via an ethernet cable.
    5. Connect your computer to Microzed with the USB cable (included with Microzed).
2. Connect to Microzed over serial port
    1. Open the Windows Device Manager via the control panel or by running `devmgmt.msc` in a windows command prompt.
    2. Under "Ports (COM & LPT)", identify the COM port number for "Silicon Labs CP210x USB to UART Bridge" (e.g. COM4).
    3. Close the device manager.
    4. Open puTTY and open a serial connection to the identified COM port. Use a Speed/Baud of 115200.
    5. Login to Microzed with username `root` and password `incendia`.
3. Change the hostname
    1. Type `ifconfig` and record the MAC address (HWaddr) for the "br0" adapter.
    2. Change the host name by typing `echo dl-microzed > /etc/hostname`, replacing the six x's with the last six nybbles of the identified MAC address.
    3. Add the following to /etc/rc.local:
         ip link set eth0 down
         ip link set eth0 hw 02:46:8a:xx:xx:xx # where xx:xx:xx matches the last six nybbles of the MAC address
         route add default gw 192.168.128.1 eth0
         # set a static IP address for the microzed.
         ip addr add 192.168.128.xxx # where xxx is available on the network.
         ip link set eth0 up
         exit 0
    4. Type `reboot` and login again. Ensure that the host name is updated.
    5. Close the puTTY session.

4. Verify SSH connection
    1. Use Git Bash (or puTTY) to open an SSH connection to `root@microzed-xx-xx-xx`, again replacing the x's.
    2. Run a test command like `pwd`.

### Creating an SSH access key

Bitbucket implements "access keys", also called "deployment keys" to enable anonymous, read-only access to repositories. We will create an access key for the Microzed. Open an SSH connection to Microzed using Git Bash (or puTTY) and do the following:

```bash
cd ~/.ssh
ssh-keygen
# press Enter three times for the following prompts
cat id_rsa.pub
```

Copy the output of the final command (i.e. "ssh-rsa ... root@microzed-xx-xx-xx") to the clipboard. Go to the repository in Bitbucket, [here](https://bitbucket.org/lumotive/zynq_dev), and click "Settings" (requires admin permissions), "Access keys", and "Add key". Paste the copied key into the "Key" field and enter the host name for the "Label" field (e.g. root@microzed-xx-xx-xx).

### Downloading sources

Open an SSH connection to Microzed using Git Bash (or puTTY) and do the following:

```bash
mkdir -p ~/git_repos
cd ~/git_repos
git clone git@bitbucket.org:lumotive/zynq_dev.git
cd zynq_dev
```

# Building

In general, the following instructions apply equally to 'delorean', 'lotus' and 'orchid' designs.

## Build and upload the FPGA bitstream
From a Git Bash shell on your computer, run the following: `. bash_tools/build_orchid.sh`.

Vivado is executed in batch mode using scripts from `tcl` and sources from `rtl_src`, `constraints`, and `sim_src`. Generated Vivado projects are stored in `vivado_projects` and all other output products are stored in `_build_orchid`, including the bitstream, which is called `orchid.bit`. Upload the bitstream to the Microzed by running the following:

```bash
. python_tools/setup.py  # only need to do this once after opening the shell
python python_tools/zynq_connection.py <hostname> orchid
```

This script will copy the bitstream (from `_build_orchid`) onto the Zynq at `/root/orchid/` and then program the Zynq fabric.

## Building the C application
Open an SSH connection to Microzed using Git Bash (or puTTY) and do the following:
```bash
cd /root/git_repos/zynq_dev
cd apps
# for lotus and delorean, also do 'cd lotus' or 'cd delorean'
make
```
This will make the `orchid_app.elf` file and copy it to `/root/orchid/`.

# Testing

In general, the following instructions apply equally to 'delorean', 'lotus' and 'orchid' designs.

## RTL behavioral simulations
(Structural simulations, with or without back-annotated timing, are not currently supported).

Running RTL simulations requires that the FPGA bitstream has been made, as described above. On your computer, run the following in a Git Bash shell to execute the simulations using Vivado:
```bash
. python_tools/setup.py  # only need to do this once after opening the shell
cd unittests/
nosetests test_orchid.py --tc=build_clean:1
```

Building is not done by default and only needs to be done once so subsequent simulations can be expedited by running
```bash
nosetests test_orchid.py
```

You can discover the list of available test by running
```bash
nosetests --collect-only --verbose --nocapture test_orchid.py
```

You can target a particular test or set of tests with the `-m` option, which overrides the regular expression that nosetests uses to identify test functions. For example, run
```bash
nosetests -m axi_loopback test_orchid.py
```

After a simulation completes, you can view the waveforms by running
```bash
cd orchid/
cd xsim -gui -view ../../sim_src/tb_orchid_behav.wcfg tb_orchid
```

## Python unit tests
Running python unit tests requires that 1) the FPGA bitstream has been made and uploaded to the Microzed and that 2) the Orchid C application has been built. In other words, the `/root/orchid` directory on Microzed should contain `orchid.bit` and `orchid_app.elf`.

On your computer, run the following in a Git Bash shell to execute the tests using "nosetests":
```bash
. python_tools/setup.py  # only need to do this once after opening the shell
nosetests -w python_tools/unittests --tc=host:<hostname>
```
You should see a '.' character displayed for each passing test and, possibly, an 'S' character for each skipped test.

# Usage

## Running the Orchid interactive prompt
Running the interactive prompt requires that 1) the FPGA bitstream has been made and uploaded to the Microzed and that 2) the Orchid C application has been built. In other words, the `/root/orchid` directory on Microzed should contain `orchid.bit` and `orchid_app.elf`.

On your computer, run the following in a Git Bash shell to invoke the interactive prompt:
```bash
. python_tools/setup.py  # only need to do this once after opening the shell
python python_tools/run_orchid.py --host=<hostname>
```

# Troubleshooting
1. What is the value of your PYTHONPATH variable? Is zynq_dev at the beginning of the list?
2. Did you install the paramiko python package? `conda install -c anaconda paramiko`
3. Did you install the scp python package? `pip install scp`
4. Is your microzed device on the network? Ping it.
5. Is the remote application already running and/or hung? Make an SSH connection and check with `ps aux | grep orchid_app.elf`. Kill the application with `killall orchid_app.elf`.

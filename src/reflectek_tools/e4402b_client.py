"""Measurment automation client for E4402B spectrum analyzer.

"""

__docformat__ = "numpy"

import time
import numpy as np
import pyvisa

class E4402BClient:
    """Drives the NSI2000 near-field scanner via COM automation.

    Parameters
    ----------
    visible : bool
        Whether to show the NSI2000 application GUI (default True).
    """

    def __init__(self, visible: bool = True):
        self.visible = visible
        self.server = None
        self.app = None
        self.cmd = None

    def connect(self):
        """Attach to (or launch) the NSI2000 COM server."""
        # try:
        #     import win32com.client
        # except ImportError:
        #     raise ImportError(
        #         "win32com (pywin32) is required for NSI2000Client. "
        #         "This is only available on Windows. "
        #         "Install with: pip install raopt[measurement]"
        #     ) from None

        # self.server = win32com.client.Dispatch("NSI2000.server")
        # self.app = self.server.AppConnection
        # self.app.Visible = self.visible
        # self.cmd = self.app.ScriptCommands
        self.rm = pyvisa.ResourceManager()
        self.sa = self.rm.open_resource("GPIB0::18::INSTR")
        self.sa.write("*RST")              # reset + clear
        self.sa.write("*CLS")
        self.sa.write(":FREQ:CENT 1.5e9")         # 1 GHz center
        self.sa.write(":FREQ:SPAN 2e9")       # 100 MHz span
        self.sa.write(":BAND:RES 3e6")        # 100 kHz RBW
        self.sa.write(":AVER ON")
        self.sa.write(":AVER:COUN 10")
        
        self.sa.timeout = 10000
        return self

    def disconnect(self):
        """Release COM references."""
        # import gc

        # self.cmd = None
        # self.app = None
        # self.server = None
        # gc.collect()
        self.sa.close()
        self.rm.close()

    def run_scan_get_hor_amp(self, filename, beam):
        """Run a scan and return the amplitude at all near-field points.

        Parameters
        ----------
        filename : str
            NSI2000 scan configuration file path.
        beam : int
            Beam index (frequency selection).

        Returns
        -------
        ndarray, shape (nf_vpts, nf_hpts)
            Amplitude values at each near-field grid point.
        """
        # start_time = time.time()
        # self.cmd.MEAS_CREATE_NEW_SCAN()
        # self.cmd.MEAS_ACQUIRE(filename, True)

        # nf_hpts = int(self.cmd.NF_HPTS)
        # nf_vpts = int(self.cmd.NF_VPTS)
        # amp = np.zeros((nf_vpts, nf_hpts))
        # self.cmd.SELECT_BEAM(beam)
        # for i in range(nf_vpts):
        #     for j in range(nf_hpts):
        #         amp[i, j] = self.cmd.NFPOL1_AMP(j, i)[0]

        # elapsed = time.time() - start_time
        # print(f"Acquisition completed in {elapsed:.1f} seconds")
        #self.sa.write(":AVER:CLE")
        self.sa.write(":INIT:CONT OFF")         # single-sweep mode
        self.sa.write(":INIT:IMM")              # trigger sweep
        self.sa.query("*OPC?")                  # wait until done

        data = self.sa.query(":TRAC:DATA? TRACE1")
        self.sa.control_ren(pyvisa.constants.RENLineOperation.deassert)
        points = np.array([float(x) for x in data.split(",")])
        max_pwr = np.max(points)
        print(max_pwr)
        return max_pwr

    def save_scan(self, k, is_loss_plus, cal_folder):
        """Save scan data to an ASCII listing file.

        Parameters
        ----------
        k : int
            Iteration index for filename.
        is_loss_plus : bool
            True for loss-plus calibration, False for loss-minus.
        cal_folder : Path
            Directory to save the listing file.
        """
        if is_loss_plus:
            cal_file = cal_folder / f"cal_iter_{k}_Lp.asc"
        else:
            cal_file = cal_folder / f"cal_iter_{k}_Lm.asc"
        self.cmd.NF_LISTING_TO_FILE(cal_file)

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()
        return False

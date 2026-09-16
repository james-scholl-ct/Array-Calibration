# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:55:13 2021

@author: rdpatton
"""

import clr # provided by pythonnet, .NET interface layer
import sys
import time

# NB the 
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.KCube.PositionAlignerCLI")
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("System")

from Thorlabs.MotionControl.KCube.PositionAlignerCLI import KCubePositionAligner, PositionAlignerStatus, XYPosition
from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from System import Decimal, Double


MONITOR = PositionAlignerStatus.OperatingModes.Monitor
OPENLOOP = PositionAlignerStatus.OperatingModes.OpenLoop
CLOSEDLOOP = PositionAlignerStatus.OperatingModes.ClosedLoop


def list_devices():
    """Return a list of Kinesis serial numbers"""
    DeviceManagerCLI.BuildDeviceList()
    return DeviceManagerCLI.GetDeviceList()


def status_handler(source, args):
    print(f'{args.Status.PositionDifference.X:.4f},'
          f'{args.Status.PositionDifference.Y:.4f},'
          f'{args.Status.Sum:.4f}')
    #return [args.Status.PositionDifference.X,
     #                                   args.Status.PositionDifference.Y,
      #                                  args.Status.Sum]


class PositionAlignerWrapper():
    def __init__(self, serial_number=None):
        if serial_number is None:
            devices = list_devices()
            for device in devices:
                if device[:2] == '69':
                    serial_number = device
                    break
        if serial_number is None:
            raise Exception('Position Aligner device not found.')
        self._ser = str(serial_number)
        DeviceManagerCLI.BuildDeviceList()
        self._aligner = KCubePositionAligner.CreateKCubePositionAligner(self._ser)
        self._aligner.data_buffer = []
        self.connected = False


    def connect(self):
        """Initialise communications, populate channel list, etc."""
        assert not self.connected
        self._aligner.Connect(self._ser)
        self.connected = True
        self._aligner.WaitForSettingsInitialized(5000)
        self._aligner.StartPolling(250) # getting the voltage only works if you poll!
        time.sleep(0.5) # ThorLabs have this in their example...
        self._aligner.EnableDevice()
        # I don't know if the lines below are necessary or not - but removing them
        # may or may not work...
        time.sleep(0.5)
        config = self._aligner.GetPositionAlignerConfiguration(self._aligner.DeviceID)
        info = self._aligner.GetDeviceInfo()


    def close(self):
        """Shut down communications"""
        if not self.connected:
            print(f"Not closing piezo device {self._ser}, it's not open!")
            return
        self._aligner.data_buffer = []
        self.stop_polling()
        self._aligner.Disconnect(True)
        self.connected = False


    def __del__(self):
        try:
            if self.connected:
                self.close()
        except:
            print(f"Error closing communications on deletion of device {self._ser}")


    def get_position(self):
        """Retrieve the output voltages as a list of floating-point numbers"""
        status = self._aligner.Status
        return [status.PositionDifference.X,
                status.PositionDifference.Y,
                status.Sum]
    
    
    def stop_polling(self):
        self._aligner.StopPolling()
        
        
    def start_polling(self, rate_ms):
        self._aligner.StartPolling(int(rate_ms))
    
    
    def status(self):
        return self._aligner.Status

    
    def set_mode(self, mode):
        """Set the device to Monitor mode."""
        self._aligner.SetOperatingMode(mode, False)


    def get_mode(self):
        """Get the device operating mode."""
        return self._aligner.GetOperatingMode()
        
        
    def set_closed_loop_position(self, position):
        xyposition = XYPosition(Double(position[0]), Double(position[1]))
        self._aligner.SetClosedLoopPosition(xyposition)
        
        
    def get_closed_loop_position(self):
        return self._aligner.GetClosedLoopPosition()
        
        
    def get_digital_output(self):
        return self._aligner.GetDigitalOutput()
        
        
    def get_pos_demand_parameters(self):
        return self._aligner.GetPosDemandParams()
        
        
    def get_demanded_position(self):
        return self._aligner.GetDemandedPosition()
        
        
    def set_position(self, position):
        xyposition = XYPosition(Double(position[0]), Double(position[1]))
        self._aligner.SetPosition(xyposition)
        
        
    def zero_setpoint(self):
        """Zero the setpoint."""
        self._aligner.SetPosition(XYPosition(Double(0), Double(0)))
        
    
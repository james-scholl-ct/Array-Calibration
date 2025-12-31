# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 10:44:58 2025

@author: SchollJamesAC3CARILL
"""
import paramiko
from typing import Optional

class PiController:
    """
    Connects to a raspberry pi via ssh with Paramiko. Copies a local low and high band file to the files on the PI. 
    Runs the remote command that starts the python program on the PI which updates the DACs via SPI.
    Creates a stop text file that is watched for by that program to stop it so that it can read another set of HB and LB voltages
    """
    def __init__(
        self,
        host: str,
        username: str,
        password: Optional[str],
        local_file_hb: str,
        local_file_lb: str,
        remote_file_hb: str,
        remote_file_lb: str,
        remote_command: str,
        port: int,
        key_filename: Optional[str],
        stop_file: str,
    ):
        self.host = host
        self.username = username
        self.password = password
        self.local_file_hb = local_file_hb
        self.local_file_lb = local_file_lb
        self.remote_file_hb = remote_file_hb
        self.remote_file_lb = remote_file_lb
        self.remote_command = remote_command
        self.port = port
        self.key_filename = key_filename
        self.stop_file = stop_file
        self.client = None
        
    def connect(self):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        client.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            key_filename=self.key_filename,
            look_for_keys=True,
        )
        self.client = client
        return self
    
    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None
                
    def remove_stop_file(self):
        sftp = self.client.open_sftp()
        try:
            sftp.remove(self.stop_file)
            print("Stop File removed")
        except FileNotFoundError:
            pass
        sftp.close()
        
    def stop_program(self):
        sftp = self.client.open_sftp()
        with sftp.open(self.stop_file, "w"):
            pass
        sftp.close()
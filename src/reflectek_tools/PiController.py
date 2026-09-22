# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 10:44:58 2025

@author: SchollJamesAC3CARILL
"""
import paramiko
from typing import Optional
import time

class PiController:
    """
    Connects to a raspberry pi via ssh with Paramiko. Copies a local low and high band file to the files on the PI. 
    Runs the remote command that starts the python program on the PI which updates the DACs via SPI.
    Creates a stop text file that is watched for by that program to stop it so that it can read another set of HB and LB voltages
    #Reopens VNA connection each time, closes resource manager at the very end
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
    def _transport_ok(self):
        try:
            t = self.client.get_transport() if self.client else None
            return (t is not None) and t.is_active()
        except Exception:
            return False
    def ensure_connected(self):
        if not self._transport_ok():
            self.close()
            self.connect()
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
        t = self.client.get_transport()
        if t:
            t.set_keepalive(30)
            
        return self
    
    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None
                
    def remove_stop_file(self):
        self.client.exec_command(f"rm -f {self.stop_file}")
        
    def stop_program(self):
        self.client.exec_command(f"touch {self.stop_file}")
        
    def upload_lb_and_hb_files(self):
        sftp = self.client.open_sftp()
        print(f"Uploading {self.local_file_hb} -> {self.remote_file_hb} ...")
        sftp.put(self.local_file_hb, self.remote_file_hb)
        print("Upload complete")
        
        print(f"Uploading {self.local_file_lb} -> {self.remote_file_lb} ...")
        sftp.put(self.local_file_lb, self.remote_file_lb)
        print("Upload complete")
        sftp.close()
        
    def run_remote_command(self, wait: bool = False, get_pty: bool =False):
        print(f"Running remote command: {self.remote_command}")
        stdin, stdout, stderr = self.client.exec_command(self.remote_command, get_pty=get_pty)
        if not wait:
            return None
        exit_status = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return out, err, exit_status
    def update_dacs(self):
        for attempt in range(2):
            try: 
                self.ensure_connected()
                self.stop_program()
                time.sleep(1)
                self.remove_stop_file()
                self.upload_lb_and_hb_files()
                self.run_remote_command()
                return
            except (paramiko.SSHException, ConnectionResetError, OSError) as e:
                if attempt == 0:
                    self.close()
                    time.sleep(.5)
                    continue
                raise
        
    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, tb):
        self.close()

## General Python
import numpy as np
import os

## Data Loaders
import xarray as xr

## Astropy Tools
from astropy.time import Time
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import EarthLocation

## Plotting
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from tqdm import tqdm


class DDMI:

    def __init__(self,fname,source='CYGNSS'):
        if source == 'CYGNSS':
            data = xr.open_dataset(fname)

            self.startTime = Time(data.time_coverage_start)
            self.brcs = np.array(data.brcs)
            self.time = Time(np.array(data.ddm_timestamp_utc))
            self.gpsCode = np.array(data.prn_code).astype(int)
            self.cygNumber = int(data.spacecraft_num)

            ##Get Errors
            self.errorCodes = np.array(data.quality_flags)
            self.isError = self.errorCodes > 0

            ##Get Positions and Velocities
            sourceX = np.array(data.tx_pos_x)
            sourceY = np.array(data.tx_pos_y)
            sourceZ = np.array(data.tx_pos_z)
            sourceVX = np.array(data.tx_vel_x)
            sourceVY = np.array(data.tx_vel_y)
            sourceVZ = np.array(data.tx_vel_z)
            specularX = np.array(data.sp_pos_x)
            specularY = np.array(data.sp_pos_y)
            specularZ = np.array(data.sp_pos_z)
            specularVX = np.array(data.sp_vel_x)
            specularVY = np.array(data.sp_vel_y)
            specularVZ = np.array(data.sp_vel_z)
            observerX = np.array(data.sc_pos_x)
            observerY = np.array(data.sc_pos_y)
            observerZ = np.array(data.sc_pos_z)
            observerVX = np.array(data.sc_vel_x)
            observerVY = np.array(data.sc_vel_y)
            observerVZ = np.array(data.sc_vel_z)
            

            self.sourcePos = np.transpose(np.array([sourceX, sourceY, sourceZ]), (1, 2, 0))
            self.specularPos = np.transpose(np.array([specularX, specularY, specularZ]), (1, 2, 0))
            self.observerPos = np.array([observerX, observerY, observerZ]).T

            self.sourceVel = np.transpose(np.array([sourceVX, sourceVY, sourceVZ]), (1, 2, 0))
            self.specularVel = np.transpose(np.array([specularVX, specularVY, specularVZ]), (1, 2, 0))
            self.observerVel = np.array([observerVX, observerVY, observerVZ]).T

            ## Delay and Doppler
            self.nDoppler = data.dims['doppler']
            self.nDelay = data.dims['delay']
            self.nDDM = data.dims['ddm']
            self.chipSize = ((1./1023000.)*u.s).to(u.us)
            self.delay = np.linspace(-(self.nDelay-1)//2,(self.nDelay-1)//2,self.nDelay)*float(data.delay_resolution)*self.chipSize
            self.doppler = np.linspace(-(self.nDoppler-1)//2,(self.nDoppler-1)//2,self.nDoppler)*float(data.dopp_resolution)*u.Hz
            self.tau0 = np.array(data.brcs_ddm_sp_bin_delay_row)*float(data.delay_resolution)*self.chipSize+self.delay[0]
            self.fd0 = np.array(data.brcs_ddm_sp_bin_dopp_col)*float(data.dopp_resolution)*u.Hz+self.doppler[0]

            year = str(self.startTime.datetime.year)
            month = self.startTime.datetime.month
            day = self.startTime.datetime.day
            if month<10:
                month = '0'+str(month)
            else:
                month=str(month)
            if day<10:
                day = '0'+str(day)
            else:
                day=str(day)  
            self.date=year+month+day

    def repack_data(self,dir='.'):
        for nSource in np.array([2, 3, 4]):
            hasCount = np.argwhere(np.sum(self.isError, 1) == (self.nDDM - nSource))[:, 0]

            tempIDs = self.errorCodes[hasCount] == 0
            tempGPS = np.reshape(self.gpsCode[hasCount][tempIDs], (-1, nSource))
            tempTime = self.time[hasCount]
            tempSourcePos = np.reshape(self.sourcePos[hasCount][tempIDs], (-1, nSource, 3))
            tempSpecularPos = np.reshape(self.specularPos[hasCount][tempIDs], (-1, nSource, 3))
            tempObserverPos = self.observerPos[hasCount]
            tempSourceVel = np.reshape(self.sourceVel[hasCount][tempIDs], (-1, nSource, 3))
            tempSpecularVel = np.reshape(self.specularVel[hasCount][tempIDs], (-1, nSource, 3))
            tempObserverVel = self.observerVel[hasCount]
            tempDDMs = np.reshape(self.brcs[hasCount][tempIDs], (-1, nSource, self.nDelay, self.nDoppler))
            tempTau0 = np.reshape(self.tau0[hasCount][tempIDs],(-1, nSource))
            tempFd0 = np.reshape(self.fd0[hasCount][tempIDs],(-1, nSource))

            fname = f"{self.date}-cyg0{self.cygNumber}-{nSource}source.npz"
            np.savez(
                os.path.join(dir,fname),
                time=tempTime.mjd,
                ddm=tempDDMs,
                delay=self.delay,
                doppler=self.doppler,
                tau0=tempTau0,
                fd0=tempFd0,
                specularPos=tempSpecularPos,
                observerPos=tempObserverPos,
                sourcePos=tempSourcePos,
                specularVel=tempSpecularVel,
                observerVel=tempObserverVel,
                sourceVel=tempSourceVel,
                gpsCode=tempGPS,
                sampleNumber=hasCount,
            )
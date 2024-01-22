
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

import arcqua.fitting as af


class DDMI:

    def __init__(self,fname,source='CYGNSS'):
        if source == 'CYGNSS':
            self.freq = 1.57542 * u.GHz
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
            

            self.sourcePos = np.transpose(np.array([sourceX, sourceY, sourceZ]), (1, 2, 0))*u.m
            self.specularPos = np.transpose(np.array([specularX, specularY, specularZ]), (1, 2, 0))*u.m
            self.observerPos = np.array([observerX, observerY, observerZ]).T*u.m

            self.sourceVel = np.transpose(np.array([sourceVX, sourceVY, sourceVZ]), (1, 2, 0))*u.m/u.s
            self.specularVel = np.transpose(np.array([specularVX, specularVY, specularVZ]), (1, 2, 0))*u.m/u.s
            self.observerVel = np.array([observerVX, observerVY, observerVZ]).T*u.m/u.s

            ## Delay and Doppler
            self.nDoppler = data.dims['doppler']
            self.nDelay = data.dims['delay']
            self.nDDM = data.dims['ddm']
            self.chipSize = ((1./1023000.)*u.s).to(u.us)
            self.delay = np.linspace(-(self.nDelay-1)//2,(self.nDelay-1)//2,self.nDelay)*float(data.delay_resolution)*self.chipSize
            self.doppler = np.linspace(-(self.nDoppler-1)//2,(self.nDoppler-1)//2,self.nDoppler)*float(data.dopp_resolution)*u.Hz
            self.tau0 = np.array(data.brcs_ddm_sp_bin_delay_row)*float(data.delay_resolution)*self.chipSize+self.delay[0]
            self.fd0 = np.array(data.brcs_ddm_sp_bin_dopp_col)*float(data.dopp_resolution)*u.Hz+self.doppler[0]

            timeString = self.startTime.value
            year=timeString.split('-')[0]
            month=timeString.split('-')[1]
            day=timeString.split('-')[2].split('T')[0]
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
    
    def read_fits(self,dir='.'):
        self.fits={}
        for nSource in np.array([2,3,4]):
            try:
                arch=np.load(os.path.join(dir,f'{self.date}-cyg0{self.cygNumber}-{nSource}source_fits_full.npz'))
                self.fits.update({nSource : {}})
                powers=arch['powers']
                asymm=arch['asymm']
                thetas=arch['thetas']*u.deg
                fitPars=arch['fitPars']
                fitRes=arch['fitRes']
                sampleNumber=arch['sampleNumber']
                self.fits[nSource].update({'powers' : powers})
                self.fits[nSource].update({'thetas' : thetas})
                self.fits[nSource].update({'asymm' : asymm})
                self.fits[nSource].update({'fitPars' : fitPars})
                self.fits[nSource].update({'fitRes' : fitRes})
                self.fits[nSource].update({'sampleNumber' : sampleNumber})
                self.fits[nSource].update({'best' : fitRes[:,:,1]==fitRes[:,:,1].min(1)[:,np.newaxis]})
            except:
                print(f'{nSource} fit does not exist for cyg{self.cygNumber} on {self.date}')
    
    def read_archive(self,dir='.'):
        self.archival = xr.load_dataset(os.path.join(dir,f'{self.date}.nc'))

    def _build_results(self,results,pars):
        names=['left','right']
        best=names[np.argmin(results[:,2])]
        worst=names[1-np.argmin(results[:,2])]
        fitRes = {}
        fitRes.update({'left':{}})
        fitRes.update({'right':{}})
        fitRes.update({'best':best})
        fitRes.update({'worst':worst})
        for i in range(2):
            fitRes[names[i]].update({'mean': results[i,0]})
            fitRes[names[i]].update({'err': results[i,1]})
            fitRes[names[i]].update({'chi': results[i,2]})
            fitRes[names[i]].update({'std': results[i,3]})
            fitRes[names[i]].update({'fits': pars[i]})
        return(fitRes)
    
    def plot_sample(self,nSample,nSource,fname=None):
        nSampleFull = self.fits[nSource]['sampleNumber'][nSample]
        goodDDM = np.invert(self.isError[nSampleFull])
        fitRes = self._build_results(self.fits[nSource]['fitRes'][nSample],self.fits[nSource]['fitPars'][nSample])
        af.single_plot(self.brcs[nSampleFull,goodDDM],
                self.sourcePos[nSampleFull,goodDDM],
                self.specularPos[nSampleFull,goodDDM],
                self.observerPos[nSampleFull],
                self.sourceVel[nSampleFull,goodDDM],
                self.observerVel[nSampleFull],
                self.delay,self.doppler,
                self.tau0[nSampleFull,goodDDM],self.fd0[nSampleFull,goodDDM],
                self.freq,
                fitRes,
                self.fits[nSource]['thetas'][nSample],
                self.fits[nSource]['powers'][nSample],
                fname=fname)
        
    def _calc_UV_thetas(self, nSource, nSample):
        nSampleFull = self.fits[nSource]["sampleNumber"][nSample]
        goodDDM = np.invert(self.isError[nSampleFull])
        fitRes = self._build_results(
            self.fits[nSource]["fitRes"][nSample], self.fits[nSource]["fitPars"][nSample]
        )
        speculars = self.specularPos[nSampleFull, goodDDM]
        emitters = self.sourcePos[nSampleFull, goodDDM]
        receiver = self.observerPos[nSampleFull]
        eLoc = EarthLocation(x=speculars[:, 0], y=speculars[:, 1], z=speculars[:, 2])
        lon = eLoc.lon
        lat = eLoc.lat
        time = self.time[nSampleFull]
        idLon = np.argmin(
            np.abs(np.array(self.archival.longitude) * u.deg - lon[:, np.newaxis]), 1
        )
        idLat = np.argmin(
            np.abs(np.array(self.archival.latitude) * u.deg - lat[:, np.newaxis]), 1
        )
        idTime = np.argmin(np.abs(Time(np.array(self.archival.time)) - time))

        ddmUV = (
            np.array(
                [
                    af.toUV(
                        emitters,
                        speculars,
                        receiver,
                        fitRes[fitRes["best"]]["mean"] * u.deg,
                        i,
                    )
                    for i in range(nSource)
                ]
            )
            * u.m
            / u.s
        )
        ddmTheta = np.mod(
            np.arctan2(ddmUV[:, 0], ddmUV[:, 1]).to(u.deg) - 180 * u.deg, 360 * u.deg
        )
        archiveUV = (
            np.array(
                [
                    self.archival.u10[idTime, idLat, idLon],
                    self.archival.v10[idTime, idLat, idLon],
                ]
            )
            * u.m
            / u.s
        )
        archiveTheta = np.diag(
            np.mod(np.arctan2(*archiveUV).to(u.deg) - 180 * u.deg, 360 * u.deg)
        )
        archiveSpeed = np.diag(np.sqrt(np.sum(archiveUV**2,0)))

        mwd = np.diag(self.archival.mwd[idTime, idLat, idLon].values) * u.deg
        mdww = np.diag(self.archival.mdww[idTime, idLat, idLon].values * u.deg)
        mdts = np.diag(self.archival.mdts[idTime, idLat, idLon].values * u.deg)
        return (ddmTheta, archiveTheta, mwd, mdww, mdts,archiveSpeed)
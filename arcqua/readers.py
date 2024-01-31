
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

    def __init__(self,
                 date,
                 fitsDir='.',
                 ddmiDir='.',
                 archiveDir = '.',
                 instrument='cyg',
                 nSats=8,
                 nSources=np.array([2,3,4]).astype(int)):
        self.date=date
        self.instrument=instrument
        self.fitsDir=fitsDir
        self.ddmiDir=ddmiDir
        self.archiveDir = archiveDir
        self.nSats=nSats
        self.nSources=nSources
        self.freq = 1.57542 * u.GHz
        self._read_ddms()
        self._read_fits()
        self._read_archive()
    
    def _read_ddms(self):
        self.ddms={}
        for satNum in range(self.nSats):
            fnames = [os.path.join(self.ddmiDir,f'{self.date}-{self.instrument}{str(satNum+1).zfill(2)}-{nSource}source.npz') for nSource in self.nSources]
            exists = np.array([os.path.exists(fname) for fname in fnames])
            if np.any(exists):
                self.ddms.update({satNum+1 : {}})
                for id in range(self.nSources.shape[0]):
                    if exists[id]:
                        arch=np.load(fnames[id])
                        self.ddms[satNum+1].update({self.nSources[id]: {
                            "time" : Time(arch['time'],format='mjd'),
                            "ddm" : arch['ddm'],
                            "delay" : arch['delay']*u.us,
                            "doppler" : arch['doppler']*u.Hz,
                            "tau0" : arch['tau0']*u.us,
                            "fd0" : arch['fd0']*u.Hz,
                            "specularPos" : arch['specularPos']*u.m,
                            "observerPos" : arch['observerPos']*u.m,
                            "sourcePos" : arch['sourcePos']*u.m,
                            "specularVel" : arch['specularVel']*u.m/u.s,
                            "observerVel" : arch['observerVel']*u.m/u.s,
                            "sourceVel" : arch['sourceVel']*u.m/u.s,
                            "gpsCode" : arch['gpsCode'],
                            "sampleNumber" : arch['sampleNumber']
                        }})



    def _read_fits(self):
        self.fits={}
        for satNum in range(self.nSats):
            self.fits.update({satNum+1 : {}})
            for nSource in self.nSources:
                try:
                    arch=np.load(os.path.join(self.fitsDir,f'{self.date}-{self.instrument}{str(satNum+1).zfill(2)}-{nSource}source_fits_full.npz'))
                    self.fits[satNum+1].update({nSource : {}})
                    powers=arch['powers']
                    asymm=arch['asymm']
                    thetas=arch['thetas']*u.deg
                    fitPars=arch['fitPars']
                    fitRes=arch['fitRes']
                    sampleNumber=arch['sampleNumber']
                    self.fits[satNum+1][nSource].update({'powers' : powers})
                    self.fits[satNum+1][nSource].update({'thetas' : thetas})
                    self.fits[satNum+1][nSource].update({'asymm' : asymm})
                    self.fits[satNum+1][nSource].update({'fitPars' : fitPars})
                    self.fits[satNum+1][nSource].update({'fitRes' : fitRes})
                    self.fits[satNum+1][nSource].update({'sampleNumber' : sampleNumber})
                    self.fits[satNum+1][nSource].update({'best' : fitRes[:,:,1]==fitRes[:,:,1].min(1)[:,np.newaxis]})
                except:
                    print(f'{nSource} fit does not exist for cyg{self.instrument}{str(satNum+1).zfill(2)} on {self.date}')
    
    def _read_archive(self):
        self.archival = xr.load_dataset(os.path.join(self.archiveDir,f'{self.date}.nc'))

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
    
    def plot_sample(self,satNum,nSample,nSource,fname=None):
        fitRes = self._build_results(self.fits[satNum][nSource]['fitRes'][nSample],self.fits[satNum][nSource]['fitPars'][nSample])
        af.single_plot(self.ddms[satNum][nSource]['ddm'][nSample],
                self.ddms[satNum][nSource]['sourcePos'][nSample],
                self.ddms[satNum][nSource]['specularPos'][nSample],
                self.ddms[satNum][nSource]['observerPos'][nSample],
                self.ddms[satNum][nSource]['sourceVel'][nSample],
                self.ddms[satNum][nSource]['observerVel'][nSample],
                self.ddms[satNum][nSource]['delay'],
                self.ddms[satNum][nSource]['doppler'],
                self.ddms[satNum][nSource]['tau0'][nSample],
                self.ddms[satNum][nSource]['fd0'][nSample],
                self.freq,
                fitRes,
                self.fits[satNum][nSource]['thetas'][nSample],
                self.fits[satNum][nSource]['powers'][nSample],
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
    

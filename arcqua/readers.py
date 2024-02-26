
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

from tqdm.notebook import tqdm

import arcqua.fitting as af


class DDMI:

    def __init__(self,
                 date,
                 fitsDir='.',
                 ddmiDir='.',
                 archiveDir = '.',
                 uvDir = '.',
                 mapDir = '.',
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
        self.uvDir=uvDir
        self.mapDir = mapDir
        print('Load DDMs')
        self._read_ddms()
        print('Load Fits')
        self._read_fits()
        print('Load Archival')
        self._read_archive()
        try:
            self._load_UV()
        except:
            print('Calculate UVs')
            self._calc_UV()
            self._save_UV()
        try:
            self._load_uv_map()
        except:
            print('Calculate UV Map')
            self._calc_uv_map()
            self._save_uv_map()
    
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
    
    def plot_sample(self,satNum,nSample,nSource,fname=None,archival=False):
        fitRes = self._build_results(self.fits[satNum][nSource]['fitRes'][nSample],self.fits[satNum][nSource]['fitPars'][nSample])
        if archival:
            time = self.ddms[satNum][nSource]['time'][nSample]
            archivalID = np.argmin(np.abs(self.ddms[1][4]['time'][0]-Time(np.array(self.archival.time))))
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
                    fname=fname,archival=self.archival,archivalID=archivalID)
        else:
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

    def _save_UV(self):
          for satID in tqdm(range(self.nSats),position=0):
            satNum=satID+1
            for nSource in tqdm(self.nSources,position=1,leave=False):
                np.savez(os.path.join(self.uvDir,
                                      f'{self.date}-{self.instrument}{str(satNum+1).zfill(2)}-{nSource}source_uv.npz'),
                        uv=self.recoverUV[satNum][nSource])
   
    def _load_UV(self):
        self.recoverUV = {}
        for satID in tqdm(range(self.nSats),position=0):
            satNum=satID+1
            self.recoverUV.update({satNum : {}})
            for nSource in tqdm(self.nSources,position=1,leave=False):
                arch=np.load(os.path.join(self.uvDir,
                                      f'{self.date}-{self.instrument}{str(satNum+1).zfill(2)}-{nSource}source_uv.npz'))
                self.recoverUV[satNum].update({nSource : arch['uv']})
    
    def _calc_UV(self):
        self.recoverUV = {}
        for satID in tqdm(range(self.nSats),position=0):
            satNum=satID+1
            self.recoverUV.update({satNum : {}})
            for nSource in tqdm(self.nSources,position=1,leave=False):
                uv = np.zeros((self.fits[satNum][nSource]["fitRes"].shape[0],nSource,2))
                for nSample in tqdm(range(uv.shape[0]),position=2,leave=False):
                    fitRes = self._build_results(
                        self.fits[satNum][nSource]["fitRes"][nSample],
                        self.fits[satNum][nSource]["fitPars"][nSample],
                    )
                    theta = fitRes[fitRes["best"]]["mean"] * u.deg

                    ## Asymmetry Stuff
                    dirs = (self.ddms[satNum][nSource]["specularPos"][nSample]- self.ddms[satNum][nSource]["observerPos"][nSample][np.newaxis, :])
                    vel = self.ddms[satNum][nSource]["observerVel"][nSample]
                    dirSigns = (dirs * vel).sum(1)
                    thetaID = np.argmin(np.abs(self.fits[satNum][nSource]["thetas"][nSample] - theta))
                    asymm=self.fits[satNum][nSource]["asymm"][nSample, :, thetaID]*dirSigns

                    xAxis, yAxis, zAxis, psis, des, drs = af.coords(
                        self.ddms[satNum][nSource]["sourcePos"][nSample],
                        self.ddms[satNum][nSource]["specularPos"][nSample],
                        self.ddms[satNum][nSource]["observerPos"][nSample],
                    )
                    thetas = af.thetaConvert(theta, xAxis, yAxis, zAxis)
                    thetaIs = af.wavesToImages(thetas, psis)
                    emitterVels2 = af.convertToSpecular(
                        self.ddms[satNum][nSource]["sourceVel"][nSample], xAxis, yAxis, zAxis
                    )
                    receiverVel2 = af.convertToSpecular(
                        self.ddms[satNum][nSource]["observerVel"][nSample], xAxis, yAxis, zAxis
                    )

                    sHats = np.array(
                        [np.cos(thetaIs).value, np.sin(thetaIs).value, np.zeros(nSource)]
                    ).T
                    rHats = np.array(
                        [np.cos(psis).value, np.zeros(nSource), np.sin(psis).value]
                    ).T
                    eHats = np.array(
                        [-np.cos(psis).value, np.zeros(nSource), np.sin(psis).value]
                    ).T
                    cosProd = np.cos(psis) * np.cos(thetaIs)
                    vels = np.sum(
                            receiverVel2 * (sHats - rHats * cosProd[:, np.newaxis])
                            + emitterVels2
                            * (sHats + eHats * cosProd[:, np.newaxis])
                            * (drs / des)[:, np.newaxis],
                            1,
                        ).value
                    
                    dirVel = np.sign(np.mean(vels*asymm))

                    for sourceNum in range(nSource):
                        uv[nSample,sourceNum,:] = af.toUV(
                            self.ddms[satNum][nSource]["sourcePos"][nSample],
                            self.ddms[satNum][nSource]["specularPos"][nSample],
                            self.ddms[satNum][nSource]["observerPos"][nSample],
                            theta,
                            id=sourceNum,
                        )*dirVel
                self.recoverUV[satNum].update({nSource : uv})

    def _calc_uv_map(self):
        lats = np.array(self.archival.latitude) * u.deg
        lons = np.linspace(0,360,self.archival.longitude.shape[0]+1)*u.deg
        us = np.zeros((24,lats.shape[0],lons.shape[0]))
        vs = np.copy(us)
        counts = np.copy(us)

        for satID in tqdm(range(8), position=0):
            satNum = satID + 1
            for nSource in [2, 3, 4]:
                for nSample in tqdm(
                    range(self.fits[satNum][nSource]["fitRes"].shape[0]),
                    position=1,
                    leave=False,
                ):
                    for i in range(nSource):
                        pos = self.ddms[satNum][nSource]["specularPos"][nSample][i]
                        loc = EarthLocation(x=pos[0], y=pos[1], z=pos[2])
                        lat = loc.lat
                        lon = loc.lon
                        time = self.ddms[satNum][nSource]["time"][nSample]
                        timeID = np.argmin(np.abs(time - Time(np.array(self.archival.time))))
                        lonID = np.argmin(np.abs(np.mod(lon, 360 * u.deg) - lons))
                        latID = np.argmin(np.abs(lat - lats))
                        us[timeID, latID, lonID] += self.recoverUV[satNum][nSource][nSample,i,0]
                        vs[timeID, latID, lonID] += self.recoverUV[satNum][nSource][nSample,i,1]
                        counts[timeID, latID, lonID] += 1
        us[:,:,0]+=us[:,:,-1]
        vs[:,:,0]+=vs[:,:,-1]
        counts[:,:,0]+=counts[:,:,-1]
        us=us[:,:,:-1]
        vs=vs[:,:,:-1]
        counts=counts[:,:,:-1]
        lons=lons[:-1]
        self.uvMap = {"u" : us/counts,
                      "v" : vs/counts,
                      "counts" : counts,
                      "lats" : lats,
                      "lons" : lons}
    
    def _save_uv_map(self):
        np.savez(os.path.join(self.mapDir,f'{self.date}-{self.instrument}-map.npz'),
                 u=self.uvMap['u'],
                 v=self.uvMap['v'],
                 counts=self.uvMap['counts'],
                 lats=self.uvMap['lats'],
                 lons=self.uvMap['lons'])

    def _load_uv_map(self):
        arch=np.load(os.path.join(self.mapDir,f'{self.date}-{self.instrument}-map.npz'))
        self.uvMap = {"u" : arch['u'],
                      "v" : arch['v'],
                      "counts" : arch['counts'],
                      "lats" : arch["lats"],
                      "lons" : arch["lons"]}
    
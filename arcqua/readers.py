
## General Python
import numpy as np
import os
from scipy.optimize import curve_fit

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

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except:
    from tqdm import tqdm
    
import arcqua.fitting as af

import scintools.ththmod as thth
import pickle as pkl


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
    

class DDMStream:
    mandatoryParams = ['freq',
                                'prn',
                                'ddms',
                                'times',
                                'delay',
                                'doppler',
                                'specularDelay',
                                'specularDoppler',
                                'obsPos',
                                'specPos',
                                'emPos',
                                'obsVel',
                                'emVel'
                                ]
    expensiveParams = ['etas',
                            'etaErrors',
                            'chiArr',
                            'thetas']

    def __init__(self,freq, prn,
                 ddms, times,
                 delay, doppler, specularDelay, specularDoppler,
                 obsPos, specPos, emPos,
                 obsVel, emVel
                 ) -> None:
        self.freq = freq
        self.prn = prn

        self.ddms = ddms
        self.times = times

        self.offsetTime = (self.times-self.times[0]).to_value(u.s)

        self.delay = delay
        self.doppler = doppler
        self.specularDelay = specularDelay
        self.specularDoppler = specularDoppler

        self.obsPos = obsPos
        self.specPos = specPos
        self.emPos = emPos

        self.obsVel = obsVel
        self.emVel = emVel

        self.nSamples = self.ddms.shape[0]
        
        self._find_crossing()

    def _find_crossing(self):
        xAxis, yAxis, zAxis, psis, des, drs = af.coords(self.emPos,
                                                self.specPos,
                                                self.obsPos)
        xVel = np.sum(self.obsVel*xAxis,1)
        if xVel.min()<0 and xVel.max()>0:
            self.crossingID = np.argmin(np.abs(xVel))

    def plot_sample(self,id):
        plt.figure()
        extent = thth.ext_find(self.doppler,self.delay.to(u.us))
        plt.imshow(self.ddms[id],
                   origin='lower',aspect='auto',
                   extent=extent,
                   cmap='magma')
        plt.plot(self.specularDoppler[id],self.specularDelay[id].to(u.us),'g.')
        if hasattr(self, 'etas'):
            plt.plot(self.doppler,
                     (self.etas[id]*(self.doppler-self.specularDoppler[id])**2 + self.specularDelay[id]).to(u.us))
            plt.ylim(extent[2:])

    def animate_range(self,idMin = 0, idMax = np.inf, fps=10,fname = None):
        animMin = max(idMin,0)
        animMax = min(idMax,self.nSamples)
        i=animMin

        extent = thth.ext_find(self.doppler,self.delay.to(u.us))
        fig = plt.figure( figsize=(4,4))
        ax = plt.subplot(111)
        im = plt.imshow(self.ddms[i],
                        origin='lower',aspect='auto',
                        extent=extent,
                        interpolation='none',cmap='magma')
        im.set_clim(self.ddms[i].min(),self.ddms[i].max())
        apex, = plt.plot(self.specularDoppler[i].value,self.specularDelay[i].to_value(u.us),'g.')
        if hasattr(self, 'etas'):
            arc, = plt.plot(self.doppler.value,
                     (self.etas[i]*(self.doppler-self.specularDoppler[i])**2 + self.specularDelay[i]).to_value(u.us),'g')
            ax.set_ylim(extent[2:])


        def animate_func(i):
            id = i+animMin
            im.set_array(self.ddms[id])
            im.set_clim(self.ddms[id].min(),self.ddms[id].max())     
            apex.set_data(self.specularDoppler[id].value,self.specularDelay[id].to_value(u.us))
            if hasattr(self,'etas'):
                arc.set_data(self.doppler.value,
                     (self.etas[id]*(self.doppler-self.specularDoppler[id])**2 +
                      self.specularDelay[id]).to_value(u.us))
            return

        anim = animation.FuncAnimation(
                                    fig, 
                                    animate_func, 
                                    frames = animMax-animMin+1,
                                    interval = 1000 / fps, # in ms
                                    )
        if fname:
            anim.save(fname,fps=fps)
            plt.close(fig)
        else:
            plt.show()

    def fit_curvatures(self, fw = .1,
                       etaMin = 7e-13*u.s**3, etaMax = 1.2e-12*u.s**3,
                       nEtas = 100, edges = np.linspace(-2500, 2500, 256) * u.Hz, mode='square',progress = -np.inf):
        self.etas = np.zeros(self.nSamples)*u.s**3
        self.etaErrors = np.zeros(self.nSamples)*u.s**3
        if progress <0:
            it = range(self.nSamples)
        else:
            it = tqdm(range(self.nSamples),position=progress,leave=False)
        for id in it:
            etaSearch = np.linspace(etaMin << u.s**3,etaMax << u.s**3,nEtas)
            self.etas[id], self.etaErrors[id] = self.single_fit(id,etaSearch,edges,fw,mode=mode)
            

    def single_fit(self,id,etas,edges,fw=.1,plot=False,mode : str = 'square',progress = -np.inf):
        eigs = np.zeros(etas.shape[0])
        if progress <0:
            it = range(eigs.shape[0])
        else:
            it = tqdm(range(eigs.shape[0]),position=progress,leave=False)
        for i in it:
            ththMatrix = thth.thth_map(
                self.ddms[id],
                (self.delay-self.specularDelay[id]).to(u.us),
                (self.doppler-self.specularDoppler[id]).to(u.mHz),
                etas[i],
                edges.to(u.mHz),
                hermetian=False,
            ).real
            U,S,W=np.linalg.svd(ththMatrix)
            if 'square' in mode.lower():
                S=S**2
            if 'sub' in mode.lower():
                S-=np.median(S)
            if 'norm' in mode.lower():
                eigs[i] = S[0]/S[1]
            elif 'sum' in mode.lower():
                eigs[i] = np.sqrt(S[0]/np.sum(S))
            else:
                eigs[i]=S[0]
        etas = etas[np.isfinite(eigs)]
        eigs = eigs[np.isfinite(eigs)]
        try:
            # Reduced range around peak to be withing fw times curvature of
            #     maximum eigenvalue
            etas_fit = etas[
                np.abs(etas - etas[eigs == eigs.max()])
                < fw * etas[eigs == eigs.max()]
            ]
            eigs_fit = eigs[
                np.abs(etas - etas[eigs == eigs.max()])
                < fw * etas[eigs == eigs.max()]
            ]
            
            # Initial Guesses
            C = eigs_fit.max()
            x0 = etas_fit[eigs_fit == C][0].value
            if x0 == etas_fit[0].value:
                A = (eigs_fit[-1] - C) / ((etas_fit[-1].value - x0) ** 2)
            else:
                A = (eigs_fit[0] - C) / ((etas_fit[0].value - x0) ** 2)

            # Fit parabola around peak
            popt, pcov = curve_fit(
                thth.chi_par, etas_fit.value, eigs_fit, p0=np.array([A, x0, C])
            )
            
            # Record curvauture fit and error
            etaFit = popt[1] * u.us / u.mHz**2
            etaSig = (
                np.sqrt(
                    (eigs_fit - thth.chi_par(etas_fit.value, *popt)).std()
                    / np.abs(popt[0])
                )
                * u.us
                / u.mHz**2
            )
            if plot:
                plt.figure()
                plt.plot(etas,eigs,'.')
                plt.plot(etas_fit,thth.chi_par(etas_fit.value, *popt))
                plt.xlabel(r'$\eta~\left(\rm{s}^3\right)$')
                plt.ylabel('Peak Eigenvalue')
            return(etaFit,etaSig)
        except:
            return(np.nan,np.nan)
        

    def fit_thetas(self, thetas = np.linspace(-90,90,361)[:-1]*u.deg):
        if not hasattr(self,'etas'):
            self.fit_curvatures()
        xAxis, yAxis, zAxis, psis, des, drs = af.coords(self.emPos,
                                                self.specPos,
                                                self.obsPos)

        obsVel2 = af.convertToSpecular(self.obsVel, xAxis, yAxis, zAxis)
        emVel2 = af.convertToSpecular(self.emVel, xAxis, yAxis, zAxis)

        specLoc = EarthLocation(x=self.specPos[:,0],y=self.specPos[:,1],z=self.specPos[:,2])

        self.thetas = thetas
        self.chiArr = np.zeros((self.thetas.shape[0],self.nSamples))
        for i,theta in enumerate(self.thetas):

            offsetLoc = EarthLocation(lat = specLoc.lat+1e-2*np.sin(theta)*u.deg,
                                    lon = specLoc.lon+1e-2*np.cos(theta)*u.deg)
            offset = np.array(offsetLoc.to_geocentric()).T-np.array(specLoc.to_geocentric()).T
            offset/=np.sqrt(np.sum(offset**2,1))[:,np.newaxis]
            
            waveX = np.sum(offset*xAxis,1)
            waveY = np.sum(offset*yAxis,1)

            thetaW = np.arctan2(waveY,waveX)
            thetaI = af.wavesToImages(thetaW,psis)


            etas = af.calcCurvatures(thetaI, psis, des, drs, obsVel2, emVel2, self.freq)
            self.chiArr[i] = (np.abs(etas-self.etas)**2)/self.etaErrors**2

    def plot_chis(self, angles = [], width = 0, format = 'imshow', fname = None):
        if not hasattr(self,'chiArr'):
            self.fit_thetas()
        assert format in ['imshow', 'pcolor']
        plt.figure()
        if format == 'imshow':
            dth = (self.thetas[1]-self.thetas[0]).value/2
            th0 = self.thetas[0].value
            th1 = self.thetas[-1].value
            for i in range(self.nSamples):
                plt.imshow(self.chiArr[:,i:i+1],origin='lower',
                        aspect='auto',
                        extent=[self.offsetTime[i]-.5,self.offsetTime[i]+.5,th0-dth,th1+dth],norm=colors.LogNorm(),cmap='magma')
            plt.xlim((-.5,self.offsetTime.max()+.5))
        
        if format == 'pcolor':
            x,y = np.meshgrid(self.offsetTime,self.thetas.value)
            plt.pcolormesh(x,y,self.chiArr,norm=colors.LogNorm(),cmap='magma')
        
        for angle in angles:
            for offset in [-width,width]:
                plt.axhline(angle+offset)

        plt.xlabel(r'\Delta t ~\left(\rm{s}\right)')
        plt.ylabel(r'\theta~\left(^{\circ}\right)')
        plt.tight_layout()
        if fname:
            plt.savefig(fname)

    def save(self,filename):
        with open(f'{filename}.pkl', 'wb') as handle:
            pkl.dump(self, handle, protocol=pkl.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls,filename,**kwargs) -> object:
        with open(f'{filename}.pkl', 'rb') as handle:
            loaded : object = pkl.load(handle)
        
        loadDict = {}
        for parameter in cls.mandatoryParams:
            if hasattr(loaded,parameter):
                loadDict.update({parameter : loaded.__getattribute__(parameter)})
            elif parameter in kwargs.keys():
                loadDict.update({parameter : loaded.__getattribute__(parameter)})
            else:
                print(f'Missing parameter: {parameter}')
        new = cls(**loadDict)
        for parameter in cls.expensiveParams:
            if hasattr(loaded,parameter):
                new.__setattr__(parameter,loaded.__getattribute__(parameter))
            elif parameter in kwargs.keys():
                new.__setattr__(parameter,kwargs[parameter])
        new.fromFile = filename
        return(new)


        

class TRITON():
    def __init__(self,year,day) -> None:
        self.freq : u.Quantity = 1.57542*u.GHz
        self.scale : u.Quantity = 1.0 * u.s / (65536.0 * 16.0)
        self.year : int = year
        self.day : int = day
        self.prefix : str = str(year)+'_'+str(day).zfill(3)
        self.streams : list[DDMStream] = list()
        pass

    def load_data(self, dataDir, spDir, time, mode='raw', streamDir = '.') -> None:
        assert mode in ['raw', 'power']
        fileName : str = os.path.join(dataDir,f'{self.prefix}_{time}_TRITON_CorDDM.nc')
        data = xr.load_dataset(fileName)
        if mode =='raw':
            ddms = np.array(data.rawDDM)
        else:
            ddms = np.array(data.DDMpower)

        

        sampleNumber = np.linspace(0,ddms.shape[0]-1,
                                   ddms.shape[0],
                                   dtype=int)[:,np.newaxis]*np.ones(ddms.shape[:2])
        channelNumber = np.ones(ddms.shape[:2])*np.linspace(0,ddms.shape[1]-1,
                                                            ddms.shape[1],
                                                            dtype=int)
        prn = np.array(data.PRN)
        times = (np.array(data.GPSSec)*u.s+np.array(data.GPSWeek)*u.week).value

        flags = np.array(data.quality_flags)
        flags2 = np.array(data.quality_flags_2)

        times = Time((np.ones(flags.shape)*times[:,np.newaxis])[flags==0],format='gps')
        ddms = np.transpose(ddms[flags == 0], (0, 2, 1))
        prn = prn[flags==0]
        flags2 = flags2[flags==0]
        sampleNumber = sampleNumber[flags==0]
        channelNumber = channelNumber[flags==0]


        observerPos = np.array([(np.ones(flags.shape)*np.array(data.SVPosX)[:,np.newaxis])[flags==0],
                                (np.ones(flags.shape)*np.array(data.SVPosY)[:,np.newaxis])[flags==0],
                                (np.ones(flags.shape)*np.array(data.SVPosZ)[:,np.newaxis])[flags==0]]).T*u.m
        observerVel = np.array([(np.ones(flags.shape)*np.array(data.SVVelX)[:,np.newaxis])[flags==0],
                                (np.ones(flags.shape)*np.array(data.SVVelY)[:,np.newaxis])[flags==0],
                                (np.ones(flags.shape)*np.array(data.SVVelZ)[:,np.newaxis])[flags==0]]).T*u.m/u.s
        specularPos = np.array([np.array(data.SPPosX)[flags==0],
                                np.array(data.SPPosY)[flags==0],
                                np.array(data.SPPosZ)[flags==0]]).T*u.m
        sourcePos = np.array([np.array(data.GPSPosX)[flags==0],
                                np.array(data.GPSPosY)[flags==0],
                                np.array(data.GPSPosZ)[flags==0]]).T*u.m
        sourceVel = np.array([np.array(data.GPSVelX)[flags==0],
                                np.array(data.GPSVelY)[flags==0],
                                np.array(data.GPSVelZ)[flags==0]]).T*u.m/u.s

        spFile = np.loadtxt(os.path.join(spDir,f"{self.prefix}_{time}_TRITON_roughness_windspeed.nc.txt"),skiprows=1)

        spChannelNumber = spFile[:,0]
        spSampleNumber = spFile[:,1]
        spDelay = spFile[:,6]*self.scale
        spDoppler = spFile[:,7]*u.Hz

        delayRes : u.Quantity = data.attrs['codephase resolution (chip)']*self.scale
        dopplerRes : u.Quantity = data.attrs['Doppler resolution (Hz)']*u.Hz
        delay  = np.linspace(0, ddms.shape[1] - 1, ddms.shape[1]) * delayRes
        doppler = np.linspace(0, ddms.shape[2] - 1, ddms.shape[2]) * dopplerRes
        delay -= delay[65] 
        doppler -= doppler[33] 

        for usePrn in np.unique(prn):
            spID,ddmID = self._select_indices(usePrn,prn,
                                              spChannelNumber,spSampleNumber,
                                              sampleNumber,channelNumber)
            if len(spID)>0:
                filename = os.path.join(streamDir,f'{self.prefix}_{time}_{usePrn}')
                if os.path.exists(f'{filename}.pkl'):
                    self.streams.append(DDMStream.from_pickle(filename))
                    print(f'loaded {filename}')
                else:
                    newStream = DDMStream(self.freq,usePrn, ddms[ddmID], times[ddmID],
                                                delay, doppler,
                                                spDelay[spID],spDoppler[spID],
                                                observerPos[ddmID],specularPos[ddmID],sourcePos[ddmID],
                                                observerVel[ddmID],sourceVel[ddmID]
                                                )
                    newStream.save(filename)
                    self.streams.append(newStream)
                    

    def _select_indices(self, usePrn, prn,
                        spChannelNumber, spSampleNumber,
                        sampleNumber, channelNumber):
        spID = list()
        ddmID = list()
        for i in range(channelNumber.shape[0]):
            if not prn[i] == usePrn:
                continue
            cnp = channelNumber[i]
            snp = sampleNumber[i]
            for j in range(spChannelNumber.shape[0]):
                cns = spChannelNumber[j]
                sns = spSampleNumber[j]
                if (cns==cnp) and  (sns==snp):
                    spID.append(j)
                    ddmID.append(i)
                    break
        return(spID,ddmID)
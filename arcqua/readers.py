
## General Python
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

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

##Downloading Data
import mechanize
from getpass import getpass
import datetime

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

    def plot_sample(self,id, fname = None):
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
        if fname:
            plt.savefig(fname)

    def animate_range(self,idMin = 0, idMax = np.inf, fps=10,fname = None):
        animMin = max(idMin,0)
        animMax = min(idMax,self.nSamples-1)
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
                       etaMin = 3e-13*u.s**3, etaMax = 2.0e-12*u.s**3,
                       nEtas = 200, edges = np.linspace(-2500, 2500, 256) * u.Hz,
                       mode='square',progress = -np.inf, pool = None, cutTHTH = False,**kwargs):
        self.etas = np.zeros(self.nSamples)*u.s**3
        self.etaErrors = np.zeros(self.nSamples)*u.s**3
        if progress <0:
            it = range(self.nSamples)
        else:
            it = tqdm(range(self.nSamples),position=progress,leave=False)
        for id in it:
            etaSearch = np.linspace(etaMin << u.s**3,etaMax << u.s**3,nEtas)
            self.etas[id], self.etaErrors[id] = self.single_fit(id,etaSearch,edges,fw,mode=mode,progress=progress+1,pool=pool,cutTHTH=cutTHTH)
            
    def single_fit(self,id,etas,edges,fw=.1,plot=False,mode : str = 'square',progress = -np.inf, pool = None, cutTHTH = False):
        eigs = np.zeros(etas.shape[0])
        if pool:
            it = range(eigs.shape[0])
            args = []
            for i in it:
                params = [id,etas[i],edges,mode]
                args.append(params)
            res = pool.map(self.find_evals_pool, args)
            for i in it:
                eigs[i]=res[i]
        else:
            if progress <0:
                it = range(eigs.shape[0])
            else:
                it = tqdm(range(eigs.shape[0]),position=progress,leave=False)
            for i in it:
                eigs[i]=self.find_evals(id,etas[i],edges,mode, cutTHTH=cutTHTH)
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
        except Exception as e:
            return(np.nan,np.nan)

    def find_evals_pool(self,params):
        return(self.find_evals(*params))
    
    def find_evals(self,id,eta,edges,mode : str = 'square', pad=False, cutTHTH = False): 
        ththMatrix = thth.thth_map(
                self.ddms[id],
                (self.delay-self.specularDelay[id]).to(u.us),
                (self.doppler-self.specularDoppler[id]).to(u.mHz),
                eta,
                edges.to(u.mHz),
                hermetian=False,
            ).real
        cents = (edges[1:]+edges[:-1])/2
        if cutTHTH:
            ththMatrix = ththMatrix[:,eta*cents**2+self.specularDelay[id] < self.delay.max()]
            ththMatrix = ththMatrix[-eta*cents**2+self.specularDelay[id] > self.delay.min()]
        U,S,W=np.linalg.svd(ththMatrix)
        if 'square' in mode.lower():
            S=S**2
        if 'sub' in mode.lower():
            S-=np.median(S)
        if 'norm' in mode.lower():
            eig = (S[0]-np.median(S))/(S[1]-np.median(S))
        elif 'sum' in mode.lower():
            eig = S[0]**2/np.sum(S**2)
        else:
            eig=S[0]
        return(eig)

    def fit_thetas(self, thetas = np.linspace(-90,90,361)[:-1]*u.deg, progress = -np.inf, pool = None):
        if not hasattr(self,'etas'):
            self.fit_curvatures(progress = progress, pool = pool)
        xAxis, yAxis, zAxis, psis, des, drs = af.coords(self.emPos,
                                                self.specPos,
                                                self.obsPos)

        obsVel2 = af.convertToSpecular(self.obsVel, xAxis, yAxis, zAxis)
        emVel2 = af.convertToSpecular(self.emVel, xAxis, yAxis, zAxis)

        specLoc = EarthLocation(x=self.specPos[:,0],y=self.specPos[:,1],z=self.specPos[:,2])

        self.thetas = thetas
        self.chiArr = np.zeros((self.thetas.shape[0],self.nSamples))
        if progress <0:
            it = self.thetas
        else:
            it = tqdm(self.thetas,position=progress,leave=False)
        for i,theta in enumerate(it):

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

    def fit_asymm(self,edges = np.linspace(-2500, 2500, 256) * u.Hz,**kwargs):
        if not hasattr(self,"etas"):
            self.fit_curvatures(**kwargs)
        self.asymm = np.zeros(self.nSamples)
        etaInterp = interp1d(self.offsetTime[np.isfinite(self.etas)],
                                self.etas[np.isfinite(self.etas)],
                                bounds_error=False,fill_value='extrapolate',kind='cubic'
                                )
        cents=(edges[1:]+edges[:-1])/2
        for sampleID in range(self.nSamples):
            eta = etaInterp(self.offsetTime[sampleID])<<self.etas.unit
            ththMatrix= thth.thth_map(
                self.ddms[sampleID],
                (self.delay-self.specularDelay[sampleID]).to(u.us),
                (self.doppler-self.specularDoppler[sampleID]).to(u.mHz),
                eta,
                edges.to(u.mHz),
                hermetian=False,
            )
            ththMatrix = ththMatrix[-eta*cents**2+self.specularDelay[sampleID]>self.delay.min()][:,eta*cents**2+self.specularDelay[sampleID]<self.delay.max()]
            centsX = cents[eta*cents**2+self.specularDelay[sampleID]<self.delay.max()]
            U,S,W = np.linalg.svd(ththMatrix)
            left = W[0,centsX<0]
            right = W[0,centsX>0]
            self.asymm[sampleID]=(np.sum(left)-np.sum(right))/((np.sum(left)+np.sum(right))/2)



    def plot_chis(self, angles = [], width = 0, format = 'imshow', fname = None):
        if not hasattr(self,'chiArr'):
            self.fit_thetas()
        assert format in ['imshow', 'pcolor']
        plt.figure()
        if format == 'imshow':
            dth = (self.thetas[1]-self.thetas[0]).value/2
            th0 = self.thetas[0].value
            th1 = self.thetas[-1].value
            mx = 10
            mn = 1
            chiArr2 = np.zeros((self.chiArr.shape[0],int(np.round(self.offsetTime.max())+1)))*np.nan
            for i1,i2 in enumerate(self.offsetTime):
                chiArr2[:,int(np.round(i2))]=self.chiArr[:,i1]
            plt.imshow(chiArr2,origin='lower',
                    aspect='auto',
                    extent=[self.offsetTime[0]-.5,self.offsetTime[-1]+.5,th0-dth,th1+dth],norm=colors.LogNorm(vmin=mn,vmax=mx),cmap='magma',interpolation='None')
            plt.xlim((-.5,self.offsetTime.max()+.5))
        
        if format == 'pcolor':
            x,y = np.meshgrid(self.offsetTime,self.thetas.value)
            plt.pcolormesh(x,y,self.chiArr,norm=colors.LogNorm(),cmap='magma')
        
        for angle in angles:
            for offset in [-width,width]:
                plt.axhline(angle+offset)

        plt.xlabel(r'$\Delta t ~\left(\rm{s}\right)$')
        plt.ylabel(r'$\theta~\left(^{\circ}\right)$')
        plt.tight_layout()
        if fname:
            plt.savefig(fname)

    def save(self,fileName=None):
        if fileName:
            with open(fileName, 'wb') as handle:
                pkl.dump(self, handle, protocol=pkl.HIGHEST_PROTOCOL)
            return
        if hasattr(self,'loadPath'):
            with open(self.loadPath, 'wb') as handle:
                pkl.dump(self, handle, protocol=pkl.HIGHEST_PROTOCOL)
            return
        raise AttributeError("Stream wasn't loaded from an existing file. File name required.")

    @classmethod
    def from_pickle(cls,filename,**kwargs) -> object:
        with open(f'{filename}', 'rb') as handle:
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
        new.__setattr__('loadPath',os.path.abspath(filename))
        return(new)


        

class TRITON():
    def __init__(self,version = '1.0', rootDir = '.') -> None:
        self.rootDir = rootDir
        self.version = version
        self.streams = []

    def load_data(self, mode='raw',source='TRITON',verbose=False,**kwargs) -> None:
        assert mode in ['raw', 'power']
        assert source.lower() in ['triton', 'cygnss']
        dateString = self.get_date_string(**kwargs)
        if not dateString:
            raise ValueError('No matching call signature for date format')

        dataDir = os.path.join(self.rootDir,dateString,source.lower())
        streamDir = os.path.join(self.rootDir,dateString,'streams')
        if not os.path.exists(streamDir):
                os.makedirs(streamDir)
        if source.lower() == 'triton':
            if not os.path.exists(dataDir):
                self.download_triton_data(**kwargs)
            elif len(os.listdir(dataDir))<2:
                self.download_triton_data(**kwargs)
        
            if 'groundTimes' in kwargs:
                groundTimes = kwargs['groundTimes']
                if not isinstance(groundTimes,list):
                    groundTimes = [groundTimes]
            else:
                groundTimes = np.array([file[7:13] for file in os.listdir(dataDir)])
                groundTimes = np.unique(groundTimes)
            fileNames = [file for file in os.listdir(dataDir) if file[7:13] in groundTimes and 'CorDDM' in file and file[-6:-3]==self.version]
        
        if source.lower() == 'cygnss':
            dataDir = os.path.join(dataDir,'data')
                
            if 'cygCodes' in kwargs:
                cygCodes = kwargs['cygCodes']
                if not isinstance(cygCodes,list):
                    cygCodes = [cygCodes]
                cygCodes = [code.zfill(2) for code in cygCodes]
            else:
                cygCodes = [f[3:5] for f in os.listdir(dataDir) if f[-3:]=='.nc']
                cygCodes = np.unique(cygCodes)
                
            fileNames = [file for file in os.listdir(dataDir) if file[3:5] in cygCodes and file[-3:]=='.nc']   
                
        for fileName in fileNames:
            # if len(fileName)>0:
            #     fileName = fileName[0]
            # else:
            #     continue
            
            
            if source.lower() == 'triton':
                (freq,prn,ddms,times,
                 delay,doppler,spDelay,spDoppler,
                 observerPos,specularPos,sourcePos,
                 observerVel,sourceVel) = self.parse_triton_data(os.path.join(dataDir,fileName),mode)
                obsName = 'triton_'+fileName[7:28]
                
            if source.lower() == 'cygnss':
                (freq,prn,ddms,times,
                 delay,doppler,spDelay,spDoppler,
                 observerPos,specularPos,sourcePos,
                 observerVel,sourceVel) = self.parse_cygnss_data(os.path.join(dataDir,fileName))
                obsName = fileName[:5]
    
            for usePrn in np.unique(prn):
                offsetTime = (times[prn == usePrn] - times[prn == usePrn][0]).to_value(u.s)
                jumps = np.ravel(np.argwhere(np.diff(offsetTime)>60))+1
                jumps=np.concatenate((np.zeros(1,dtype=int),
                                      jumps,
                                      np.ones(1,dtype=int)*offsetTime.shape[0]))
                
                for section in range(jumps.shape[0]-1):
                    cut = slice(jumps[section],jumps[section+1])
                    if offsetTime[cut].shape[0]>60:
                        fileName = os.path.join(streamDir,f'{obsName}_{usePrn}_v{self.version}_{section}.pkl')
                        if os.path.exists(fileName):
                            self.streams.append(DDMStream.from_pickle(fileName))
                            if verbose:
                                print(f'loaded {fileName}')
                        else:
                            newStream = DDMStream(freq,usePrn, ddms[prn == usePrn][cut], times[prn == usePrn][cut],
                                                        delay, doppler,
                                                        spDelay[prn == usePrn][cut],spDoppler[prn == usePrn][cut],
                                                        observerPos[prn == usePrn][cut],specularPos[prn == usePrn][cut],sourcePos[prn == usePrn][cut],
                                                        observerVel[prn == usePrn][cut],sourceVel[prn == usePrn][cut]
                                                        )
                            newStream.save(fileName=fileName)
                            newStream.loadPath = os.path.abspath(fileName)
                            self.streams.append(newStream)
    def parse_cygnss_data(self,fileName):
        freq = 1.57542*u.GHz
        data = xr.load_dataset(fileName)
        
        ddms = np.array(data.brcs)
        times = np.array(data.ddm_timestamp_utc)
        prn = np.array(data.prn_code).astype(int)
        
        ##Get Errors
        useable = (np.array(data.quality_flags) == 0) * (np.array(data.quality_flags_2) != 0)
        
        times = Time(np.repeat(times[:,np.newaxis],useable.shape[1],1)[useable])
        prn=prn[useable]
        ddms=ddms[useable]
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
        
        
        sourcePos = np.transpose(np.array([sourceX, sourceY, sourceZ]), (1, 2, 0))[useable]*u.m
        specularPos = np.transpose(np.array([specularX, specularY, specularZ]), (1, 2, 0))[useable]*u.m
        observerPos = np.array([observerX, observerY, observerZ]).T*u.m
        observerPos = np.repeat(observerPos[:,np.newaxis,:],useable.shape[1],1)[useable]
        
        sourceVel = np.transpose(np.array([sourceVX, sourceVY, sourceVZ]), (1, 2, 0))[useable]*u.m/u.s
        specularVel = np.transpose(np.array([specularVX, specularVY, specularVZ]), (1, 2, 0))[useable]*u.m/u.s
        observerVel = np.array([observerVX, observerVY, observerVZ]).T*u.m/u.s
        observerVel = np.repeat(observerVel[:,np.newaxis,:],useable.shape[1],1)[useable]
        
        ## Delay and Doppler
        nDoppler = data.dims['doppler']
        nDelay = data.dims['delay']
        nDDM = data.dims['ddm']
        chipSize = ((1./1023000.)*u.s).to(u.us)
        delay = np.linspace(-(nDelay-1)//2,(nDelay-1)//2,nDelay)*float(data.delay_resolution)*chipSize
        doppler = np.linspace(-(nDoppler-1)//2,(nDoppler-1)//2,nDoppler)*float(data.dopp_resolution)*u.Hz
        spDelay = np.array(data.brcs_ddm_sp_bin_delay_row)[useable]*float(data.delay_resolution)*chipSize+delay[0]
        spDoppler = np.array(data.brcs_ddm_sp_bin_dopp_col)[useable]*float(data.dopp_resolution)*u.Hz+doppler[0]

        return(freq,
               prn,
               ddms,
               times,
               delay,doppler,
               spDelay,spDoppler,
               observerPos,specularPos,sourcePos,
               observerVel,sourceVel)
        
    def parse_triton_data(self,fileName,mode):
        freq = 1.57542*u.GHz
        scale = 1.0 * u.s / (65536.0 * 16.0)
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
        useable = (flags == 0)*(flags2 != 0)

        times = Time((np.ones(flags.shape)*times[:,np.newaxis])[useable],format='gps')
        ddms = np.transpose(ddms[useable], (0, 2, 1))
        prn = prn[useable]
        sampleNumber = sampleNumber[useable]
        channelNumber = channelNumber[useable]


        observerPos = np.array([(np.ones(flags.shape)*np.array(data.SVPosX)[:,np.newaxis])[useable],
                                (np.ones(flags.shape)*np.array(data.SVPosY)[:,np.newaxis])[useable],
                                (np.ones(flags.shape)*np.array(data.SVPosZ)[:,np.newaxis])[useable]]).T*u.m
        observerVel = np.array([(np.ones(flags.shape)*np.array(data.SVVelX)[:,np.newaxis])[useable],
                                (np.ones(flags.shape)*np.array(data.SVVelY)[:,np.newaxis])[useable],
                                (np.ones(flags.shape)*np.array(data.SVVelZ)[:,np.newaxis])[useable]]).T*u.m/u.s
        specularPos = np.array([np.array(data.SPPosX)[useable],
                                np.array(data.SPPosY)[useable],
                                np.array(data.SPPosZ)[useable]]).T*u.m
        sourcePos = np.array([np.array(data.GPSPosX)[useable],
                                np.array(data.GPSPosY)[useable],
                                np.array(data.GPSPosZ)[useable]]).T*u.m
        sourceVel = np.array([np.array(data.GPSVelX)[useable],
                                np.array(data.GPSVelY)[useable],
                                np.array(data.GPSVelZ)[useable]]).T*u.m/u.s


        metaName = fileName.split('CorDDM')[0]+'metadata'+fileName.split('CorDDM')[1]
        meta = xr.load_dataset(metaName)
        spDelay = np.array(meta['SP_CodePhase_shift'])[useable]*scale
        spDoppler = np.array(meta['SP_DopplerFrequency_shift'])[useable]*u.Hz

        delayRes : u.Quantity = data.attrs['codephase resolution (chip)']*scale
        dopplerRes : u.Quantity = data.attrs['Doppler resolution (Hz)']*u.Hz
        delay  = np.linspace(0, ddms.shape[1] - 1, ddms.shape[1]) * delayRes
        doppler = np.linspace(0, ddms.shape[2] - 1, ddms.shape[2]) * dopplerRes
        delay -= delay[65] 
        doppler -= doppler[33]
        return(freq,
               prn,
               ddms,
               times,
               delay,doppler,
               spDelay,spDoppler,
               observerPos,specularPos,sourcePos,
               observerVel,sourceVel)
        
    def get_date_string(self,**kwargs):
        if 'dateString' in kwargs:
            return(kwargs['dateString'])
        if 'date' in kwargs:
            date = kwargs['date']
            if isinstance(date,int):
                date = str(date)
            date = datetime.datetime(int(date[:4]),int(date[4:6]),int(date[6:8]))
            date = date.timetuple()
            year = date.tm_year
            day = str(date.tm_yday).zfill(3)
            dateString = f'{year}.{day}'
            return(dateString)
        elif 'year' in kwargs:
            if 'day' in kwargs:
                if 'month' in kwargs:
                    date = datetime.datetime(int(kwargs['year']),int(kwargs['month']),int(kwargs['day']))
                    date = date.timetuple()
                    year = date.tm_year
                    day = str(date.tm_yday).zfill(3)
                    dateString = f'{year}.{day}'
                    return(dateString)
                else:
                    year = kwargs['year']
                    day = str(kwargs['day']).zfill(3)
                    dateString = f'{year}.{day}'
                    return(dateString)

        return(None)
                    

    def download_triton_data(self,**kwargs):
        if 'user' in kwargs:
            user = kwargs['user']
        else:
            user = input('Username:')
        if 'pwd' in kwargs:
            pwd = kwargs['pwd']
        elif 'password' in kwargs:
            pwd = kwargs['password']
        else:
            pwd = getpass('Password:')
        
        dateString = self.get_date_string(**kwargs)

        if not dateString:
            raise ValueError('No matching call signature for date format')
        baseURL = 'https://tacc.cwa.gov.tw/data-service/triton/level1b/'
        ## Prepare browser
        br = mechanize.Browser()
        br.set_handle_robots(False)
        br.set_handle_refresh(False) 
        br.add_password(baseURL, user, pwd)
        br.open(baseURL)

        dataFolderUrls = [link.absolute_url for link in br.links() if link.text[:8]==dateString]
        if len(dataFolderUrls)>0:
            if not os.path.exists(os.path.join(self.rootDir,dateString)):
                os.makedirs(os.path.join(self.rootDir,dateString))
            br.open(dataFolderUrls[0])
            files = [(link.absolute_url,link.text) for link in br.links() if link.text[:7]=="TRITON_" and link.text[-2:]=='nc']
            for pair in files:
                br.add_password(pair[0], user, pwd)
                if 'CorDDM' in pair[1] or 'metadata' in pair[1]:
                    fname = os.path.join(self.rootDir,dateString,'triton',pair[1])
                    if os.path.exists(fname):
                        print(f'File {pair[1]} already exists')
                    else:
                        br.retrieve(pair[0],fname)[0]
        else:
            print(f'No Matching Folder exists for {dateString}')


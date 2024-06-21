## General Python
import numpy as np
import os
from tqdm import tqdm

## Astropy Tools
from astropy.time import Time
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import EarthLocation

## Plotting
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import LineString
from shapely.ops import split
from shapely.affinity import translate

## Scintools
import scintools.ththmod as THTH



def coords(emitters, speculars, receiver):
    """
    Calculate basis vectors, inclinations, and distances for each specular point

    Parameters
    ----------
    emitters : 2D Array
        XYZ coordinates of the source
    speculars : 2D Array
        XYZ coordinates of the specular points
    reciever : 1D Array
        XYZ coordinates of the reciever
    """
    eVec = emitters - speculars
    eHat = eVec / np.sqrt(np.sum(eVec**2, 1))[:,np.newaxis]
    rVec = receiver - speculars
    rHat = rVec / np.sqrt(np.sum(rVec**2, 1))[:,np.newaxis]
    zAxis = (eHat + rHat) / 2
    zAxis /= np.sqrt(np.sum(zAxis**2, 1))[:,np.newaxis]
    xAxis = receiver - emitters
    xAxis -= np.sum(xAxis * zAxis, 1)[:,np.newaxis] * zAxis
    xAxis /= np.sqrt(np.sum(xAxis**2, 1))[:,np.newaxis]
    yAxis = np.cross(xAxis,zAxis)
    psis = np.arccos(np.sum(rHat * xAxis, 1))
    des = np.sqrt(np.sum(eVec**2, 1))
    drs = np.sqrt(np.sum(rVec**2, 1))
    return (xAxis, yAxis, zAxis, psis, des, drs)


def convertToSpecular(vecs, xAxis, yAxis, zAxis):
    unitless = (vecs * u.s).value
    if unitless.shape == xAxis.shape:
        vecs2 = np.array(
            [
                np.sum(unitless * xAxis, 1),
                np.sum(unitless * yAxis, 1),
                np.sum(unitless * zAxis, 1),
            ]
        ).T
    else:
        vecs2 = np.array(
            [
                np.sum(unitless * xAxis, 1),
                np.sum(unitless * yAxis, 1),
                np.sum(unitless * zAxis, 1),
            ]
        ).T
    return vecs2 * (vecs * u.s).unit / u.s

def thetaConvert(theta, xAxis, yAxis, zAxis):
    """
    Convert angle from the first specular point to all others

    Parameters
    ----------
    theta : float
        Orientation angle at the first specular point
    xAxis : 2D Array
        Normal vectors for the x axes at all specular points
    yAxis : 2D Array
        Normal vectors for the y axes at all specular points
    zAxis : 2D Array
        Normal vectors for the z axes at all specular points
    """
    vecs = np.ones(xAxis.shape)
    vecs *= np.cos(theta) * xAxis[0] + np.sin(theta) * yAxis[0]
    vecs-=np.sum(vecs*zAxis,1)[:,np.newaxis]*zAxis
    thetas = np.arctan2(
        np.sum(vecs * yAxis, 1),
        np.sum(vecs * xAxis, 1),
    ) << u.rad
    while thetas.min() < -180 * u.deg:
        thetas[thetas < -180 * u.deg] += 360 * u.deg
    while thetas.max() > 180 * u.deg:
        thetas[thetas > 180 * u.deg] -= 360 * u.deg
    return thetas.to(u.deg)

def toUV(emitters, speculars, receiver, theta, id=0):
    xAxis, yAxis, zAxis, psis, des, drs = coords(emitters, speculars, receiver)
    thetas = thetaConvert(theta, xAxis, yAxis, zAxis)
    velFit = xAxis[id] * np.cos(thetas[id]) + yAxis[id] * np.sin(thetas[id])

    specularLoc = EarthLocation(
        speculars[id,0],
        speculars[id,1],
        speculars[id,2],
    )
    lat = specularLoc.lat.to(u.deg)
    lon = specularLoc.lon.to(u.deg)
    latUnit = np.array(
        [-np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)]
    )
    lonUnit = np.array([-np.sin(lon), np.cos(lon), 0])
    UV = np.array([np.dot(velFit, lonUnit), np.dot(velFit, latUnit)])
    UV /= np.sqrt(np.sum(UV**2))
    return UV

def wavesToImages(thetas, psis):
    thetaIs = np.arctan(np.tan(thetas) * (np.sin(psis) ** 2))
    thetaIs[thetas < -90 * u.deg] -= 180 * u.deg
    thetaIs[thetas > 90 * u.deg] += 180 * u.deg
    return thetaIs

def calcCurvatures(thetaIs, psis, des, drs, Vrs, Ves, freq):
    nSource = psis.shape[0]
    As = (
        (1 - (np.cos(psis) * np.cos(thetaIs)) ** 2)
        * ((1 / des) + (1 / drs))
        / (2 * const.c)
    )
    sHats = np.array([np.cos(thetaIs).value, np.sin(thetaIs).value, np.zeros(nSource)]).T
    rHats = np.array([np.cos(psis).value, np.zeros(nSource), np.sin(psis).value]).T
    eHats = np.array([-np.cos(psis).value, np.zeros(nSource), np.sin(psis).value]).T
    cosProd=np.cos(psis) * np.cos(thetaIs)
    Bs = (freq / (const.c * drs)) * np.sum(
        Vrs * (sHats - rHats * cosProd[:,np.newaxis])
        + Ves * (sHats + eHats * cosProd[:,np.newaxis]) * (drs / des)[:,np.newaxis],
        1,
    )
    etas = (As / (Bs**2)).to(u.s**3)
    return etas

def pad_DDM(ddms,doppler,delay,fd0,tau0,npad=1):
    newSize = npad+1
    dopplerPad = np.linspace(-newSize, newSize, 1 + (doppler.shape[0] - 1) * newSize) * doppler.max()
    delayPad = np.linspace(-newSize, newSize, 1 + (delay.shape[0] - 1) * newSize) * delay.max()
    ddmsPad = np.random.normal(
        0,
        1,
        (ddms.shape[0], 1 + (delay.shape[0] - 1) * newSize, 1 + (doppler.shape[0] - 1) * newSize),
    )
    ddmsPad *= np.std(ddms[:, 0, :], axis=1)[:, np.newaxis, np.newaxis]
    ddmsPad[
        :,
        npad*(delay.shape[0] - 1)//2 : npad*(delay.shape[0] - 1)//2 + delay.shape[0],
        npad*(doppler.shape[0] - 1)//2 : npad*(doppler.shape[0] - 1)//2 + doppler.shape[0],
    ] = np.copy(ddms)
    for i in range(ddmsPad.shape[0]):
        idFd = int(np.round((fd0[i]-dopplerPad[0]) / (dopplerPad[1] - dopplerPad[0])))
        idTau = int(np.round((tau0[i]-delayPad[0]) / (delayPad[1] - delayPad[0])))
        ddmsPad[i, idTau, idFd] = 0
    return(dopplerPad,delayPad,ddmsPad)

def powerCalc(
    theta,
    ddms,
    doppler,
    delay,
    fd0,
    tau0,
    emitters,
    speculars,
    receiver,
    emitterVels,
    receiverVel,
    freq,
    nedge=200,
    edgelim=0,
):
    if edgelim == 0:
        edgelim = doppler.max().value
    nSource = emitters.shape[0]
    xAxis, yAxis, zAxis, psis, des, drs = coords(emitters, speculars, receiver)
    thetas = thetaConvert(theta, xAxis, yAxis, zAxis)
    thetaIs = wavesToImages(thetas, psis)
    emitterVels2 = convertToSpecular(emitterVels, xAxis, yAxis, zAxis)
    receiverVel2 = convertToSpecular(receiverVel, xAxis, yAxis, zAxis)
    etas = calcCurvatures(thetaIs, psis, des, drs, receiverVel2, emitterVels2, freq)
    powerFrac = np.zeros(nSource)
    asymm = np.zeros(nSource)
    edges = np.linspace(-edgelim, edgelim, nedge) * u.Hz
    nc = (nedge // 2) - 1
    # print(etas)
    for i in range(nSource):
        thth = THTH.thth_map(
            ddms[i],
            delay - tau0[i],
            doppler - fd0[i],
            etas[i],
            edges,
            hermetian=False,
        ).real
        if np.isfinite(thth.max()):
            U, S, W = np.linalg.svd(thth)
            powerFrac[i] = np.sqrt(S.max() ** 2 / np.sum(S**2))
            asymm[i] = (np.sum(np.abs(W[0, :nc])) - np.sum(np.abs(W[0, nc + 1 :]))) / (
                np.sum(np.abs(W[0])) - np.abs(W[0, nc])
            )
        else:
            powerFrac[i] = np.nan
            asymm[i] = np.nan
    powerFrac[np.invert(np.isfinite(powerFrac))] = np.nan
    return powerFrac, asymm

def search(
    ddms,
    doppler,
    delay,
    fd0,
    tau0,
    emitters,
    speculars,
    receiver,
    emitterVels,
    receiverVel,
    freq,
    thetas=np.linspace(-180, 180, 721)[:-1] * u.deg,
    nedge=200,
    edgelim=0,
):
    nSource = emitters.shape[0]
    powers = np.zeros((nSource,thetas.shape[0]))
    asymm = np.zeros((nSource,thetas.shape[0]))

    dopplerPad,delayPad,ddmsPad = pad_DDM(ddms,doppler,delay,fd0,tau0)
    for i in range(thetas.shape[0]):
        # try:

        powers[:, i], asymm[:, i] = powerCalc(
            thetas[i],
            ddmsPad,
            dopplerPad,
            delayPad,
            fd0,
            tau0,
            emitters,
            speculars,
            receiver,
            emitterVels,
            receiverVel,
            freq,
            nedge=nedge,
            edgelim=edgelim,
        )
    return powers, asymm

def fitPeaks(
    emitters,
    speculars,
    receiver,
    receiverVel,
    power,
    thetas,
    dots,
    hfw=10,
):
    cent = thetas.mean()
    leftPower = power[:,thetas < cent]
    rightPower = power[:,thetas > cent]
    leftTheta = thetas[thetas < cent].value
    rightTheta = thetas[thetas > cent].value
    nSource = leftPower.shape[0]
    leftPeaks = np.zeros(nSource)
    rightPeaks = np.zeros(nSource)
    leftError = np.zeros(nSource)
    rightError = np.zeros(nSource)
    
    fitsLeft = np.zeros((nSource,3))
    fitsRight = np.zeros((nSource,3))
    for i in range(nSource):
        try:
            idx = np.argwhere(leftPower[i] == leftPower[i].max()).max()
            thetas_fit = leftTheta[max((idx - hfw, 0)) : min((idx + hfw + 1, leftPower.shape[1]))]
            powers_fit = leftPower[i,max((idx - hfw, 0)) : min((idx + hfw + 1, leftPower.shape[1]))]

            ## Initial Guesses
            C = powers_fit.max()
            x0 = thetas_fit[powers_fit == C][0]
            if x0 == thetas_fit[0]:
                A = (powers_fit[-1] - C) / ((thetas_fit[-1] - x0) ** 2)
            else:
                A = (powers_fit[0] - C) / ((thetas_fit[0] - x0) ** 2)

            ## Fit parabola around peak
            popt, pcov = THTH.curve_fit(
                THTH.chi_par, thetas_fit, powers_fit, p0=np.array([A, x0, C])
            )
            err = np.std(powers_fit - THTH.chi_par(thetas_fit, *popt)) * np.ones(
                thetas_fit.shape[0]
            )
            popt, pcov = THTH.curve_fit(
                THTH.chi_par,
                thetas_fit,
                powers_fit,
                p0=popt,
                sigma=err,
                absolute_sigma=True,
            )
            fitsLeft[i,:]=np.copy(popt)
            leftPeaks[i] = popt[1]
            leftError[i] = np.sqrt(
                (powers_fit - THTH.chi_par(thetas_fit, *popt)).std() / np.abs(popt[0])
            )
        except:
            leftError[i]=np.nan
            leftPeaks[i]=np.nan
            fitsLeft[i,:]=np.array([np.nan,np.nan,np.nan])
        try:
            idx = np.argwhere(rightPower[i] == rightPower[i].max()).max()
            thetas_fit = rightTheta[max((idx - hfw, 0)) : min((idx + hfw + 1, leftPower.shape[1]))]
            powers_fit = rightPower[i,max((idx - hfw, 0)) : min((idx + hfw + 1, leftPower.shape[1]))]

            ## Initial Guesses
            C = powers_fit.max()
            x0 = thetas_fit[powers_fit == C][0]
            if x0 == thetas_fit[0]:
                A = (powers_fit[-1] - C) / ((thetas_fit[-1] - x0) ** 2)
            else:
                A = (powers_fit[0] - C) / ((thetas_fit[0] - x0) ** 2)

            ## Fit parabola around peak
            popt, pcov = THTH.curve_fit(
                THTH.chi_par, thetas_fit, powers_fit, p0=np.array([A, x0, C])
            )
            err = np.std(powers_fit - THTH.chi_par(thetas_fit, *popt)) * np.ones(
                thetas_fit.shape[0]
            )
            popt, pcov = THTH.curve_fit(
                THTH.chi_par,
                thetas_fit,
                powers_fit,
                p0=popt,
                sigma=err,
                absolute_sigma=True,
            )
            fitsRight[i,:]=np.copy(popt)
            rightPeaks[i] = popt[1]
            rightError[i] = np.sqrt(
                (powers_fit - THTH.chi_par(thetas_fit, *popt)).std() / np.abs(popt[0])
            )
        except:
            rightPeaks[i] = np.nan
            rightError[i] = np.nan
            fitsRight[i,:]=np.array([np.nan,np.nan,np.nan])
    
    bothGood = np.isfinite(leftPeaks)*np.isfinite(rightPeaks)
    leftPeaks = leftPeaks[bothGood]
    rightPeaks = rightPeaks[bothGood]
    leftError = leftError[bothGood]
    rightError = rightError[bothGood]
    nFitted = leftPeaks.shape[0]
    if nFitted>1:
        leftMean = np.sum(leftPeaks / leftError**2) / np.sum(1 / leftError**2)
        rightMean = np.sum(rightPeaks / rightError**2) / np.sum(1 / rightError**2)
        leftErr = 1 / np.sqrt(np.sum(np.sum(1 / leftError**2)))
        rightErr = 1 / np.sqrt(np.sum(np.sum(1 / rightError**2)))
        leftChi = np.sum(((leftPeaks - leftMean) / leftError) ** 2) / (nFitted - 1)
        rightChi = np.sum(((rightPeaks - rightMean) / rightError) ** 2) / (nFitted - 1)
        leftStd = leftPeaks.std()
        rightStd = rightPeaks.std()
    else:
        leftMean=np.nan
        rightMean=np.nan
        leftChi=np.nan
        rightChi=np.nan
        leftErr=np.nan
        rightErr=np.nan
        leftStd=np.nan
        rightStd=np.nan
    fitRes = {'left' : {}, 'right' : {}}
    fitRes['left'].update({'mean' : leftMean})
    fitRes['left'].update({'err' : leftErr})
    fitRes['left'].update({'chi' : leftChi})
    fitRes['left'].update({'std' : leftStd})
    fitRes['left'].update({'fits' : fitsLeft})
    fitRes['right'].update({'mean' : rightMean})
    fitRes['right'].update({'err' : rightErr})
    fitRes['right'].update({'chi' : rightChi})
    fitRes['right'].update({'std' : rightStd})
    fitRes['right'].update({'fits' : fitsRight})
    if leftChi<rightChi:
        fitRes.update({'best' : 'left'})
        fitRes.update({'worst' : 'right'})
    else:
        fitRes.update({'best' : 'right'})
        fitRes.update({'worst' : 'left'})
    return (fitRes)

def err_string(val,err,unit):
    nE = 0
    eStr = f'%.{nE}e' %err
    ePow = int(eStr[3:])
    if eStr[2]=='-':
        ePow*=-1
    if eStr[0]=='1':
        nE+=1
    eStr = f'%.{nE}e' %err

    nV = 0
    vStr = f'%.{nV}e' %val
    if val<0:
        offset=1
    else:
        offset = 0
    vPow = int(vStr[2+offset:])
    # if vStr[2+offset]=='-':
    #     vPow*=-1
    nV = nE+(vPow-ePow)
    vNum = nV-vPow
    eNum = nE-ePow
    while (eNum<0 or vNum<0):
        eNum+=1
        vNum+=1
    vStr = f'%.{vNum}f' %val
    eStr = f'%.{eNum}f' %err
    return(vStr +f' $\pm$ '+eStr+unit)


    
def single_plot(ddms,
                emitters,
                speculars,
                receiver,
                emitterVels,
                receiverVel,
                delay,doppler,
                tau0,fd0,
                freq,
                fitRes,
                theta,
                power,fname=None,archival = None, archivalID = 0):
    nSource=ddms.shape[0]
    xAxis, yAxis, zAxis, psis, des, drs = coords(emitters, speculars, receiver)
    thetaWs = thetaConvert(fitRes[fitRes['best']]['mean']*u.deg, xAxis, yAxis, zAxis).to(u.deg)
    Ves = convertToSpecular(emitterVels, xAxis, yAxis, zAxis)
    Vrs = convertToSpecular(receiverVel, xAxis, yAxis, zAxis)
    thetaIs = wavesToImages(thetaWs, psis)
    etas = calcCurvatures(thetaIs, psis, des, drs, Vrs, Ves, freq)

    nRows=3
    if archival:
        nRows+=4
    grid = plt.GridSpec(nrows=nRows, ncols=nSource)
    plt.figure(figsize=(2*nSource, nRows*2))
    c = ["tab:blue", "tab:gray", "tab:purple", "tab:brown"]
    for i in range(nSource):
        extent = THTH.ext_find(doppler, delay)
        info=r"$\theta_w=$"+ "%.2f" % thetaWs[i].to_value(u.deg)+ f"\n"+ r"$\theta_i=$"+ "%.2f" % thetaIs[i].to_value(u.deg)+ f"\n"+ r"$\psi=$"+ "%.2f" % psis[i].to_value(u.deg)+ f"\n"+ r"$\eta=$"+ "%.2e" % etas[i].value+ r"$\rm{s}^3$"
        plt.subplot(grid[0, i])
        plt.imshow(ddms[i], origin="lower", aspect="auto", extent=extent, cmap="magma")
        plt.plot(
            np.linspace(doppler.min(),doppler.max(),1000),
            (etas[i] * (np.linspace(doppler.min(),doppler.max(),1000) - fd0[i]) ** 2).to(delay.unit) + tau0[i],
            c=colors.TABLEAU_COLORS[c[i]],
        )
        plt.xlim(extent[:2])
        plt.ylim(extent[2:])
        plt.xlabel(r"$f_D~\left(\rm{Hz}\right)$")
        if i == 0:
            plt.ylabel(r"$\tau~\left(\mu\rm{s}\right)$")
        else:
            plt.yticks([])
        plt.title(info)
        plt.subplot(grid[1, i])
        plt.plot(np.linspace(-2,2,2)*np.cos(thetaWs[i]),np.linspace(-2,2,2)*np.sin(thetaWs[i]),label='Wave Direction')
        plt.plot(np.linspace(-2,2,2)*np.cos(thetaIs[i]),np.linspace(-2,2,2)*np.sin(thetaIs[i]),label='Image Direction')
        plt.xlim((-1,1))
        plt.ylim((-1,1))
        plt.xlabel("X")
        if i == 0:
            plt.ylabel("Y")
            plt.legend()
        plt.yticks([])
        plt.xticks([])
            
        plt.subplot(grid[2, :])
        plt.plot(
            theta,
            power[i],
            # power[:,i],
            c=colors.TABLEAU_COLORS[c[i]],
        )
    plt.ylim((0.8, 1.01))
    plt.yticks([0.8, 0.85, 0.9, 0.95, 1.0])
    # plt.ylim((0.0,1.0))
    plt.ylabel("Relative Power")
    plt.xlabel("Wind Direction (deg)")
    plt.axvspan(
        fitRes[fitRes['best']]['mean'] -  fitRes[fitRes['best']]['err']*fitRes[fitRes['best']]['chi'],
        fitRes[fitRes['best']]['mean'] +  fitRes[fitRes['best']]['err']*fitRes[fitRes['best']]['chi'],
        alpha=0.5,
        color="green",label=err_string(fitRes[fitRes['best']]['mean'],fitRes[fitRes['best']]['err']*fitRes[fitRes['best']]['chi'],r'$^\circ$')
    )
    plt.axvspan(
        fitRes[fitRes['worst']]['mean'] -  fitRes[fitRes['worst']]['err']*fitRes[fitRes['worst']]['chi'],
        fitRes[fitRes['worst']]['mean'] +  fitRes[fitRes['worst']]['err']*fitRes[fitRes['worst']]['chi'],
        alpha=0.5,
        color="red",label=err_string(fitRes[fitRes['worst']]['mean'],fitRes[fitRes['worst']]['err']*fitRes[fitRes['worst']]['chi'],r'$^\circ$')
    )
    plt.legend()
    
    if archival:

        speed = np.sqrt(np.array(archival.u10**2+archival.v10**2))[archivalID][::-1,:]
        lat = np.array(archival.latitude)[::-1]*u.deg
        lon = np.array(archival.longitude)*u.deg
        angle = np.arctan2(np.array(archival.u10),np.array(archival.v10))[archivalID][::-1,:]

        oLoc = EarthLocation(x=receiver[0],y=receiver[1],z=receiver[2])
        sLoc = EarthLocation(x=speculars.T[0],y=speculars.T[1],z=speculars.T[2])
        eLoc = EarthLocation(x=emitters.T[0],y=emitters.T[1],z=emitters.T[2])

        cent = np.mod(sLoc.lon.to(u.deg).mean(),360*u.deg)

        lon2 = lon-180*u.deg-cent
        lon2 = np.mod(lon2+180*u.deg,360*u.deg)-180*u.deg
        speed2=np.copy(speed)
        speed2=speed2[:,np.argsort(lon2)]
        angle2=np.copy(angle)
        angle2=angle2[:,np.argsort(lon2)]
        lon2=np.sort(lon2)

        ax1 = plt.subplot(grid[3:5, :])
        im1=ax1.imshow(np.roll(speed2,720,axis=1),origin='lower',aspect='auto',extent=THTH.ext_find(lon2,lat),
                cmap='Greys')
        shifted_world(cent.value).plot(ax=ax1, color="white")
        ax1.set_xlim((-180,180))
        ax1.set_ylim((-90,90))
        for n in range(nSource):
            ax1.plot(np.mod(sLoc[n].lon-cent+180*u.deg,360*u.deg)-180*u.deg,sLoc[n].lat,'*',c=colors.TABLEAU_COLORS[c[n]])
            ax1.plot(np.mod(eLoc[n].lon-cent+180*u.deg,360*u.deg)-180*u.deg,eLoc[n].lat,'^',c=colors.TABLEAU_COLORS[c[n]])
        ax1.plot(np.mod(oLoc.lon-cent+180*u.deg,360*u.deg)-180*u.deg,oLoc.lat,'r.',label='Receiver')
        ax1.set_title(r'$\rm{Wind~Speed}~\left(\rm{m}~\rm{s}^{-1}\right)$')
        ax1.set_xticks([])
        ax1.set_ylabel(r'$\rm{Latitude}~\left(\rm{deg}\right)$')
        plt.colorbar(im1,ax=ax1)
        ax1.plot(np.array([]),np.array([]),'k*',label='Specular Point')
        ax1.plot(np.array([]),np.array([]),'k^',label='CYGNSS Satelite')
        ax1.legend()

        ax2 = plt.subplot(grid[5:, :])
        im2=ax2.imshow(np.roll(angle2,720,axis=1),origin='lower',aspect='auto',extent=THTH.ext_find(lon2,lat),
                cmap='twilight')
        shifted_world(cent.value).plot(ax=ax2, color="white")
        ax2.set_xlim((-180,180))
        ax2.set_ylim((-90,90))
        for n in range(nSource):
            ax2.plot(np.mod(sLoc[n].lon-cent+180*u.deg,360*u.deg)-180*u.deg,sLoc[n].lat,'*',c=colors.TABLEAU_COLORS[c[n]])
            ax2.plot(np.mod(eLoc[n].lon-cent+180*u.deg,360*u.deg)-180*u.deg,eLoc[n].lat,'^',c=colors.TABLEAU_COLORS[c[n]])
        ax2.plot(np.mod(oLoc.lon-cent+180*u.deg,360*u.deg)-180*u.deg,oLoc.lat,'r.')

        ax2.set_xticklabels(np.mod(ax2.get_xticks()+cent.value+180,360).astype(int)-180)
        ax2.set_ylabel(r'$\rm{Latitude}~\left(\rm{deg}\right)$')
        ax2.set_xlabel(r'$\rm{Longitude}~\left(\rm{deg}\right)$')
        ax2.set_title(r'$\rm{Wind~Direction}~\left(\rm{deg}\right)$')
        plt.colorbar(im2,ax=ax2)

    plt.tight_layout()
    if fname:
        plt.savefig(fname)


def shifted_world(shift):
    countries = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    shift -= 180
    movedGeometry = []
    splittedGeometry = []
    border = LineString([(shift,90),(shift,-90)])
    
    for row in countries["geometry"]:
        splittedGeometry.append(split(row, border))
    for element in splittedGeometry:
        items = list(element)
        for item in items:
            minx, miny, maxx, maxy = item.bounds
            if minx >= shift:
                movedGeometry.append(translate(item, xoff=-180-shift))
            else:
                movedGeometry.append(translate(item, xoff=180-shift))
    
    # got `moved_geom` as the moved geometry            
    return(geopandas.GeoDataFrame({"geometry": movedGeometry}))
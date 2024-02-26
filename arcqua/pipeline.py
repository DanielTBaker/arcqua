import cdsapi
import os
from astropy.time import Time
import astropy.units as u
## Data Loaders
import xarray as xr

def download_archival_velocities(day,month,year,outputDir='.'):
    c = cdsapi.Client()
    tStart=Time(f'{year}-{month}-{day}T00:00:00.000')
    fname=f'{tStart.value[:4]}{tStart.value[5:7]}{tStart.value[8:10]}.nc'
    fname=os.path.join(outputDir,fname)
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'mean_direction_of_total_swell',
                'mean_direction_of_wind_waves',
                'mean_wave_direction',
            ],
            'year': f'{year}',
            'month': f'{month}',
            'day': [f'{day}'],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'format': 'netcdf',
        },
        fname)
    
def download_CYGNNS_data(day,month,year,outputDir='.'):
    tStart=Time(f'{year}-{month}-{day}T00:00:00.000')
    tEnd=tStart+1*u.day
    outputDir = os.path.join(outputDir,'data')
    prefix = f'podaac-data-downloader -c CYGNSS_L1_V3.1 -d {outputDir} --start-date '
    cmd = prefix + tStart.value[:-4]+'Z --end-date '+tEnd.value[:-4]+'Z -e \"\"'
    os.system(cmd)

def full_download(day,month,year,outputDir='.'):
    cygnssDir = os.path.join(outputDir,'CYGNSS')
    os.makedirs(cygnssDir,exist_ok=True)
    archivalDir = os.path.join(outputDir,'archival')
    os.makedirs(archivalDir,exist_ok=True)
    download_CYGNNS_data(day,month,year,outputDir=cygnssDir)
    download_archival_velocities(day,month,year,outputDir=archivalDir)

def repack_data(date,satNum=1,sourceDir='.',outputDir='.',source='CYGNSS'):
    if source == 'CYGNSS':
        fname=f"cyg0{satNum}.ddmi.s{date}-000000-e{date}-235959.l1.power-brcs.a31.d32.nc"
        data = xr.open_dataset(os.path.join(sourceDir,fname))

        startTime = Time(data.time_coverage_start)
        brcs = np.array(data.brcs)
        time = Time(np.array(data.ddm_timestamp_utc))
        gpsCode = np.array(data.prn_code).astype(int)
        cygNumber = int(data.spacecraft_num)

        ##Get Errors
        errorCodes = np.array(data.quality_flags)
        isError = errorCodes > 0

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
        

        sourcePos = np.transpose(np.array([sourceX, sourceY, sourceZ]), (1, 2, 0))*u.m
        specularPos = np.transpose(np.array([specularX, specularY, specularZ]), (1, 2, 0))*u.m
        observerPos = np.array([observerX, observerY, observerZ]).T*u.m

        sourceVel = np.transpose(np.array([sourceVX, sourceVY, sourceVZ]), (1, 2, 0))*u.m/u.s
        specularVel = np.transpose(np.array([specularVX, specularVY, specularVZ]), (1, 2, 0))*u.m/u.s
        observerVel = np.array([observerVX, observerVY, observerVZ]).T*u.m/u.s

        ## Delay and Doppler
        nDoppler = data.dims['doppler']
        nDelay = data.dims['delay']
        nDDM = data.dims['ddm']
        chipSize = ((1./1023000.)*u.s).to(u.us)
        delay = np.linspace(-(nDelay-1)//2,(nDelay-1)//2,nDelay)*float(data.delay_resolution)*chipSize
        doppler = np.linspace(-(nDoppler-1)//2,(nDoppler-1)//2,nDoppler)*float(data.dopp_resolution)*u.Hz
        tau0 = np.array(data.brcs_ddm_sp_bin_delay_row)*float(data.delay_resolution)*chipSize+delay[0]
        fd0 = np.array(data.brcs_ddm_sp_bin_dopp_col)*float(data.dopp_resolution)*u.Hz+doppler[0]

        timeString = startTime.value
        year=timeString.split('-')[0]
        month=timeString.split('-')[1]
        day=timeString.split('-')[2].split('T')[0]
        date=year+month+day
    for nSource in np.array([2, 3, 4]):
        hasCount = np.argwhere(np.sum(isError, 1) == (nDDM - nSource))[:, 0]

        tempIDs = errorCodes[hasCount] == 0
        tempGPS = np.reshape(gpsCode[hasCount][tempIDs], (-1, nSource))
        tempTime = time[hasCount]
        tempSourcePos = np.reshape(sourcePos[hasCount][tempIDs], (-1, nSource, 3))
        tempSpecularPos = np.reshape(specularPos[hasCount][tempIDs], (-1, nSource, 3))
        tempObserverPos = observerPos[hasCount]
        tempSourceVel = np.reshape(sourceVel[hasCount][tempIDs], (-1, nSource, 3))
        tempSpecularVel = np.reshape(specularVel[hasCount][tempIDs], (-1, nSource, 3))
        tempObserverVel = observerVel[hasCount]
        tempDDMs = np.reshape(brcs[hasCount][tempIDs], (-1, nSource, nDelay, nDoppler))
        tempTau0 = np.reshape(tau0[hasCount][tempIDs],(-1, nSource))
        tempFd0 = np.reshape(fd0[hasCount][tempIDs],(-1, nSource))

        fname = f"{date}-cyg0{cygNumber}-{nSource}source.npz"
        np.savez(
            os.path.join(outputDir,fname),
            time=tempTime.mjd,
            ddm=tempDDMs,
            delay=delay,
            doppler=doppler,
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


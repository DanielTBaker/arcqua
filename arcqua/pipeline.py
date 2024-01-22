import cdsapi
import os
from astropy.time import Time
import astropy.units as u

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

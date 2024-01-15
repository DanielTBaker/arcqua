import cdsapi
import os
from astropy.time import Time
import astropy.units as u

def download_archival_velocities(day,month,year,fname='download.nc'):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '10m_u_component_of_neutral_wind', '10m_u_component_of_wind', '10m_v_component_of_neutral_wind',
                '10m_v_component_of_wind',
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
    
def download_CYGNNS_data(day,month,year):
    if month<10:
        month='0'+str(month)
    if day<10:
        day='0'+str(day)
    tStart=Time(f'{year}-{month}-{day}T00:00:00.000')
    tEnd=tStart+1*u.day
    currentDir = os.getcwd()
    outputDir = os.path.join(currentDir,'data')
    prefix = f'podaac-data-downloader -c CYGNSS_L1_V3.1 -d {outputDir} --start-date '
    cmd = prefix + tStart.value[:-4]+'Z --end-date '+tEnd.value[:-4]+'Z -e \"\"'
    os.system(cmd)
# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
import scipy.spatial
import matplotlib.pyplot as plt
import geopandas

from skyfield.api import EarthSatellite, Star
from skyfield.api import load, utc, wgs84
from skyfield.data import hipparcos
from skyfield.positionlib import ICRF


class SatelliteView:
    '''
    Class to simulate the view of Earth, the solar system, and the stars from
    an Earth satellite's position pointing at a given target at a given time.
    '''
    def __init__(self,
                 satellite: EarthSatellite,
                 points_at,
                 utc_time: datetime = datetime.utcnow(),
                 ephemeris = None,
                 plot_radec: bool = False) -> None:

        # Load arguments
        self.eph = ephemeris if ephemeris else load('de440.bsp')
        self.earth = self.eph['earth']
        self.satellite = satellite
        self.observer = satellite + self.earth
        self.utc_time = utc_time
        self.plot_radec = plot_radec

        # Initiate data dictionary
        self.data = {}

        # Check if target is geocentric object
        if points_at.center == 399:
            self.points_at = points_at + self.earth
        else:
            self.points_at = points_at


    @property
    def t(self):
        # Set utc time scale
        ts = load.timescale()
        return ts.from_datetime(self.utc_time.replace(tzinfo=utc))


    @property
    def radec0(self):
        # Get right ascension and declination of pointing target
        ra0, dec0, _ = self.observer.at(self.t).observe(self.points_at).apparent().radec()
        return ra0, dec0


    def angle_from(self,position):
        '''
        Convert target position to observation angle (or right ascension/declination)
        in degrees.
        '''
        ra, dec, _ = position.radec()
        if self.plot_radec:
            return np.array([ra._degrees, dec.degrees])
        else:
            xangle_deg = np.mod(-ra._degrees+self.radec0[0]._degrees+180, 360)-180
            yangle_deg = np.mod(dec.degrees-self.radec0[1].degrees+90,180)-90
            return np.array([xangle_deg, yangle_deg])


    def angle_celestial(self,target):
        '''
        Calculate observation angle (or right ascension/declination) of a
        celestial target in degrees.
        '''
        pos = self.observer.at(self.t).observe(target).apparent()

        return self.angle_from(pos)


    def approx_angle_latlon(self, latitude, longitude):
        '''
        Calculate the approx. observation angle (or right ascension/declination)
        of latitude and longitude on Earth in degrees.
        '''
        # Get relative position between geographic position and observer
        sat_xyz = self.satellite.at(self.t).xyz.au
        target_xyz = wgs84.latlon(latitude, longitude).at(self.t).xyz.au
        pos = ICRF((target_xyz.T - sat_xyz).T)

        # Calculate observation angle
        pos_angle = self.angle_from(pos)

        # Remove locations behind earth
        Re = wgs84.radius.km
        behind_earth = pos.distance().km**2 > self.km_to(self.earth)**2 - Re**2
        for ii in range(2):
            pos_angle[ii][behind_earth] = np.nan

        # Check if locations are sunlit
        sun = self.eph['sun']
        sunlit = np.dot(target_xyz.T, (sun-self.earth).at(self.t).xyz.au ) < 0

        return pos_angle, sunlit


    def km_to(self,target,observer=None) -> float:
        '''
        Distance in km from satellite observer to a target.
        '''
        if observer == None:
            observer = self.observer
        return (target-observer).at(self.t).distance().km


    def load_stars(self,
                   catalog_url: str = hipparcos.URL,
                   min_magnitude: float = 7):
        '''
        Load star catalog and filter bright stars based on a minimum magnitude.
        '''
        # Load stars as dataframe
        with load.open(catalog_url) as f:
            self.df_stars = hipparcos.load_dataframe(f)
        
        # Filter bright stars
        self.df_stars = self.df_stars[(self.df_stars.magnitude < min_magnitude)]
        return self.df_stars


    def plot_stars(self, max_star_size: int = 150) -> None:
        '''
        Plot stars as a function of observation angle.
        The marker sizes are set by the magnitudes of the stars.
        '''
        # Load star catalog
        if not hasattr(self,'df_stars'):
            self.load_stars()
        
        # Get observation angle from satellite of stars
        stars = Star.from_dataframe(self.df_stars)

        # Add to dictionary
        self.data['stars'] = {}
        self.data['stars']['angle'] = self.angle_celestial(stars)
        self.data['stars']['magnitude'] = self.df_stars.magnitude
        
        # Set marker size for stars
        marker_size = max_star_size*10**(self.data['stars']['magnitude']/-2.5)
        plt.scatter(*self.data['stars']['angle'], s=marker_size,
                    color='w', marker='$✴$', linewidths=0, label='Stars')


    def plot_solarsystem(self) -> None:
        '''
        Plot solar system function of observation angle.
        '''
        # Define markers
        self.data['solarsystem'] = {
            'Moon':{'marker':'$☽$','radius_km': 1737.4},
            'Sun':{'marker':'$☉$','radius_km': 696340},
            'Mercury':{'marker':'$☿$'},
            'Venus':{'marker':'$♀$'},
            'Mars':{'marker':'$♂$'},
            'Jupiter':{'marker':'$♃$'},
            'Saturn':{'marker':"$♄$"},
            'Uranus':{'marker':'$⛢$'},
            'Neptune':{'marker':'$♆$'}
            }

        # Loop through solar system objects
        ii = 0
        for name, info in self.data['solarsystem'].items():
            # Get observation angle
            if name == 'Moon' or name == 'Sun':
                obj = self.eph[f'{name}']
            else:
                obj = self.eph[f'{name}_barycenter']
            self.data['solarsystem'][name]['angle'] = self.angle_celestial(obj)
            

        for name, info in self.data['solarsystem'].items():
            # Plot sun/moon as marker
            plt.plot(*info['angle'], marker=info['marker'], ls='', ms=12, label=name)

            if 'radius_km' in info:
                # Calculate angular radius and add circle
                circle_r = np.rad2deg(np.arctan(info['radius_km']/self.km_to(obj)))
                circle = plt.Circle(info['angle'], circle_r)
                plt.gca().add_patch(circle)
            ii += 1

        
    def load_coast(self,
                   URL: str = "http://d2ad6b4ur7yvpq.cloudfront.net" +
                   "/naturalearth-3.3.0/ne_110m_land.geojson") -> None:
        '''
        Load coastlines and extract latitudes and longitudes.
        '''
        # Load coastlines
        df_coasts = geopandas.read_file(URL)

        # Get coastline coordinates and combine
        self.coast_lon = []
        self.coast_lat = []
        for ii, row in df_coasts.iterrows():
            self.coast_lon.extend(row.geometry.exterior.xy[0])
            self.coast_lat.extend(row.geometry.exterior.xy[1])
            self.coast_lon.append(np.nan)
            self.coast_lat.append(np.nan)
        
        return self.coast_lat, self.coast_lat


    def plot_earth(self) -> None:
        '''
        Plot earth as a function of observation angle.
        '''
        # Define colors for earth polygons
        EARTH_COLORS = ['#040404','#203080']

        # Add to dictionary
        self.data['earth'] = {}
        self.data['earth']['grid'] = {}
        self.data['earth']['poly'] = {}
        self.data['earth']['coast'] = {}
        
        # Create a fine grid on Earth
        earth_grid = np.meshgrid(np.arange(-90,90),np.arange(-180,180))

        # Loop for latitudes and longitudes
        self.data['earth']['grid']['angle'] = []
        for kk in range(2):
            grid_angle, sunlit = self.approx_angle_latlon(earth_grid[kk].flatten(),
                                                          earth_grid[1-kk].flatten())
            self.data['earth']['grid']['angle'].extend(grid_angle[:,earth_grid[1].flatten()%10 == 0])

        # Plot
        plt.plot(*self.data['earth']['grid']['angle'],'-',color='gray',alpha=0.25)

        # Check if subpoint is sunlit
        subpoint_latlon = wgs84.latlon_of(self.earth.at(self.t).observe(self.observer))
        _ , subpoint_sunlit = self.approx_angle_latlon([subpoint_latlon[0].degrees],
                                                       [subpoint_latlon[1].degrees])

        # Create polygons
        is_earth = ~np.isnan(grid_angle[0,:])
        is_overlay = is_earth * (subpoint_sunlit == sunlit)
        earth_poly = scipy.spatial.ConvexHull(grid_angle[:,is_earth].T)
        overlay_poly = scipy.spatial.ConvexHull(grid_angle[:,is_overlay].T)

        # Plot polygons of earth and overlay (sunlit/shade)
        if subpoint_sunlit:
            EARTH_COLORS = EARTH_COLORS[::-1]
        
        self.data['earth']['poly']['color'] = EARTH_COLORS
        self.data['earth']['poly']['angle'] = [earth_poly.points[earth_poly.vertices,:].T,
                                               overlay_poly.points[overlay_poly.vertices,:].T]
        
        for poly, col in zip(self.data['earth']['poly']['angle'],
                             self.data['earth']['poly']['color']):
            plt.fill(*poly,color=col)


        # Load coastlines and get angles
        if not hasattr(self,'coast_lat'):
            self.load_coast()
        self.data['earth']['coast'] = {}
        self.data['earth']['coast']['angle'], _ = self.approx_angle_latlon(self.coast_lat,
                                                                  self.coast_lon)

        # Plot coastlines
        plt.plot(*self.data['earth']['coast']['angle'],'-',color='gray')
        

    def plot_all(self) -> None:
        '''
        Plot all celestial objects as a function of observation angle.
        '''
        self.plot_stars()
        self.plot_solarsystem()
        self.plot_earth()

        # Plot settings
        plt.axis('equal')
        plt.xlabel('Horizontal observation angle (°)')
        plt.ylabel('Vertical observation angle (°)')
        plt.title(f"{self.satellite.name}, {self.t.utc_strftime()}") 

        # Legend
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0, box.width * 0.9, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))


if __name__ == "__main__":
    # Time in UTC
    utc_time = datetime(2025, 2, 4, 19, 0)

    # Satellite name and two line element
    satellite_name = 'METEOSAT 12 (MTG I1)'
    satellite_tle = '''1 54743U 22170C   25021.33313354  .00000011  00000-0  00000-0 0  9992
2 54743   0.5918  35.3149 0001811 255.4605 309.9016  1.00269375  7841
'''

    # Load savellite from TLE
    sat = EarthSatellite(*satellite_tle.splitlines(),
                         satellite_name)

    # Create view instance
    view = SatelliteView(satellite=sat,
                         points_at=wgs84.latlon(0,0),
                         utc_time=utc_time)
##    view.load_stars('hip_main.dat')
##    view.load_coast('ne_110m_land.geojson')

    # Plot
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12,8))
    view.plot_all()
    plt.xlim(-20,20)
    plt.ylim(-12,12)

    plt.show()

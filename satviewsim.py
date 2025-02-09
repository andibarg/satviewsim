# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
import scipy.spatial
import matplotlib.pyplot as plt

from skyfield.api import EarthSatellite, Star
from skyfield.api import load, utc, wgs84
from skyfield.data import hipparcos


class SatelliteView:
    '''
    Class to simulate the view of Earth, the solar system, and the stars from
    an Earth satellite's position pointing at a given target at a given time.
    '''
    def __init__(self,
                 satellite: EarthSatellite,
                 points_at,
                 utc_time: datetime = datetime.utcnow(),
                 ephemeris = load('de440.bsp'),
                 plot_radec: bool = False) -> None:
        self.eph = ephemeris
        self.earth = self.eph['earth']
        self.satellite = satellite
        self.observer = satellite + self.earth
        self.utc_time = utc_time
        self.plot_radec = plot_radec

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
            return ra._degrees, dec.degrees
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


    def approx_angle_latlon(self, latitude, longitude,
                            remove_behind_earth=True):
        '''
        Calculate the approx. observation angle (or right ascension/declination)
        of latitude and longitude on Earth in degrees.
        '''
        target = (wgs84.latlon(latitude, longitude)+self.earth)
        pos = (target-self.observer).at(self.t) # Currently requires small modification of skyfield code to enable broadcasting!
        pos_angle = self.angle_from(pos)

        # Remove locations behind earth
        Re = wgs84.radius.km
        if remove_behind_earth:
            behind_earth = self.km_to(target)**2 > self.km_to(self.earth)**2 - Re**2
            for ii in range(2):
                pos_angle[ii][behind_earth] = np.nan

        # Check if locations are sunlit
        sun = self.eph['sun']
        sunlit = self.km_to(self.earth,sun)**2-Re**2 < self.km_to(target,sun)**2 

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


    def plot_stars(self,
                   max_star_size=100,
                   *args,
                   **kwargs) -> None:
        '''
        Plot stars as a function of observation angle.
        The marker sizes are set by the magnitudes of the stars.
        '''
        # Load star catalog
        if not hasattr(self,'df_stars'):
            self.load_stars(*args, **kwargs)
        
        # Get observation angle from satellite of stars
        stars = Star.from_dataframe(self.df_stars)
        stars_angle = self.angle_celestial(stars)  
        
        # Set marker size for stars
        marker_size = max_star_size*10**(self.df_stars.magnitude/-2.5)

        plt.scatter(*stars_angle, s=marker_size,
                    color='w', marker='$✴$', linewidths=0, label='Stars')
        self.plot_configs()


    def plot_sun_moon(self) -> None:
        '''
        Plot sun and moon as a function of observation angle.
        '''
        # Define markers
        MOON_SUN = {'Moon':{'marker':'$☽$','radius_km': 1737.4},
                    'Sun':{'marker':'$☉$','radius_km': 696340}}

        # Loop through moon and sun
        ii = 0
        for name, info in MOON_SUN.items():
            # Get observation angle
            obj = self.eph[f'{name}']
            obj_angle = self.angle_celestial(obj)

            # Plot sun/moon as marker
            plt.plot(*obj_angle, marker=info['marker'], ls='', ms=12, label=name)

            # Calculate angular radius and add circle
            circle_r = np.rad2deg(np.arctan(info['radius_km']/self.km_to(obj)))
            circle = plt.Circle(obj_angle, circle_r, color=f'C{ii:02d}')
            plt.gca().add_patch(circle)
            ii += 1
        self.plot_configs()


    def plot_planets(self) -> None:
        '''
        Plot planets as a function of observation angle.
        '''
        # Define markers
        PLANETS = {'Mercury':'$☿$',
                   'Venus':'$♀$',
                   'Mars':'$♂$',
                   'Jupiter':'$♃$',
                   'Saturn':"$♄$",
                   'Uranus':'$⛢$',
                   'Neptune':'$♆$'}

        # Loop through all planets
        for name, marker in PLANETS.items():
            # Get observation angle
            planet = self.eph[f'{name}_barycenter']
            planet_angle = self.angle_celestial(planet)

            # Plot planet as marker
            plt.plot(*planet_angle, marker=marker, ls='', ms=12, label=name)
        self.plot_configs()
        

    def plot_earth(self) -> None:
        '''
        Plot earth as a function of observation angle.
        '''
        # Define colors for earth polygons
        EARTH_COLORS = ['#040404','C04']
        
        # Create a fine grid on Earth
        earth_grid = np.meshgrid(np.arange(-90,90),np.arange(-180,180))

        # Loop for latitudes and longitudes
        for kk in range(2):
            grid_angle, sunlit = self.approx_angle_latlon(earth_grid[kk].flatten(),
                                                          earth_grid[1-kk].flatten())

            # Plot
            plt.plot(*grid_angle[:,earth_grid[1].flatten()%10 == 0],
                     '-',color='gray',alpha=0.25)

        # Check if subpoint is sunlit
        subpoint_latlon = wgs84.latlon_of(view.earth.at(view.t).observe(view.observer))
        _ , subpoint_sunlit = view.approx_angle_latlon(subpoint_latlon[0].degrees,
                                                       subpoint_latlon[1].degrees,False)

        # Create polygons
        is_earth = ~np.isnan(grid_angle[0,:])
        is_overlay = is_earth * (subpoint_sunlit == sunlit)
        earth_poly = scipy.spatial.ConvexHull(grid_angle[:,is_earth].T)
        overlay_poly = scipy.spatial.ConvexHull(grid_angle[:,is_overlay].T)

        # Plot polygons
        if subpoint_sunlit:
            EARTH_COLORS = EARTH_COLORS[::-1]
        plt.fill(earth_poly.points[earth_poly.vertices,0],
                 earth_poly.points[earth_poly.vertices,1],
                 color=EARTH_COLORS[0])
        plt.fill(overlay_poly.points[overlay_poly.vertices,0],
                 overlay_poly.points[overlay_poly.vertices,1],
                 color=EARTH_COLORS[1])


    def plot_configs(self) -> None:
        '''
        Set common plot configurations
        '''
        plt.axis('equal')
        plt.xlabel('Horizontal observation angle (°)')
        plt.ylabel('Vertical observation angle (°)')
        plt.xlim(-180,180)
        plt.title(f"{self.satellite.name}, {self.t.utc_strftime()}") 
        

    def plot_all(self,
                 star_catalog_url: str = hipparcos.URL) -> None:
        '''
        Plot all celestial objects as a function of observation angle.
        '''
        self.plot_stars(catalog_url=star_catalog_url)
        self.plot_sun_moon()
        self.plot_planets()
        self.plot_earth()


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
                         points_at=wgs84.latlon(0,0))
    view.utc_time = utc_time



##    earth_grid = np.meshgrid(np.arange(-90,90),np.arange(-180,180))
##    grid_angle, sunlit = view.approx_angle_latlon(earth_grid[0].flatten(),earth_grid[1].flatten())
##
##    # Check if subpoint is sunlit
##    subpoint_latlon = wgs84.latlon_of(view.earth.at(view.t).observe(view.observer))
##    _ , sp_sunlit = view.approx_angle_latlon(subpoint_latlon[0].degrees,
##                                             subpoint_latlon[1].degrees,False)
##
##    # Create polygons
##    is_nan = ~np.isnan(grid_angle[0,:])
##    earth_poly = scipy.spatial.ConvexHull(grid_angle[:,is_nan].T)
##    
##
##    plt.fill(earth_poly.points[earth_poly.vertices,0],
##            earth_poly.points[earth_poly.vertices,1])

    

    # Plot
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12,8))
    view.plot_all()

    # Legend
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.9, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))

    plt.show()

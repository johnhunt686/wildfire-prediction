import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box

# Generic data URL or file path - replace with your data source
DATA_URL = 'sample_data.csv'  # Replace with actual URL or local path

STATES_DATA_URL = ("https://www2.census.gov/geo/tiger/TIGER2020/STATE/"
                   "tl_2020_us_state.zip")

EXCLUDED_STATES_ABBR = ['AK', 'HI', 'PR', 'VI']

CRS_LAT_LON = "EPSG:4326"  # WGS 84 geographic CRS (lat/lon)
CRS_ALBERS_EQUAL_AREA = "EPSG:5070"  # Projected CRS in meters

CONUS_BOUNDS_MIN_X = -2500000
CONUS_BOUNDS_MIN_Y = 100000
CONUS_BOUNDS_MAX_X = 2500000
CONUS_BOUNDS_MAX_Y = 3200000

GRID_SIZE_MILES = 50
HEATMAP_GRID_SIZE = 80500  # ~50 miles in meters.

# Load generic data - assume CSV with columns 'lat', 'lon', and optionally 'value'
df_raw = pd.read_csv(DATA_URL)
df = df_raw[~df_raw['st'].isin(EXCLUDED_STATES_ABBR)].copy() if 'st' in df_raw.columns else df_raw.copy()

# Create geometry from lat/lon
geometry = gpd.points_from_xy(df['lon'], df['lat'], crs=CRS_LAT_LON)
gdf = gpd.GeoDataFrame(df, geometry=geometry).to_crs(CRS_ALBERS_EQUAL_AREA)

# Load states data
states_gdf = gpd.read_file(STATES_DATA_URL)
states_gdf = states_gdf[~states_gdf['STUSPS'].isin(EXCLUDED_STATES_ABBR)].copy()
states_gdf = states_gdf.to_crs(CRS_ALBERS_EQUAL_AREA)

# Clip to CONUS bounds
conus_bounds_box = box(CONUS_BOUNDS_MIN_X, CONUS_BOUNDS_MIN_Y,
                       CONUS_BOUNDS_MAX_X, CONUS_BOUNDS_MAX_Y)

clipped_states = gpd.clip(states_gdf, conus_bounds_box)

gdf = gdf[gdf.geometry.within(conus_bounds_box)].copy()

# Create grid bins
x_bins = np.arange(CONUS_BOUNDS_MIN_X, CONUS_BOUNDS_MAX_X + 
                   HEATMAP_GRID_SIZE, HEATMAP_GRID_SIZE)
y_bins = np.arange(CONUS_BOUNDS_MIN_Y, CONUS_BOUNDS_MAX_Y + 
                   HEATMAP_GRID_SIZE, HEATMAP_GRID_SIZE)

# Create heatmap - use weights if 'value' column exists, else count
if 'value' in gdf.columns:
    heatmap, x_edges, y_edges = np.histogram2d(gdf.geometry.x, 
                                               gdf.geometry.y, 
                                               bins=[x_bins, y_bins],
                                               weights=gdf['value'])
    value_label = 'Value Sum per Grid Cell'
else:
    heatmap, x_edges, y_edges = np.histogram2d(gdf.geometry.x, 
                                               gdf.geometry.y, 
                                               bins=[x_bins, y_bins])
    value_label = 'Count per Grid Cell'

cmap = plt.cm.hot
norm = None

fig, ax = plt.subplots(figsize=(15, 12))
clipped_states.plot(ax=ax, color='none', 
                    edgecolor='white', linewidth=1)
vmax = np.max(heatmap)
img = ax.imshow(heatmap.T,
                extent=[x_edges[0], x_edges[-1], 
                        y_edges[0], y_edges[-1]],
                origin='lower', cmap=cmap, 
                norm=norm, alpha=1.0,                
                vmin=0, vmax=vmax)
ax.set_title('Generic Data Heatmap on US Map', fontsize=22)
ax.set_xlabel('')
ax.set_ylabel('')
ax.axis('off')

ticks = np.linspace(0, vmax, 6, dtype=int)
cbar = plt.colorbar(img, ax=ax, shrink=0.6, ticks=ticks)
cbar.set_label(f'\n{value_label}', fontsize=15)
cbar.ax.set_yticklabels(list(map(str, ticks)))

#plt.show()
plt.savefig('heatmap.png')

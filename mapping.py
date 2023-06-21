import os
import bpnsdata
import pandas as pd 
import numpy as np
import geopandas
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import joblib
from sklearndf.transformation import (
    OneHotEncoderDF
)
import xarray
from celluloid import Camera
from tqdm import tqdm


def create_time_space_df(start_time, end_time, borders_df, cell_size, freq_resolution):
    # Create a grid of the bpns
    minx, miny, maxx, maxy = borders_df.total_bounds

    # create the cells in a loop
    xv = np.arange(minx, maxx, cell_size)
    yv = np.arange(miny, maxy, cell_size)

    x, y = np.meshgrid(xv, yv)
    geometry_points = geopandas.points_from_xy(x.reshape((-1, 1))[:, 0], y.reshape((-1, 1))[:, 0], crs='epsg:4326')
    geodf = geopandas.GeoDataFrame(geometry=geometry_points)

    # Could be that we have a df (here a random one) and we want to add a geolocation to it
    # Create a random dataframe to work with
    time_array = pd.date_range(start=start_time, end=end_time, freq=freq_resolution, tz='UTC')

    # Define the seadatamanager
    env_vars = [
        "shipping",
        "time",
        "habitat_suitability",
        "seabed_habitat",
        "sea_surface",
        "sea_wave",
        "wrakken_bank",
        "bathymetry"
    ]
    manager = bpnsdata.SeaDataManager(env_vars)
    sl = bpnsdata.geolocation.SurveyLocation()
    df = geopandas.GeoDataFrame()
    for time_slot in tqdm(time_array, total=len(time_array)):
        geodf['datetime'] = time_slot
        geodf = sl.add_distance_to_coast(geodf, coastfile='geo/belgium_coast/basislijn_BE.shp')
        # Call the manager
        df_env = manager(geodf, datetime_column='datetime')
        df = pd.concat([df, df_env], ignore_index=True)

    # Add the season var
    df['season'] = df.datetime.dt.isocalendar().week

    # Set the desired instrument depth
    instrument_depth = 5
    df['instrument_depth'] = instrument_depth

    return df


def convert_to_samples(x):
    x['current'] = np.sqrt(x['surface_baroclinic_eastward_sea_water_velocity']**2 +
                           x['surface_baroclinic_northward_sea_water_velocity']**2)
    env_labels_rename = {
        'sea_surface_height_above_sea_level': 'tide',
        'sea_surface_salinity': 'salinity',
        'sea_surface_temperature': 'temperature',
        'route_density': 'shipping'
    }
    x = x.rename(columns=env_labels_rename)

    x = x.replace(['Astronomical twilight', 'day_moment_Civil twilight', 'day_moment_Nautical twilight'],
                  ['twilight', 'twilight', 'twilight'])

    ENV_LABELS = [
        'shipping',
        'season',
        'moon_phase',
        'day_moment',
        'benthic_habitat',
        'substrate',
        'seabed_habitat',
        'tide',
        'salinity',
        'temperature',
        'current',
        'bathymetry',
        'shipwreck_distance',
        'coast_dist',
        'instrument_depth'
    ]

    x = x[ENV_LABELS]

    # Prepare the data for the RF
    for cy_var in ['moon_phase', 'season']:
        if cy_var == 'season':
            # Convert the week number into a degree
            x[cy_var] = x[cy_var] / 52 * 2 * np.pi
        x[cy_var + '_sin'] = np.sin(x[cy_var])
        x[cy_var + '_cos'] = np.cos(x[cy_var])
        x = x.drop([cy_var], axis=1)

    CATEGORICAL_VARS = ['day_moment', 'benthic_habitat', 'substrate', 'seabed_habitat']

    for c in x.columns:
        if c in CATEGORICAL_VARS:
            x[c] = x[c].astype(str)

    x.rename(columns={'moon_phase_sin': 'growing_moon', 'moon_phase_cos': 'new_moon',
                      'season_sin': 'week_n_sin', 'season_cos': 'week_n_cos'},
              inplace=True)

    onehot_enc = OneHotEncoderDF(handle_unknown='error', sparse=False).fit(x[CATEGORICAL_VARS])
    x_encoded = onehot_enc.transform(x[CATEGORICAL_VARS])
    x_numerical = x.drop(columns=CATEGORICAL_VARS)
    return x_numerical.join(x_encoded)


working_dir = pathlib.Path('output/predictions')
if not working_dir.exists():
    os.mkdir(working_dir)
borders_EEZ = pathlib.Path('geo/boundaries/eez_boundaries_v10_BE_epsg4326.shp')
belgium_borders = geopandas.read_file(borders_EEZ)
spatial_spacing = 0.01
time_spacing = 'h'

start_time_ = '2020-05-12 12:00'
end_time_ = '2020-06-12 12:00'
df_path = working_dir.joinpath('df_env_res_%s_%s.pkl' % (spatial_spacing, time_spacing))
if not df_path.exists():
    df_raw = create_time_space_df(start_time_, end_time_, belgium_borders, spatial_spacing, time_spacing)
    df_raw.to_pickle(df_path)
else:
    df_raw = pd.read_pickle(df_path)

print('preparing data...')
df_raw['latitude'] = df_raw.geometry.y
df_raw['longitude'] = df_raw.geometry.x
df_raw = df_raw.dropna()
samples_rf = convert_to_samples(df_raw)

best_model = joblib.load(working_dir.joinpath('model_predictions/final_model.joblib'))
for used_column in best_model.sample_.features.columns:
    if used_column not in samples_rf.columns:
        samples_rf[used_column] = 0

samples_rf = samples_rf[best_model.sample_.features.columns]

print('predicting...')
rf_clusters = best_model.pipeline.native_estimator.predict(samples_rf)
df_raw.loc[samples_rf.index, 'predicted_class'] = rf_clusters.astype(int)

df_raw = df_raw.set_index(['datetime', 'latitude', 'longitude'])
ds = df_raw.to_xarray()

# Create a cmap to match publication
cmap_colors = plt.get_cmap('tab20', 17).colors[1:, :]
cmap = ListedColormap(cmap_colors, 'clusters')

ds['datetime'] = pd.to_datetime(ds['datetime'])
frames = []
fig, ax = plt.subplots()
camera = Camera(fig)
print('generating gif...')
for t, ds_t in tqdm(ds.groupby('datetime')):
    xarray.plot.pcolormesh(ds_t['predicted_class'], x='longitude', y='latitude', cmap=cmap, add_colorbar=False,
                           vmin=0, vmax=17, animated=True, ax=ax)
    belgium_borders.to_crs(df_raw.crs).plot(ax=ax, color='k')
    ax.set_title('')
    ax.text(0.1, 1.01, t, transform=ax.transAxes)

    # Take a snapshot for the gif
    plt.pause(0.1)
    camera.snap()

animation = camera.animate()
animation.save(working_dir.joinpath('map_predictions', 'animation_res%s_%s.gif' % (spatial_spacing, time_spacing)),
               fps=2)

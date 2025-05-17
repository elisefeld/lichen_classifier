import logging
from utils import data, plotting
from config import Config
cfg = Config()

logging.basicConfig(level=cfg.log_level)
logger = logging.getLogger(__name__)

df = data.load_and_clean_obs_data()
print(df.info())

keep_cols = ['uuid',
             'filename',
             'observed_on_dt',
             'observed_on_month',
             'observed_on_day',
             'observed_on_year',
             'time_observed_at_dt',
             'time_observed_at_hour',
             'time_observed_at_minute',
             'time_observed_at_second',
             'time_zone',
             'large_image_url',
             'num_identification_agreements',
             'num_identification_disagreements',
             'latitude',
             'longitude',
             'taxon_id',
             'genus',
             'scientific_name',
             'morphology'
             ]

df = df[keep_cols].copy()
df = df.sort_values(by=['scientific_name'], ascending=False)
print(df.head())
logger.info('Number of unique genus values: %d', df['genus'].nunique())

# Plotting
plotting.plot_class_distribution(df)
plotting.plot_time(df, column='observed_on_day', type='Day')
plotting.plot_time(df, column='observed_on_month', type='Month')
plotting.plot_time(df, column='observed_on_year', type='Year')
plotting.plot_location(df, facet_type='facetted')
plotting.plot_location(df, facet_type='non-facetted')

# Download images
failed_uuids = data.save_imgs(df)

if failed_uuids and len(failed_uuids) > 0:
    logger.info('Failed to download images for UUIDs: %s', failed_uuids)
    df = df[~df['uuid'].isin(failed_uuids)].reset_index(drop=True)

# Save counts of observations
if cfg.save_counts:
    data.save_counts(df, col='scientific_name')
    data.save_counts(df, col='genus')

# Save location data
if cfg.save_location:
    df[['filename', 'latitude', 'longitude']].copy().to_csv(
        cfg.EDA_dir/'location.csv', index=False)

# Saving cleaned results
df.to_csv(cfg.data_dir / 'obs_data_cleaned.csv', index=False)

# Split data into train, validation, and test sets
data.train_test_split(overwrite=True)

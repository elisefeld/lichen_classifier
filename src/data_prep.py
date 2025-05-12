import logging
from utils import data, scraping, plotting
from config import Config
cfg = Config()

logging.basicConfig(level=cfg.log_level)
logger = logging.getLogger(__name__)


df = data.load_and_clean_obs_data(cfg.data_paths)
logger.info(df.info())

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
             ]

df = df[keep_cols].copy()
df.sort_values(by=['scientific_name'], ascending=False, inplace=True)
logger.info(df.head())
logger.info("Number of unique genus values: %d", df['genus'].nunique())

# Plotting
to_filter = True
plotting.plot_class_distribution(df, filter=to_filter)

plotting.plot_time(df, column='observed_on_day', type='Day')
plotting.plot_time(df, column='observed_on_month', type='Month')
plotting.plot_time(df, column='observed_on_year', type='Year')

plotting.plot_location(df, filter=to_filter, facet=False)
plotting.plot_location(df, filter=to_filter, facet=True)

#######################
#### DOWNLOAD IMGS ####
#######################
filter_list = ['Xanthoria',
               'Xanthomendoza',
               'Vulpicida',
               'Usnea',
               'Umbilicaria',
               'Teloschistes',
               'Rusavskia',
               'Rhizoplaca',
               'Punctelia',
               'Porpidia',
               'Platismatia',
               'Pilophorus',
               'Physcia',
               'Phaeophyscia',
               'Peltigera',
               'Parmotrema',
               'Parmelia',
               'Niebla',
               'Multiclavula',
               'Lichenomphalia',
               'Letharia',
               'Lecanora',
               'Lasallia',
               'Icmadophila',
               'Heterodermia',
               'Herpothallon',
               'Graphis',
               'Flavopunctelia',
               'Evernia',
               'Dimelaena',
               'Dibaeis',
               'Candelaria',
               'Arrhenia',
               'Acarospora']

failed_uuids = scraping.save_imgs(df,
                                  output_dir=cfg.image_dir,
                                  to_filter=True,
                                  filter_type='genus',
                                  filter_list=filter_list)
if failed_uuids and len(failed_uuids) > 0:
    logger.info("Failed to download images for UUIDs: %s", failed_uuids)
    df = df[~df['uuid'].isin(failed_uuids)].reset_index(drop=True)

# Saving cleaned results
if cfg.save_counts:
    data.save_counts(df, col='scientific_name')
    data.save_counts(df, col='genus')

if cfg.save_location:
    path = cfg.get_file_name(cfg.location_dir, 'location', 'csv')
    df[['filename', 'latitude', 'longitude']].copy().to_csv(path, sep='\t', index=False)

df.to_csv(cfg.data_dir / 'obs_data_cleaned.csv', sep='\t', index=False)



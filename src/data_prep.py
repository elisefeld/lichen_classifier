import logging
from utils import data
from utils import scraping, plotting

from config import Config
cfg = Config()

logging.basicConfig(level=logging.INFO)
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

# Plotting
plotting.plot_class_distribution(df, filter=True)

plotting.plot_time(df, column='observed_on_day', type='Day')
plotting.plot_time(df, column='observed_on_month', type='Month')
plotting.plot_time(df, column='observed_on_year', type='Year')

logger.info("Number of unique genus values: %d", df['genus'].nunique())
plotting.plot_location(df, filter=True, facet=False)
plotting.plot_location(df, filter=True, facet=True)

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
data.save_counts(df, cfg.data_dir)
location = df[['filename', 'latitude', 'longitude']].copy()
location.to_csv(cfg.data_dir / 'location.csv', sep='\t', index=False)
df.to_csv(cfg.data_dir / 'obs_data_cleaned.csv', sep='\t', index=False)

scraping.train_test_split(source_dir=cfg.full_img_dir,
                          dest_train_dir=cfg.train_dir,
                          dest_test_dir=cfg.test_dir,
                          dest_val_dir=cfg.val_dir,
                          ratio=cfg.val_test_split)

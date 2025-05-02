import logging
import utils.obs_data
import visualization.obs_plots
from config import Config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
cfg = Config()

df = utils.obs_data.load_and_clean_obs_data(cfg.data_paths)
df = utils.obs_data.remove_duplicates(df)
utils.obs_data.save_counts(df, cfg.data_dir)
logger.info("Null values:\n%s", df.isnull().sum())

print(df.info())

keep_cols = ['uuid',
             'filename',
             'observed_on',
             'time_observed_at',
             'time_zone',
             'large_image_url',
             'num_identification_agreements',
             'num_identification_disagreements',
             'latitude',
             'longitude',
             'taxon_id',
             'genus',
             'scientific_name']

df = df[keep_cols].copy()
df.sort_values(by=['scientific_name'], ascending=False, inplace=True)
print(df.head())

print('Null: \n', df.isnull().sum())

utils.visualization.obs_plots.plot_time(df)
utils.visualization.obs_plots.plot_location(df)


#######################
#### DOWNLOAD IMGS ####
#######################
filter_list = ['Xanthoria',
               'Xanthomendoza',
               'Vulpicida',
               'Usnea',
               # 'Umbilicaria',
               # 'Teloschistes',
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


location = df[['filename', 'latitude', 'longitude']].copy()
location.to_csv(cfg.data_dir / 'location.csv', sep='\t', index=False)

if cfg.download:
    failed_uuids = utils.save_imgs(df,
                                   output_dir=cfg.image_dir,
                                   to_filter=True,
                                   filter_type='genus',
                                   filter_list=filter_list)
    df = df[~df['uuid'].isin(failed_uuids)].reset_index(drop=True)
    df.to_csv(cfg.data_dir / 'obs_data_cleaned.csv', sep='\t', index=False)


utils.scraping.train_test_split(source_dir=cfg.full_img_dir,
                       dest_train_dir=cfg.train_dir,
                       dest_test_dir=cfg.test_dir,
                       dest_val_dir=cfg.val_dir,
                       ratio=cfg.val_test_split,
                       seed=cfg.seed)

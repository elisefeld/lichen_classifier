import pandas as pd
import utils
from config import Config

#######################
#### DOWNLOAD DATA ####
#######################
cfg = Config()
raw_obs = pd.concat([pd.read_csv(path)
                    for path in cfg.data_paths], ignore_index=True)

#######################
##### CLEAN DATA  #####
#######################
raw_obs['large_image_url'] = raw_obs['image_url'].str.replace(
    'medium', 'large')
raw_obs['genus'] = raw_obs['scientific_name'].str.split(' ').str[0]
raw_obs['filename'] = raw_obs['uuid'].apply(lambda x: str(x) + '_full.jpg')

utils.save_counts(raw_obs, cfg.data_dir)

print(raw_obs.info())

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
             'scientific_name',
             'common_name']

df = raw_obs[keep_cols].copy()
df.sort_values(by=['scientific_name'], ascending=False, inplace=True)
print(df.head())

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
                'Phaeophyscia'
                'Peltigera',
                'Parmotrema',
                'Parmelia',
                'Niebla',
                'Multiclavula',
                'Lichenomphalia',
                'Letharia',
                'Lecanora'
                'Lasallia',
                'Icmadophila'
                'Heterodermia',
                'Herpothallon',
                'Graphis',
                'Flavopunctelia',
                'Evernia',
                'Dimelaena',
                'Dibaeis',
                'Candelaria',
                'Arrhenia'
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


if cfg.download:
    failed_uuids = utils.save_imgs(df,
                                   output_dir=cfg.image_dir,
                                   to_filter=False)
    df = df[~df['uuid'].isin(failed_uuids)].reset_index(drop=True)
    df.to_csv(cfg.image_dir / 'obs_data_cleaned.csv', sep='\t', index=False)

utils.train_test_split(source_dir=cfg.full_img_dir,
                       dest_train_dir=cfg.train_dir,
                       dest_test_dir=cfg.test_dir,
                       dest_val_dir=cfg.val_dir,
                       ratio=cfg.val_test_split,
                       seed=cfg.seed,)
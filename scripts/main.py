import pandas as pd
import utils
from config import Config
import matplotlib.pyplot as plt
import seaborn as sns

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

print('Null: \n', df.isnull().sum())

duplicate_uuids = df[df.duplicated('uuid', keep=False)]
print(duplicate_uuids.sort_values('uuid'))
df = df.drop_duplicates(subset='uuid', keep='first')

df['observed_on'] = pd.to_datetime(df['observed_on'], errors='coerce')

df['observed_on'].dt.to_period('M').value_counts().sort_index().plot(
    kind='line', title='Observations Over Time', figsize=(12, 6))
plt.show()

df['hour'] = pd.to_datetime(df['time_observed_at'], errors='coerce').dt.hour
df['hour'].value_counts().sort_index().plot(kind='bar', title='Observations by Hour')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(df['longitude'], df['latitude'], alpha=0.5, s=1)
plt.title('Geospatial Distribution of Observations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


top_genera = df['genus'].value_counts().head(10).index
df_top = df[df['genus'].isin(top_genera)]

plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_top, x='longitude', y='latitude', hue='genus', alpha=0.6, s=15, palette='tab20')
plt.title('Geospatial Distribution by Genus')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.tight_layout()
plt.show()


g = sns.FacetGrid(df_top, col='genus', col_wrap=3, height=4)
g.map(sns.scatterplot, 'longitude', 'latitude', alpha=0.5, s=10)
g.figure.subplots_adjust(top=0.9)
g.figure.suptitle('Geospatial Distribution by Genus')
plt.show()


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


utils.train_test_split(source_dir=cfg.full_img_dir,
                       dest_train_dir=cfg.train_dir,
                       dest_test_dir=cfg.test_dir,
                       dest_val_dir=cfg.val_dir,
                       ratio=cfg.val_test_split,
                       seed=cfg.seed)
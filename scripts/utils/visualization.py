import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_time(df: pd.DataFrame):
    df['observed_on_month'].value_counts().sort_index().plot(
        kind='line', title='Observations Over Time', figsize=(12, 6))
    plt.show()

    df['time_observed_at_hour'].value_counts().sort_index().plot(
        kind='bar', title='Observations by Hour', figsize=(12, 6))
    plt.show()

def plot_location(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['longitude'], df['latitude'], alpha=0.5, s=1)
    plt.title('Geospatial Distribution of Observations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    top_genera = df['genus'].value_counts().head(10).index
    df_top = df[df['genus'].isin(top_genera)]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_top, x='longitude', y='latitude',
                    hue='genus', alpha=0.6, s=15, palette='tab20')
    plt.title('Geospatial Distribution by Genus')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.tight_layout()
    plt.show()

    g = sns.FacetGrid(df_top, col='genus', col_wrap=3, height=4)
    g.map(sns.scatterplot, 'longitude', 'latitude', alpha=0.5, s=10)
    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle('Geospatial Distribution by Genus')
    plt.show()
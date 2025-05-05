from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import contextily as cx
from shapely.geometry import Point

def get_top_genera(df: pd.DataFrame, top_column: str = 'genus', top_n: int = 10) -> pd.DataFrame:
    '''
    Filters a DataFrame to include only the top N most frequent values in a specified column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to filter.
        top_column (str): The name of the column to analyze for the most frequent values. Defaults to 'genus'.
        top_n (int): The number of top values to include in the filtered DataFrame. Defaults to 10.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows where the specified column's value
                      is among the top N most frequent values.
    '''
    top_n_genera = df[top_column].value_counts().nlargest(top_n).index
    df_top = df[df['genus'].isin(top_n_genera)]
    return df_top

def get_class_names(image_root: Path, level: int = 1):
    class_names = []
    if level == 1:
        class_names = [d.name for d in image_root.iterdir() if d.is_dir()]
    elif level == 2:
        class_names = [f"{p.parent.name}/{p.name}"
                       for p in image_root.glob("*/*") if p.is_dir()]
    class_names = sorted(class_names)
    return class_names




def plot_time(df: pd.DataFrame):
    df = df[df['genus'].isin(get_class_names(Path()))].copy()
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
    plt.savefig()

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
    plt.savefig()


def plot_genus_distribution_map(df, top_n=10):
    df = df.dropna(subset=['latitude', 'longitude'])
    df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    gdf = gdf.to_crs(epsg=3857)

    top_genus = gdf['genus'].value_counts().nlargest(top_n).index
    gdf = gdf[gdf['genus'].isin(top_genus)]

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(ax=ax, column='genus', categorical=True, legend=True, markersize=10, alpha=0.6)
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    plt.title("Lichen Genus Distribution Across Map Tiles")
    plt.tight_layout()
    plt.savefig()
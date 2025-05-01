import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_clean_obs_data(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    df['large_image_url'] = df['image_url'].str.replace(
    'medium', 'large')
    df['genus'] = df['scientific_name'].str.split(' ').str[0]
    df['filename'] = df['uuid'].apply(lambda x: str(x) + '_full.jpg')
    df['observed_on_dt'] = pd.to_datetime(df['observed_on'], errors='coerce')
    df['observed_on_month'] = df['observed_on_dt'].dt.month
    df['observed_on_day'] = df['observed_on_dt'].dt.day
    df['observed_on_year'] = df['observed_on_dt'].dt.year
    df['time_observed_at_dt'] = pd.to_datetime(df['time_observed_at'], errors='coerce')
    df['time_observed_at_hour'] = df['time_observed_at_dt'].dt.hour
    df['time_observed_at_minute'] = df['time_observed_at_dt'].dt.minute
    df['time_observed_at_second'] = df['time_observed_at_dt'].dt.second
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_uuids = df[df.duplicated('uuid', keep=False)]
    if not duplicate_uuids.empty:
        logger.info("Found duplicate UUIDs:\n%s", duplicate_uuids.sort_values('uuid'))
    return df.drop_duplicates(subset='uuid', keep='first')

def save_counts(df: pd.DataFrame, counts_dir: Path):
    counts_dir.mkdir(parents=True, exist_ok=True)
    for var in ['scientific_name', 'genus']:
        df[var].value_counts().to_csv(
            counts_dir / f'{var}_counts.csv', sep='\t')
        


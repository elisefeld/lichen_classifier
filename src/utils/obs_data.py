import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_clean_obs_data(paths: list[Path]) -> pd.DataFrame:
    '''
    Load and clean observational data from a list of CSV file paths.

    This function reads multiple CSV files, concatenates them into a single
    DataFrame, and performs various cleaning and transformation operations
    on the data.

    Args:
        paths (list[Path]): A list of file paths to the CSV files containing
            observational data.

    Returns:
        pd.DataFrame: A cleaned and transformed DataFrame containing the
        concatenated data from all input files with additional columns:
            - 'large_image_url': Modified image URL with 'medium' replaced by 'large'.
            - 'genus': Extracted genus from the 'scientific_name' column.
            - 'filename': Generated filename based on the 'uuid' column.
            - 'observed_on_dt': Parsed datetime from the 'observed_on' column.
            - 'observed_on_month': Extracted month from 'observed_on_dt'.
            - 'observed_on_day': Extracted day from 'observed_on_dt'.
            - 'observed_on_year': Extracted year from 'observed_on_dt'.
            - 'time_observed_at_dt': Parsed datetime from the 'time_observed_at' column.
            - 'time_observed_at_hour': Extracted hour from 'time_observed_at_dt'.
            - 'time_observed_at_minute': Extracted minute from 'time_observed_at_dt'.
            - 'time_observed_at_second': Extracted second from 'time_observed_at_dt'.
    '''
    # Read and concatenate CSV files
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    # Clean and transform data
    df['filename'] = df['uuid'].apply(lambda x: str(x) + '_full.jpg')
    df['large_image_url'] = df['image_url'].str.replace(
        'medium', 'large')
    df['genus'] = df['scientific_name'].str.split(' ').str[0]
    df['observed_on_dt'] = pd.to_datetime(df['observed_on'], errors='coerce')
    df['observed_on_month'] = df['observed_on_dt'].dt.month
    df['observed_on_day'] = df['observed_on_dt'].dt.day
    df['observed_on_year'] = df['observed_on_dt'].dt.year
    df['time_observed_at_dt'] = pd.to_datetime(
        df['time_observed_at'], errors='coerce')
    df['time_observed_at_hour'] = df['time_observed_at_dt'].dt.hour
    df['time_observed_at_minute'] = df['time_observed_at_dt'].dt.minute
    df['time_observed_at_second'] = df['time_observed_at_dt'].dt.second
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Removes duplicate rows from a DataFrame based on the 'uuid' column.

    If duplicate UUIDs are found, logs the duplicates for inspection.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'uuid' column.

    Returns:
        pd.DataFrame: A DataFrame with duplicates removed, keeping the first occurrence.
    '''
    duplicate_uuids = df[df.duplicated('uuid', keep=False)]
    if not duplicate_uuids.empty:
        logger.info("Found duplicate UUIDs:\n%s",
                    duplicate_uuids.sort_values('uuid'))
    return df.drop_duplicates(subset='uuid', keep='first')


def save_counts(df: pd.DataFrame, counts_dir: Path) -> None:
    '''
    Saves the value counts of specified columns in a DataFrame to CSV files.

    This function calculates the value counts for the columns 'scientific_name' 
    and 'genus' in the provided DataFrame and saves the results as tab-separated 
    CSV files in the specified directory. If the directory does not exist, it 
    will be created.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        counts_dir (Path): The directory where the count files will be saved.

    Returns:
        None
    '''
    counts_dir.mkdir(parents=True, exist_ok=True)
    for var in ['scientific_name', 'genus']:
        df[var].value_counts().to_csv(
            counts_dir / f'{var}_counts.csv', sep='\t')

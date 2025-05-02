# Lichen Image Classification
This project is a CNN pipeline for classifying lichens in images. 

## Project Structure
[config](src/config.py) - configuration file that contains all the hyperparameters and paths to the data directories  

[data_prep](src/data_prep.py) - main script for downloading and cleaning the iNaturalist observation data  

[train](src/train.py) - main script for training the model using the training and validation sets  

### utils modules
[scraping](src/utils/scraping.py) - fetching images from iNaturalist and creating training, validation and test sets  

[obs_data](src/utils/obs_data.py) - data wrangling functions for observation data  

[img_data](src/utils/img_data.py) - loading and label functions for images   

### visualization modules
[obs_plots](src/visualization/obs_plots.py) - plotting functions for observation data  

[img_plots](src/visualization/img_plots.py) - plotting functions for image data  

### modeling modules
[cnn_model](src/modeling/cnn_model.py) - convolutional neural network model architecture  

[evaluate](src/modeling/evaluate.py) - evaluating the trained model on a test set  
    

## Datasets
The raw data can be found in the [data folder](data/raw). It was downloaded using [iNaturalist's export tool](https://www.inaturalist.org/observations/export) in two batches (before 2017-01-01 and between 2017-01-01 and 2025-04-19.) 

Filters:
- Research grade
- Most identifiers agreed
- Open geoprivacy (location not obscured)
- United States

Since lichen are not restricted to one taxonomic group, 'lichen' were defined as observations matching the 1,017 genera on the [Lichen genera](https://en.wikipedia.org/wiki/Category:Lichen_genera) wikipedia category page. Taxon IDs were queried using iNaturalist's API and then passed into the export tool. 

The exact query parameters are listed [here](data/obs_data_queries.md).


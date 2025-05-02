# Lichen Image Classification
This project is a CNN pipeline for classifying lichens in images. 

## Project Structure
[config](scripts/config.py) - configuration file that contains all the hyperparameters and paths to the data directories  

[data_prep](scripts/data_prep.py) - main script for downloading and cleaning the iNaturalist observation data  

[train](scripts/train.py) - main script for training the model using the training and validation sets  

### utils modules
[scraping](scripts/utils/scraping.py) scraping.py - fetching images from iNaturalist and creating training, validation and test sets  

[obs_data](scripts/utils/obs_data.py) - data wrangling functions for observation data  

[img_data](scripts/utils/img_data.py) - loading and label functions for images   

### visualization modules
[obs_plots](scripts/visualization/obs_plots.py) - plotting functions for observation data  

[img_plots](scripts/visualization/img_plots.py) - plotting functions for image data  

### modeling modules
[cnn_model](scripts/modeling/cnn_model.py) - convolutional neural network model architecture  

[evaluate](scripts/modeling/evaluate.py) - evaluating the trained model on a test set  
    

## Datasets
The raw data can be found in the [data folder](data/raw). It was downloaded using [iNaturalist's export tool](https://www.inaturalist.org/observations/export) in two batches (before 2017-01-01 and between 2017-01-01 and 2025-04-19. The query parameters are listed [here](data/obs_data_queries.md).


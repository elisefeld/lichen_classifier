# Lichen Image Classification
This project is a CNN pipeline for classifying lichens in images. 

## Project Structure
config.py - configuration file that contains all the hyperparameters and paths to the data directories  

data_prep.py - main script for downloading and cleaning the iNaturalist observation data  

train.py - main script for training the model using the training and validation sets  

data ---  

    scraping.py - fetching images from iNaturalist and creating training, validation and test sets  

    obs_data.py - data wrangling functions for observation data  

    img_data.py - loading and label functions for images   

visualization ---  

    obs_plots.py - plotting functions for observation data  

    img_plots.py - plotting functions for image data  

modeling ---  

    cnn_model.py - convolutional neural network model architecture  

    evaluate.py - evaluating the trained model on a test set  
    

## Datasets
The raw data is found in the folder data/raw. It was downloaded using ([iNaturalist's export tool](https://www.inaturalist.org/observations/export)) in two batches.


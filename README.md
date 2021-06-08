# InteractiveLatentSpace

Repository for the paper "Understanding Data by Exploring and Explaining the Latent Space"

Requirements:
snikit-learn
pytorch
numpy
pandas

Usage:
'app.py -d <dataset name> -r <number of points to visualise> 
    -d --dataset : name of the dataset to utilise, only "titanic" or "adult" are supported  
    -r --reduced: option to reduce the number of points to visualise, if this option is not 
                  provided then all the test dataset is used for visualisation. 
                  For the titanic dataset a number less than 5000 is recommeded'

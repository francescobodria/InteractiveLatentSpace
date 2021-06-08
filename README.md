# InteractiveLatentSpace

Repository for the paper "Understanding Data by Exploring and Explaining the Latent Space"

Requirements:<br>
- scikit-learn<br>
- pytorch<br>
- numpy<br>
- pandas<br>

Usage:<br>

    app.py -d dataset_name -r number_of_points_to_visualise

        -d --dataset : name of the dataset to utilise, only "titanic" or "adult" are supported  

        -r --reduced: option to reduce the number of points to visualise, if this option is not 
                      provided then all the test dataset is used for visualisation. 
                      For the titanic dataset a number less than 5000 is recommeded'


![figure](/assets/figure.png)

# DL4All - AutoML with Tensorflow
A short and simple 3 layers Neural Network Tensorflow code to classification  
Just specify your train and test/validation sets and let the algorithm classify your data!  
**python neural_network.py** does it all.  

## Dependencies
python 3.6.8 
#### Packages
numpy 1.16.3  
pandas 0.24.2  
scikit-learn 0.21.1  
scipy 1.2.1  
tensorflow 1.13.1 (to CPU process)  
tensorflow-gpu 1.13.1 (to GPU process)  

## Architecture
It's a 3 hidden layers MLP.  
You just need to specify a .csv file with the data to training, test or validation in dataset folder.  
You can set up the training/test file in the variable *train_path* and *test_path* **neural_network.py**  
It's set to iterate up to 300 epochs with an early stopping logic to avoid divergence.  


<img src="https://lh3.googleusercontent.com/wc7NnyZE1Y8hGL8A-fu49JVvNHJ7JAD5oALbXZW-HURfGbSZEHaW8wRDqgkJAT_d8kER8cyihtD0dMGESdrZuWzk9nlh03FJ5jy5FbuX36IBU6Ijs0WeUU5P0QdQRdkNb-szMafaoslx_UpCxNstm4mDFH4ORjI-RgFNS0ktcgPTqowQA6aa2NmX_xvKx5nPP-9sIps_doXZ9tg5dPtRRi2zHLXFEmL73bl8a-vJi8sgwTTvDQhcBIWtoXfLYFnhA4lN532YQXoEuoylk9vmZDZshQLAAxqVQ-zU3XtWPhwjinDVW1V-dIGhj-_9iv-W1-YS09B-ap_2WVs-QjtSjgldf3uq2UUJeu0lJwL5NmK24RBnZI0KaMP5cM4fYQvlt3v43Yt19QmvxRhx1RcT6DpeGD0oDFalHUMbKePlovFPGEz53oFTSB5m_zCXNiI88r61plZaWki4cbRd1VXQEVpFfwSMigqbEr_PW2LmzlKRFYs5tr-6llhwQWLbLuFD_7fup4LRyj_7_97lOPRLYSobUD1GNuFhY6HoSVon2oJN_f30z43-WWjDgcDqCWyFLUYE-OcKVREIQD6QwDxvVbWFd-mo9NyqCHSNdPWhVGbacmVm6TQfKzPmH934aTS3h7nFueTA7hFP9PmwtsGFm2yv73tCVd0=w796-h517-no" width="500" height="320">


## Hyperparameters
LAYER1 = Neurons number in the first hidden layer  
LAYER2 = Neurons number in the second hidden layer  
LAYER3 = Neurons number in the third hidden layer  
LR = learning rate  
BETA = beta parameter to regularization. 0 to ignore  
The Hyperparameter optimization search is performed by a Random Search algorithm with 3 trails of hyperparameter  
**You can change the number of trails in the main function**

## Feature Selection
The method ExtraTreesClassifier is automatically applied to reduce the data dimension.  
In the MNIST example, the data dimension is reduced of 784 features to 274.

## Results
The best result founded in the Random Search is saved in the folder **TF_model**.  
The test/validation classification is inside this folder (**classification.csv**)  
The Tensorflow model (session) is saved inside the **sess** folder.

## Learning plot
At the end of the execution, some metrics like error rate and accuracy will be presented.  
A learning plot will also be displayed with train and test error rate.  

<img src="./learning_plot.png" width="440" height="280">

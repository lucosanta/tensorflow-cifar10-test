# CIFAR-10 Tensorflow Test
In this experiment, I will try to master my abilities with Tensorflow on CIFAR-10 dataset. 

[Tensorflow](https://www.tensorflow.org/)

## CIFAR-10

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

Data could be found at [Python CIFAR-10 Data](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

## Actions
In this algorithm I had to:

  * Import CIFAR-10
  * Convert dataset to TFRecord
  * Read TFRecord file
  * Create batch
  * Create a model with 3 Conv Layer and 2 Fully Connected Layers
  * Optimizer is Adam with parametric Learning Rate with a decay 
  * Training for 1000 epochs
  * Results written inside Tensorboard for analysis
  
## Run the code

Open a Terminal and clone this repository.

Go inside the folder.

`$ cd <repository_folder>`

To run, please insert this code below:

`$ python3 main.py --train`

To see runtime visualization of how training is going, please type this in Terminal

`$ tensorboard --logdir=summary/`


## Results
In this section, there are illustrated results obtained after a training session. 

### Original graphs
![Accuracy Train](accuracy_train.png)
![Accuracy Test](accuracy_test.png)

### Smoothed graphs
![Accuracy Train Smooth](accuracy_train_smooth.png)
![Accuracy Test Smooth](accuracy_test_smooth.png)


### Comparison
The comparison between data has been produced using:
`$ python3 plot_comparison.py`
It needs a CSV file named `run_tcomparison-accuracy.csv` with 3 columns(`step`,`test`,`train`)
![Comparison](Comparison.png)  

## Conclusion

Results are just as expected which means with an accuracy near 40% in training phase compared to the ~10% obtained with random results. In a previous try, I had a 99% which was caused by missing augmentation of data inside the `shuffle_batch` related to the images and it was suffering of overfitting.

This folder implements Co-Training and PU-Learning techniques in python for detecting opinion spam in Yelp reviews

## Installation

Before running the algorithms clean the data using [cleanData](https://github.com/airesearchuwt/spamGAN/blob/master/coTrain_PULearn/cleanData.py) and generate the required data files.

### Co-Train

Use [data_Analysis](https://github.com/airesearchuwt/spamGAN/blob/master/coTrain_PULearn/data_Analysis.py) to extract features from the dataset. 

Use the file with extracted features and execute [co-training](https://github.com/airesearchuwt/spamGAN/blob/master/coTrain_PULearn/co-training.py) algorithm.

### PU-Learn

 Using the files generated from [cleanData](https://github.com/airesearchuwt/spamGAN/blob/master/coTrain_PULearn/cleanData.py) execute the [PU-Learning](https://github.com/airesearchuwt/spamGAN/blob/master/coTrain_PULearn/pu-learn.py) algorithm.


<b>Note:</b> Currently, the code is built in such a way that one needs to add datafile paths in the respective python files to get results

## Alcoholic Subject Detection: Classification of EEG signals using CasPer Algorithm
The code is based on the CasPer algorithm proposed by Treadgold, N.K. and Gedeon, T.D. (http://dx.doi.org/10.1007/3-540-63797-4_93), implemented in pytorch. And performed classification on the UCI EEG Alcoholism dataset.

## Disclaimer:
This project was done as a University research project. Datasets are not available at this repo.

## Dependencies:
DEAP <br>
Pandas <br>
Scipy.io <br>
Pytorch <br>
Matplotlib <br>
Standard Python Library

## File description:
within_subject_train_test.py - within_subject training and testing <br>
cross_subject_train_test.py - cross_subject training and testing <br>
GA_cross_subject.py - run genetic algorithm to determine the hyper-paramter for cross-subject testing <br>
GA_within_subject.py - run genetic algorithm to determine the hyper-paramter for cross-subject testing <br>
Autoencoder.py - the file containing the autoencoder model <br>
draw_subject_for_model_selection.py - can be called by other files to get the data for model selection and actual train-test <br>
model_selection.py - originally use for hyperparameter selecting, now used as determine the best topology for Casper before running GAs for hyperparameter testing <br>
casper_main.py - the file that contains the Casper model, not to be called directly (call within_subject_train_test.py or cross_subject_train_test.py instead) <br>
model/model.pth - the trained Autoencoder model <br>
extra/data_exploration_plots.ipynd - the jupyter notebook file for plotting <br>
./alcoholism - References papers for the preprocessed dataset <br>


## Procedures
casper_main.py is used for specifying Casper neural network and for training the network, it does not run on its own as it requires dataset as input,
the entire project is done by these steps using these files:
1. draw_subject_for_model_selection.py  for randomly drawing 12 subjects for model selection (redrawn if the 12 subject is extremely unbalanced)
2. determining the topology of the Casper using model_selection.py
3. train the autoencoder using autoencoder.py
4. run GA_cross_subject and GA_within_subject for determining the hyperparameters for the task
5. run within_subject_train_test.py for within subject testing
6. run cross_subject_train-test.py for cross subject testing

### Regarding the result of the model please directly run 5, 6 <br/>
within_subject_train_test.py for within subject testing <br/>
cross_subject_train-test.py for cross subject testing

## Testing result using 11-fold Cross Validation
#### Cross subject
Testing Accuracy mean: 64.2%
#### Within subject
Testing Accuracy: 87.9%

Genetic Algorithm has limited to effect to the accuracy

## Originality and references
Method: Treadgold, N.K. and Gedeon, T.D. (http://dx.doi.org/10.1007/3-540-63797-4_93).<br>
Dataset: UCI EEG Alcoholism dataset (https://archive.ics.uci.edu/ml/datasets/eeg+database), two different preprocessed version of the dataset were used in the code, both preprocessed by Yao, Plested and Gedeon (2020).

The Casper algorithm is implemented base on some sample code provided by Australian National University, algorithm/codes including normalization, plotting loss diagram and plotting error matrix and printing epoch/loss and accuracy has been copied directly from the University's course material. Some codes are provided UCI originally serve as an example for the classification of the glass dataset url: http://archive.ics.uci.edu/ml/datasets/Glass+Identification

The implementation of the Autoencoder was directly copied from https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1, with minor changes.

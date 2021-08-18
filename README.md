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
model/model.pth - the trainned Autoencoder model <br>
extra/data_exploration_plots.ipynd - the jupyter notebook file for plotting <br>
./alcoholism - contained dataset and references paper for the dataset <br>

## Procedures
casper_main.py is used for specifying Casper neural network and for training the network, it does not run on its own as it requires dataset as input,
the entire project is done by these steps using these files:
1. draw_subject_for_model_selection.py  for randomly drawing 12 subjects for model selection (redrawn if the 12 subject is extremely unbalanced)
2. determining the topology of the Casper using model_selection.py
3. train the autoencoder using autoencoder.py
4. run GA_cross_subject and GA_within_subject for determining the hyperparameter for the task
5. run within_subject_train_test.py for within subject testing
6. run cross_subject_train-test.py for cross subject testing

### Regarding the result of the model please directly run 5, 6 <br/>
within_subject_train_test.py for within subject testing <br/>
cross_subject_train-test.py for cross subject testing

## Testing result
#### Cross subject
Testing Accuracy mean: 71%

#### Within subject
Testing Accuracy: 91%  

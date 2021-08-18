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

###Regarding the result of the model please directly run 5, 6 <br/>
within_subject_train_test.py for within subject testing <br/>
cross_subject_train-test.py for cross subject testing

## Testing result
### Cross subject
sensitivity  mean:  0.6271697052500464
sensitivity  median:  0.6410256624221802
sensitivity  sd:  0.20761186370608364
accuracy  mean:  69.88162683939373
accuracy  median:  72.39709443099274
accuracy  sd:  11.80363301719955
loss  mean:  0.5397271405566822
loss  median:  0.5423167943954468
loss  sd:  0.01467593168671243
num_epoch  mean:  304.54545454545456
num_epoch  median:  250.0
num_epoch  sd:  93.41987329938274
num_neuron  mean:  1.2727272727272727
num_neuron  median:  1.0
num_neuron  sd:  0.46709936649691375
time  mean:  12.620529608293014
time  median:  10.496304512023926
time  sd:  4.022326460369308

### Within subject
Validation/Testing Accuracy: 87.88 %
sensitivity  mean:  0.8476358830928803
sensitivity  median:  0.8470421731472015
sensitivity  sd:  0.01926769680584754
accuracy  mean:  89.04809619238478
accuracy  median:  88.92785571142284
accuracy  sd:  0.9818162290823264
loss  mean:  0.11681840643286705
loss  median:  0.12064353749155998
loss  sd:  0.02765011124411904
num_epoch  mean:  5529.8
num_epoch  median:  6034.0
num_epoch  sd:  831.2397434622042
num_neuron  mean:  13.0
num_neuron  median:  14.0
num_neuron  sd:  1.632993161855452
time  mean:  147.24099092483522
time  median:  123.43196046352386
time  sd:  79.41430290424914

INFO ABOUT PROCESSED UCI EEG DATA SET:

In assignment 2, you will be provided with EEG images. Similar to assignment 1, 
please perform alcoholism classification. You can have a within-subject split 
or cross-subject split. More information can be found in the referenced paper.

The code of generating EEG images can be found at 
https://github.com/ShiyaLiu/EEG-feature-filter-and-disguising/blob/master/DataPreprocessing/eegtoimg.py

In uci_eeg_images.mat, we have:

"data" is an N x 32 x 32 x 3 tensor of EEG images where N is the number of trials,  

"subjectid" is a N-vector of subjectids since we have ~100 time series for each
of the ~120 subjects in the study.

"trialnum" is a N-vector of per-subject trial numbers. Each subject had ~120
trials in the original study but not in this data set.

"y_alcoholic" is a N-vector of binary labels, 1 if the subject is an alcoholic,
0 if the subject is not.

"y_stimulus" is a N-vector of categorical labels indicating which stimulus was
shown to the subject during the trial. In the study, there were five different
stimuli used, so the labels range from 1 to 5, indicating the following:

S2nomatch 1
S1obj 2
S2match 3
S2matcherr 4
S2nomatcherr 5

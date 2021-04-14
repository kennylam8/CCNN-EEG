INFO ABOUT PROCESSED UCI EEG DATA SET:

This is the preprocessed UCI EEG dataset, please perform alcoholism classification. 
You can have a within-subject split or cross-subject split. More information can be 
found in the referenced paper. 

In uci_eeg_features.mat, we have:

"data" is a N x A tensor of EEG measurements where N is the number of trials,  
A is the number of features (3 times of channel number). These features are the 
mean values of each frequency band of each EEG channel. They are concatenated
 in the order of (theta 1, theta 2, ..., alpha1, alpha2, ..., beta1, beta2,...)

"T" is a N-vector of time series lengths (all equal to 256).

"subjectid" is a N-vector of subjectids since we have ~100 time series for each
of the ~120 subjects in the study.

"trialnum" is a N-vector of per-subject trial numbers. Each subject had ~120
trials in the original study but not in this data set.

"y_alcoholic" is a N-vector of binary labels, 1 if the subject is an alcoholic,
0 if the subject is not.

"y_stimulus" is a N-vector of categorical labels indicating which stimulus was
shown to the subject during the trial. In the study, there were five different
stimuli used, so the labels range from 1 to 5, indicating the following:

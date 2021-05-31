"""
The Casper algorithm is implemented base on the sample code provided by COMP4660
Lab content, algorithm/codes including normalization, plotting loss diagram and plotting
error matrix and printing epoch/loss and accuracy has been copied directly from the COMP4660
Lab material. Some of which, as denoted in the lab material, is provided by UCI originally serve
as an example for the classfication of the glass dataset
url: http://archive.ics.uci.edu/ml/datasets/Glass+Identification

Other codes however, written entirely on my own (Kenny Lam).
"""

import statistics
import pandas as pd
import scipy.io
import random
import numpy as np
import torch

from draw_subject_for_model_selection import get_ms_subject_ids
from casper_main import main_casper
from casper_main import main_casper
from deap import algorithms, base, creator, tools
import random, operator
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx





mat = scipy.io.loadmat('alcoholism/uci_eeg_images_v2.mat')
data = mat["data"]
PRE = data.shape[0]
data = data.reshape(PRE, -1)
print(data.shape)
# df = pd.DataFrame.from_dict(data)
# print(df.shape)

from Autoencoder import AE
model_net = AE(input_shape=3072)
model_net.load_state_dict(torch.load("./model/model.pth"))
model_net.eval()

aaa2 = model_net.encode(torch.tensor(data).float())
aaa2 = np.array(aaa2)
df = pd.DataFrame(aaa2)

# safe the data to dataFrame for easier handling
df['y_stimulus'] = pd.DataFrame.from_dict(mat['y_stimulus']).T
df['subjectid'] = pd.DataFrame.from_dict(mat['subjectid']).T
# df['trialnum'] = pd.DataFrame.from_dict(mat['trialnum']).T
df['y_stimulus_1'] = (df['y_stimulus'] == 1).astype(int)
df['y_stimulus_2'] = (df['y_stimulus'] == 2).astype(int)
df['y_stimulus_3'] = (df['y_stimulus'] == 3).astype(int)
df['y_stimulus_4'] = (df['y_stimulus'] == 4).astype(int)
df['y_stimulus_5'] = (df['y_stimulus'] == 5).astype(int)
df['y_alcoholic'] = pd.DataFrame.from_dict(mat['y_alcoholic']).T
df = df[(df['subjectid'].isin(get_ms_subject_ids()))]
df = df.sample(frac=1)

# normalise training data by columns (commented out because not useful)
# for column in df:
#     if (column != "trialnum" and column != "subjectid" and column != "y_alcoholic" and column != "y_stimulus"):
#        df[column] = df.loc[:, [column]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#        df[column] = df.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())
print(df.shape)

def _10_fold_CV(data_frame, lst):
    print(lst)
    train_data = data_frame[~(data_frame['subjectid'].isin(lst))]
    test_data = data_frame[(data_frame['subjectid'].isin(lst))]
    train_data = train_data.drop(columns=['subjectid', 'y_stimulus'])
    test_data = test_data.drop(columns=['subjectid', 'y_stimulus'])


    n_features = train_data.shape[1] - 1
    train_input = train_data.iloc[:, :n_features]
    train_target = train_data.iloc[:, n_features]
    test_input = test_data.iloc[:, :n_features]
    test_target = test_data.iloc[:, n_features]
    return n_features, train_input, train_target, test_input, test_target, train_data


def ccs3(cutoff, k , cms,max_iter,constant_epoch):
    fold_test_set = list(df.subjectid.unique())
    random.shuffle(fold_test_set)
    train_lst = []
    accuracy = []
    print("CC",cutoff)
    for i in range(5):
        lss = fold_test_set[i*2:(i*2)+2]
        train_lst = train_lst+lss

        n_features, train_input, train_target, test_input, test_target, train_data = _10_fold_CV(df, lss)
        model = main_casper(n_features, train_input, train_target, test_input, test_target, cutoff, k, cms, max_iter,constant_epoch, "within-subject", train_data)
        accuracy.append(float(model[0]))
        print("ACC: ", model[0])
    return statistics.mean(accuracy)





# Deap setup
toolbox = base.Toolbox()

creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # -1 means that we wish to minimize the fitness
creator.create("Individual", list, fitness=creator.FitnessMax)

# cutoff, k,checkpoint_min_loss
toolbox.register("cutoff", np.random.uniform,low=0.0, high=0.5, size=1)
toolbox.register("k", np.random.uniform, low=0.0001, high=0.01, size=1)
toolbox.register("checkpoint_min_loss", np.random.uniform, low=0.001, high=0.1, size=1)
toolbox.register("max_iter", np.random.randint, low=4, high=15, size=1)
toolbox.register("constant_epoch", np.random.randint, low=25, high=75, size=1)

toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.cutoff, toolbox.k, toolbox.checkpoint_min_loss, toolbox.max_iter, toolbox.constant_epoch), )
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxUniform, indpb = 0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.3,low=[0.0,0.0001,0.001,4,25], up=[0.5,0.01,0.1,15,75], indpb=0.2)


def evaluation(hyperparameter):
    '''Evaluates an individual by converting it into
    a list of cities and passing that list to total_distance'''
    print("HYPER",hyperparameter)
    cutoff, k, c_m_loss, max_iter,constant_epoch = hyperparameter
    print("ASD",type(cutoff))
    cutoff = cutoff[0] if type(cutoff) == np.ndarray else cutoff
    k = k[0] if type(k) == np.ndarray else k
    c_m_loss = c_m_loss[0] if type(c_m_loss) == np.ndarray else c_m_loss

    max_iter = max_iter[0] if type(max_iter) == np.ndarray else max_iter
    constant_epoch = constant_epoch[0] if type(constant_epoch) == np.ndarray else constant_epoch

    out = ccs3(cutoff, k, c_m_loss, round(max_iter), round(constant_epoch))

    print("OO",out)
    return [out]

toolbox.register("evaluate", evaluation)

toolbox.register("select", tools.selTournament, tournsize=3)
pop = toolbox.population(n=50)
print(pop)
result, log = algorithms.eaSimple(pop, toolbox,
                             cxpb=0.8, mutpb=0.2,
                             ngen=200, verbose=False)


best_individual = tools.selBest(result, k=1)[0]
print()
print('Fitness of the best individual: ', evaluation(best_individual))
print('Best individual: ', (best_individual))
print()
print("RESULT: (RANKED):::")
print(result)
#



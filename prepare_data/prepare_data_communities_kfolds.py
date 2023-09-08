import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path


path = '../data/communities.data'
dataset = 'communities'
data = pd.read_csv(path, sep=',', header=None, \
        names=['state', 'county', 'community', 'communityname', 'fold', \
               'population', 'householdsize', 'racepctblack', 'racePctWhite', \
               'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29', \
               'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 'medIncome', \
               'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', \
               'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap', \
               'indianPerCap', 'AsianPerCap', 'OtherPerCap', 'HispPerCap', 'NumUnderPov', \
               'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', \
               'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', \
               'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', \
               'FemalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par', \
               'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', \
               'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5', \
               'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5', \
               'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', \
               'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous', \
               'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup', \
               'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant', \
               'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', \
               'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart', \
               'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ', 'RentMedian', 'RentHighQ', \
               'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', \
               'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState', \
               'PctSameHouse85', 'PctSameCity85', 'PctSameState85', 'LemasSwornFT', \
               'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', \
               'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', \
               'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', \
               'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', \
               'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', \
               'PolicOperBudg', 'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', \
               'LemasPctOfficDrugUn', 'PolicBudgPerPop', 'ViolentCrimesPerPop'])

print(data.head())
print(f'Number of instances loaded: {len(data)}')

# select columns
names = ['racepctblack', 'pctWInvInc', 'pctWPubAsst', 'NumUnderPov', \
         'PctPopUnderPov', 'PctUnemployed', 'MalePctDivorce', 'FemalePctDiv', \
         'TotalPctDiv', 'PersPerFam', 'PctKids2Par', 'PctYoungKids2Par', \
         'PctTeen2Par', 'NumIlleg', 'PctIlleg', 'PctPersOwnOccup', 'HousVacant', \
         'PctHousOwnOcc', 'PctVacantBoarded', 'NumInShelters', 'NumStreet', \
         'ViolentCrimesPerPop']

data = data[names]

# convert 'ViolentCrimesPerPop' to binary
data['ViolentCrimesPerPop'] = data['ViolentCrimesPerPop'].map(lambda x: 1 if x >= 0.7 else 0)
# convert 'racepctblack' to black and non-black
data['racepctblack'] = data['racepctblack'].map(lambda x: 'black' if x >= 0.06 else 'non-black')

data = data.to_numpy()

# categorical features
ind_cat = [0]
# numeric features
ind_numer = list(range(1, 21))
# labels
ind_label = [21]

data_cat = data[:, ind_cat]
enc = OrdinalEncoder()
data_cat = enc.fit_transform(data_cat)

data_numer = np.array(data[:, ind_numer], dtype=float)

features = np.concatenate([data_numer, data_cat], axis=1)
print(f'features type: {features.dtype}')

# split data into privileged features and normal features
# privileged features: racepctblack
# normal features: the rest
ind_pf = [20] # racepctblack
print(f'Number of privileged feature: {len(ind_pf)}')
ind_nf = [i for i in range(21) if i not in ind_pf]
print(f'Number of normal features: {len(ind_nf)}')

# we work on binary classification for now
# 1 (1) or 0 (0).
trans_label = {0: 0, 1: 1}
labels = np.array([trans_label[d] for d in data[:, ind_label].reshape(-1).tolist()], dtype='int')
print(f'labels shape: {labels.shape}')
uniques, counts = np.unique(labels, return_counts=True)
percentages = dict(zip(uniques, counts * 100 / len(labels)))
print(f'Percentages of each value in labels: {percentages}')

# split into 10 folds: 1 for test, 1 for valid, 8 for train
X = features
y = labels

num_folds = 10
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
print(f'Split into {skf.get_n_splits(X, y)} folds.')
folds = [test_index for  train_index, test_index in skf.split(X, y)]

for i in range(num_folds):
    test_index = folds[i]
    valid_index = folds[(i + 1) % num_folds]
    train_index = np.concatenate([folds[j] for j in range(num_folds) \
                                           if j not in [i, (i + 1) % num_folds]])
    Path(f'../data/{dataset}/fold{i}').mkdir(parents=True, exist_ok=True)
    # privileged and normal features
    dump_svmlight_file(X[train_index][:, ind_pf], y[train_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.pf.train')
    dump_svmlight_file(X[train_index][:, ind_nf], y[train_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.nf.train')

    dump_svmlight_file(X[valid_index][:, ind_pf], y[valid_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.pf.valid')
    dump_svmlight_file(X[valid_index][:, ind_nf], y[valid_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.nf.valid')

    dump_svmlight_file(X[test_index][:, ind_pf], y[test_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.pf.test')
    dump_svmlight_file(X[test_index][:, ind_nf], y[test_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.nf.test')
    # all features
    dump_svmlight_file(X[train_index], y[train_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.train')
    dump_svmlight_file(X[valid_index], y[valid_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.valid')
    dump_svmlight_file(X[test_index], y[test_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.test')


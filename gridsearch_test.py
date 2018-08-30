import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from evolutionary_search import EvolutionaryAlgorithmSearchCV
import os
# get some data
digits = load_digits()
X, y = digits.data, digits.target
print(X.shape)
# build a classifier
clf = RandomForestClassifier(n_estimators=20)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from

param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

totol_choice=1
for i,j in param_grid.items():
    totol_choice*=len(j)
print('total parameters choice',totol_choice)
# run randomized search
n_iter_search = 40
random_search = RandomizedSearchCV(clf, param_distributions=param_grid,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
#report(random_search.cv_results_)
print(random_search.best_score_)
# use a full grid over all parameters

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
#report(grid_search.cv_results_)
print(grid_search.best_score_)

# run evolutionary_
evolution_search = EvolutionaryAlgorithmSearchCV(estimator=clf,
                                   params=param_grid,
                                   #scoring="accuracy",

                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=4,)

start = time()
evolution_search.fit(X, y)
print("evolution_searchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
print(evolution_search.cv_results_)


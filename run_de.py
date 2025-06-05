"""
Differential Evolution as described in:
 Slowik, A., & Kwasnicka, H. (2020). Evolutionary algorithms and their applications to engineering problems. Neural Computing and Applications, 32, 12363-12379.
"""
import csv
import random
import time
import sys
import re

import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from deap import tools, base, creator

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
#from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from inconsistencyReport import extract_target

import coevolutionary
from classifier import evaluar_clasificador

from InconsistentExamplesMeasure import InconsistentExamplesMeasure
from MutualInformationMeasure import MutualInformationMeasure

from utils import discretizeDataset

def init_population(size, dimension, init_prob):
    return [[1 if random.random() < init_prob else 0 for _ in range(dimension)] for _ in range(size)]

def mutate_and_crossover(population, idx, F, CR):
    """Binary mutation and crossover"""
    idxs = list(range(len(population)))
    idxs.remove(idx)
    a, b, c = random.sample(idxs, 3)

    donor = []
    for i in range(len(population[0])):
        diff = population[b][i] ^ population[c][i]  # XOR difference
        gene = population[a][i] if random.random() > F else diff
        donor.append(gene)

    target = population[idx]
    trial = []
    for i in range(len(target)):
        trial.append(donor[i] if random.random() < CR else target[i])

    return trial

def de_feature_selection(toolbox, cv_partitions, clf):
    POP_SIZE = toolbox.population_size
    F = toolbox.p_mutation
    CR = toolbox.p_crossover
    GENERATIONS = toolbox.number_of_generations

    _, n_features = cv_partitions[0][0].shape
    population = init_population(POP_SIZE, n_features, toolbox.init_prob)
    fitnesses = [evaluar_clasificador(ind, cv_partitions, clf)[0] for ind in population]
    best_idx = np.argmax(fitnesses)
    best_individual = population[best_idx]
    best_fitness = fitnesses[best_idx]

    for gen in range(GENERATIONS):
        new_population = []

        for i in range(POP_SIZE):
            trial = mutate_and_crossover(population, i, F, CR)
            trial_fit = evaluar_clasificador(trial, cv_partitions, clf)[0]
            if trial_fit >= fitnesses[i]:
                new_population.append(trial)
                fitnesses[i] = trial_fit
            else:
                new_population.append(population[i])

        population = new_population
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fitness:
            best_individual = population[current_best_idx]
            best_fitness = fitnesses[current_best_idx]

        print(f"Gen {gen:02d} - Best Accuracy: {best_fitness:.4f} - Selected Features: {sum(best_individual)}")
        sys.stdout.flush()

    return ([i == 1 for i in best_individual], best_fitness)


def main(args):
    seed = 987612345
    random.seed(seed)
    np.random.seed(seed) # needed for some scikit-learn algorithms

    toolbox = base.Toolbox()
    toolbox.number_of_generations = args.generations
    toolbox.population_size       = args.population
    toolbox.p_crossover           = args.pcx
    toolbox.p_mutation            = args.pmut
    toolbox.init_prob             = args.init


    # Validation process parameters
    external_repeats  = args.repeats
    external_cv_folds = args.folds
    fitness_cv_folds  = args.ff
    train_fraction  = args.trainfrac
    if external_cv_folds > 1 and train_fraction > 0.0:
        raise Exception('CV folds and train-test-ratio are incompatible, only one can be set')
    elif external_cv_folds < 2 and not train_fraction > 0.0:
        raise Exception('CV folds or train-test-ratio are needed, one must be set')
    
    # Classifier parameters
    knn_neighbors = 0
    if re.match(r'^\d+nn$', args.candidate): # knn  ('1nn', '2nn', ... , 'Nnn')
        knn_neighbors = int(args.candidate[:-2]) 
        candidate = KNeighborsClassifier(knn_neighbors) #k-NN
    elif args.candidate == 'rf':
        candidate = RandomForestClassifier() # Bosque de árboles de clasificación [seed disrespectful]
    elif args.candidate == 'dtc':
        candidate = DecisionTreeClassifier()
    elif args.candidate == 'etc':
        candidate = ExtraTreesClassifier()
    elif args.candidate == 'xgbc':
        candidate = XGBClassifier()
    else:
        raise Exception('Unsupported candidate to optimize')

    # Parameter string for result file names
    candidate_name = str(candidate)[0:str(candidate).find('(')]
    test_param = f"t{args.trainfrac}" if train_fraction > 0.0 else f"CV{external_cv_folds}"
    params_string = f"_R{external_repeats}{test_param}ff{fitness_cv_folds}_" \
        + f"Ngen{toolbox.number_of_generations}-Nind{toolbox.population_size}-Pcx{str(toolbox.p_crossover)[2:]}-Pm{str(toolbox.p_mutation)[2:]}-Init{str(toolbox.init_prob)[2:]}_" \
        + f"{candidate_name}-K{knn_neighbors}"

    for dataset_name in args.dataset_name:
        if dataset_name[-4:] == '.csv':
            data = pandas.read_csv(dataset_name)
            target = extract_target(dataset_name)
            features = list(data.columns)
            if target not in features:
                print('Target "{}" not present in data'.format(target))
                return 1
            features.remove(target)
            X = data[features].values
            y = data[target].values
        else:
            dataset = sklearn.datasets.fetch_openml(name=dataset_name, as_frame=False)
            X = dataset['data']
            y = dataset['target']

        # Convert class a numeric for xgboost classifier
        if not isinstance(y[0], (int, float)):  # Si las etiquetas no son numéricas
            le = sklearn.preprocessing.LabelEncoder()
            y = le.fit_transform(y)

        sum_best_fitness = 0
        sum_nof = 0
        sum_validation_acc = 0
        final_filename = 'final-DE_' + dataset_name + params_string + '.csv'
        with open(final_filename, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Rep', 'Fold', 'Test', 'NoF', 'Validation', 'Individual'])

        # External validation repeated
        for ext_repeat_i in range(external_repeats):
            if train_fraction > 0.0:
                X_work, X_validation, y_work, y_validation = train_test_split(X, y, train_size=train_fraction, stratify=y)
                ext_cv_partitions = [ (X_work, X_validation, y_work, y_validation) ]
            else:
                ext_partitioner = StratifiedKFold(external_cv_folds, shuffle=True)
                ext_cv_partitions = []
                for train_indexes, test_indexes in ext_partitioner.split(X, y):
                    X_work, X_validation = X[train_indexes], X[test_indexes]
                    y_work, y_validation = y[train_indexes], y[test_indexes]
                    ext_cv_partitions.append( (X_work, X_validation, y_work, y_validation) )

            for ext_cv_i in range( len(ext_cv_partitions) ):
                X_work, X_validation, y_work, y_validation = ext_cv_partitions[ext_cv_i]

                # Cross-validation partitions for fitness evaluation
                partitioner = StratifiedKFold(fitness_cv_folds, shuffle=True)
                cv_partitions = []
                for train_indexes, test_indexes in partitioner.split(X_work, y_work):
                    X_train, X_test = X_work[train_indexes], X_work[test_indexes]
                    y_train, y_test = y_work[train_indexes], y_work[test_indexes]
                    cv_partitions.append( (X_train, X_test, y_train, y_test) )

                # Running DE
                start_time = time.process_time()
                best_ind, best_fitness = de_feature_selection(toolbox, cv_partitions, candidate)
                total_cpu_time = time.process_time() - start_time
                print('Total CPU time', total_cpu_time)

                # Final results
                with open(final_filename, 'a') as csvfile:
                    filewriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    if 'Measure' in candidate_name:
                        candidate_val = RandomForestClassifier()
                    else:
                        candidate_val = candidate
                    validation_acc = evaluar_clasificador(best_ind, [(X_work, X_validation, y_work, y_validation)], candidate_val)[0]

                    filewriter.writerow([ext_repeat_i, ext_cv_i, str(best_fitness), str(sum(best_ind)), str(validation_acc), 
                                         str(list(best_ind))])
                    sum_best_fitness += best_fitness
                    sum_nof += sum(best_ind)
                    sum_validation_acc += validation_acc

        # Averages
        n = external_repeats * len(ext_cv_partitions)
        with open(final_filename, 'a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Avg', '', sum_best_fitness / n, sum_nof / n, sum_validation_acc / n])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Differential Evolution FS experiments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset_name', nargs='+',
                    help='Data set to learn from, for example: filename_TARGET.csv or from OpenML ' + ' '.join(["OVA_Omentum", "amazon-commerce-reviews", "gina_agnostic", "Bioresponse", "Internet-Advertisements", "micro-mass", "eating", "hiva_agnostic", "CIFAR_10_small", "diabetes", "STL-10", "mouseType", "ovarianTumour", 'sa-heart']) )

    parser.add_argument('-r', '--repeats',     default=3,  help='Repetitions of full experiment', type=int)
    parser.add_argument('-f', '--folds',       default=10,  help='External validation number of CV folds', type=int)
    parser.add_argument('--ff', default=3, help='Fitness folds, internal CV folds', type=int)
    parser.add_argument('-t', '--trainfrac',  default=-1, help='Train-test with this train fraction instead of external CV, ex: 0.7', type=float)

    parser.add_argument('-g', '--generations', default=300,help='Number of generations to evolve', type=int)
    parser.add_argument('-p', '--population',  default=50, help='Number of individuals in each population', type=int)
    parser.add_argument('-c', '--pcx',       default=0.9, help='Probabilidad de cruce', type=float)
    parser.add_argument('-m', '--pmut',      default=0.25, help='Probabilidad de mutación', type=float)
    parser.add_argument('-i', '--init',      default=0.01, help='Probabilidad caracterisitcas iniciales', type=float)

    parser.add_argument('--candidate', type=str, default='3nn', help='Candidate to optimize: Nnn, rf, dtc, etc, xgbc')
    args = parser.parse_args()

    main(args)
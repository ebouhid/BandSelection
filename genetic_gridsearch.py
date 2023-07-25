import random
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# Code from genetic_alg.py adapted to include grid search for SVM parameters
N_JOBS = 15

# Get composition from command line
composition = sys.argv[1]

# Parameter grid
param_grid = {
    'C': np.logspace(-3, 4, 10),
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [1, 2, 3, 4, 5],
    'gamma': np.logspace(-3, 4, 10)
}

print(f'Param Grid:')
print(param_grid)


def generate_individual(num_features):
    individual = [random.randint(0, 1) for _ in range(num_features)]
    while sum(individual) == 0:
        individual = [random.randint(0, 1) for _ in range(num_features)]
    return individual


def generate_population(population_size, num_features):
    return [generate_individual(num_features) for _ in range(population_size)]


def calculate_fitness(individual, X_train, X_test, y_train, y_test):
    """
    Calculate the fitness of an individual by training an SVM classifier and
    evaluating its performance using accuracy.
    In this code, a grid search is performed to find the best SVM parameters.
    """
    selected_bands = list(np.nonzero(individual)[0])

    X_train_sel = [
        segment[selected_bands, :, :].reshape(-1) for segment in X_train
    ]
    X_test_sel = [
        segment[selected_bands, :, :].reshape(-1) for segment in X_test
    ]

    clf = SVC(class_weight='balanced')
    grid_search = GridSearchCV(clf,
                               param_grid,
                               cv=3,
                               scoring='balanced_accuracy',
                               n_jobs=N_JOBS)
    grid_search.fit(X_train_sel, y_train)

    best_svc = grid_search.best_estimator_
    y_pred = best_svc.predict(X_test_sel)

    return balanced_accuracy_score(y_test, y_pred)


def evaluate_population(population, X_train, X_test, y_train, y_test):
    return [
        calculate_fitness(individual, X_train, X_test, y_train, y_test)
        for individual in population
    ]


def crossover(parents, offspring_size):
    """
    Perform crossover to create 'offspring_size' new individuals.
    """
    offspring = []

    for _ in range(offspring_size):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)

        half = len(parent1) // 2
        offspring.append(parent1[:half] + parent2[half:])

    return offspring


def mutation(offspring):
    """
    Perform mutation by flipping a random bit in each individual.
    """
    mutated_offspring = []

    for individual in offspring:
        index = random.randint(0, len(individual) - 1)
        individual[index] = 1 - individual[index]
        if sum(individual) == 0:
            random_band = np.random.randint(0, 7, 3)
            for a in random_band:
                individual[a] = 1
        mutated_offspring.append(individual)

    return mutated_offspring


def select_individuals(population, scores, num_parents):
    sorted_indices = sorted(range(len(scores)),
                            key=lambda k: scores[k],
                            reverse=True)

    selected_individuals = [
        population[i] for i in sorted_indices[:num_parents]
    ]

    return crossover(selected_individuals, num_parents)


def estimate_distribution(selected_individuals, num_features):
    distribution = np.zeros(num_features)
    for individual in selected_individuals:
        distribution += np.array(individual)
    return distribution / len(selected_individuals)


def generate_offspring(parents, num_offspring, distribution, inf_lim, sup_lim):
    offspring = []
    for _ in range(num_offspring):
        individual = [int(inf_lim < prob < sup_lim) for prob in distribution]
        if sum(individual) == 0:
            random_band = np.random.randint(0, 7, 3)
            for a in random_band:
                individual[a] = 1
        offspring.append(individual)
    return mutation(offspring)


def genetic_algorithm(X_train, X_test, y_train, y_test, population_size,
                      num_generations, num_parents, num_offspring, inf_lim,
                      sup_lim):
    num_features = X_train[0].shape[0]
    population = generate_population(population_size, num_features)

    loop = tqdm(range(num_generations))

    for generation in loop:
        # print(f"Generation {generation + 1 :02d}")

        # Evaluate fitness
        scores = evaluate_population(population, X_train, X_test, y_train,
                                     y_test)

        # Select parents
        parents = select_individuals(population, scores, num_parents)

        distribution = estimate_distribution(parents, num_features)
        offspring = generate_offspring(parents, num_offspring, distribution,
                                       inf_lim, sup_lim)

        # Replace the population with offspring
        population = parents + offspring

    # Get the final fitness scores
    scores_df = pd.DataFrame()
    scores = evaluate_population(population, X_train, X_test, y_train, y_test)
    scores_df['Individual'] = population
    scores_df['Val accuracy'] = scores

    # Select the best individuals
    final_best_individuals = scores_df.sort_values(by='Val accuracy',
                                                   ascending=False)

    return final_best_individuals


# Loading dataset
X_all = []
y_all = []
for path in glob.glob(f'data/dataset_v3-{composition}/forest/*'):
    X_all.append(np.load(path))
    y_all.append(0)

for path in glob.glob(f'data/dataset_v3-{composition}/non_forest/*'):
    X_all.append(np.load(path))
    y_all.append(1)

# perform split
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5)

# call the genetic algorithm
num_best = 10
population_size = 80
num_generations = 30
num_parents = 10
num_offspring = 20
inf_lim = 0.25
sup_lim = 0.85

results = genetic_algorithm(X_train, X_val, y_train, y_val, population_size,
                            num_generations, num_parents, num_offspring,
                            inf_lim, sup_lim)

results['Test accuracy'] = results['Individual'].apply(
    lambda ind: calculate_fitness(ind, X_train, X_test, y_train, y_test))

print(results.sort_values(by='Test accuracy', ascending=False).head(num_best))

results.to_csv(f'results/genetic_gridsearch-{composition}.csv', index=False)
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import pandas as pd


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
    """
    # print(f'individual: {individual}')
    selected_bands = list(np.nonzero(individual)[0])

    X_train_sel = [
        segment[selected_bands, :, :].reshape(-1) for segment in X_train
    ]
    # print(f'X_train_sel[0].shape: {X_train_sel[0].shape}')
    X_test_sel = [
        segment[selected_bands, :, :].reshape(-1) for segment in X_test
    ]

    clf = SVC()
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_test_sel)
    return accuracy_score(y_test, y_pred)


def evaluate_population(population, X_train, X_test, y_train, y_test):
    return [
        calculate_fitness(individual, X_train, X_test, y_train, y_test)
        for individual in population
    ]


def select_individuals(population, scores, num_parents):
    sorted_indices = sorted(range(len(scores)),
                            key=lambda k: scores[k],
                            reverse=True)
    return [population[i] for i in sorted_indices[:num_parents]]


def estimate_distribution(selected_individuals, num_features):
    distribution = np.zeros(num_features)
    for individual in selected_individuals:
        distribution += np.array(individual)
    return distribution / len(selected_individuals)


def generate_offspring(parents, num_offspring, distribution):
    offspring = []
    for _ in range(num_offspring):
        individual = [int(np.random.uniform() < prob) for prob in distribution]
        offspring.append(individual)
    return offspring


def genetic_algorithm(X_train,
                      X_test,
                      y_train,
                      y_test,
                      population_size,
                      num_generations,
                      num_parents,
                      num_offspring,
                      num_best=5):
    num_features = X_train[0].shape[0]
    population = generate_population(population_size, num_features)

    for generation in range(num_generations):
        print("Generation", generation + 1)

        # Evaluate fitness
        scores = evaluate_population(population, X_train, X_test, y_train,
                                     y_test)

        # Select parents
        parents = select_individuals(population, scores, num_parents)

        distribution = estimate_distribution(parents, num_features)
        offspring = generate_offspring(parents, num_offspring, distribution)

        # Replace the population with offspring
        population = parents + offspring

    # Get the final fitness scores
    scores = evaluate_population(population, X_train, X_test, y_train, y_test)

    # print(f'scores: {scores}')

    # Select the best individuals
    best_indices = sorted(range(len(scores)),
                          key=lambda k: scores[k],
                          reverse=True)[:num_best]
    best_individuals = [population[i] for i in best_indices]
    best_fitnesses = [scores[i] for i in best_indices]

    return best_individuals, best_fitnesses


# Loading dataset
X_all = []
y_all = []
for path in glob.glob('data/dataset_v2/forest/*'):
    X_all.append(np.load(path))
    y_all.append(0)

for path in glob.glob('data/dataset_v2/non_forest/*'):
    X_all.append(np.load(path))
    y_all.append(1)

# perform split
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5)

# call the genetic algorithm
num_best = 10
population_size = 20
num_generations = 20
num_parents = 10
num_offspring = 5

best_individuals, best_fitnesses = genetic_algorithm(X_train, X_val, y_train,
                                                     y_val, population_size,
                                                     num_generations,
                                                     num_parents,
                                                     num_offspring, num_best)

data = []
for i in range(num_best):
    test_result = calculate_fitness(best_individuals[i], X_train, X_test,
                                    y_train, y_test)
    bands = np.array(np.nonzero(best_individuals[i])) + 1
    individual_str = ''.join(str(band) for band in bands)
    data.append({
        "Individual": individual_str,
        "Fitness": best_fitnesses[i],
        "Test acc": test_result
    })

data = pd.DataFrame.from_records(data)
print(data.sort_values(by='Test acc', ascending=False))
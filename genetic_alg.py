import random
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import glob


def generate_individual(num_features):
    individual = [random.randint(0, 1) for _ in range(num_features)]
    while sum(individual) == 0:
        individual = [random.randint(0, 1) for _ in range(num_features)]
    return individual


def generate_population(population_size, num_features):
    return [generate_individual(num_features) for _ in range(population_size)]


def calculate_fitness(individual, X_train, X_test, y_train, y_test):
    selected_features = [i for i, val in enumerate(individual) if val == 1]
    X_train_sel = X_train[:, selected_features]
    X_test_sel = X_test[:, selected_features]

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
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
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


def genetic_algorithm(X_train, X_test, y_train, y_test, population_size,
                      num_generations, num_parents, num_offspring):
    num_features = X_train.shape[1]
    population = generate_population(population_size, num_features)

    for generation in range(num_generations):
        scores = evaluate_population(population, X_train, X_test, y_train, y_test)
        parents = select_individuals(population, scores, num_parents)
        distribution = estimate_distribution(parents, num_features)
        offspring = generate_offspring(parents, num_offspring, distribution)
        population = parents + offspring

    scores = evaluate_population(population, X_train, X_test, y_train, y_test)
    best_individual_index = np.argmax(scores)
    best_individual = population[best_individual_index]
    best_fitness = scores[best_individual_index]

    return best_individual, best_fitness


# Loading dataset
X_all = []
y_all = []
for path in glob.glob('dataset_v2/forest/*'):
    X_all.append(np.load(path))
    y_all.append(0)

for path in glob.glob('dataset_v2/non_forest/*'):
    X_all.append(np.load(path))
    y_all.append(1)

# Perform train-test split
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5)

# Call the genetic algorithm
population_size = 50
num_generations = 10
num

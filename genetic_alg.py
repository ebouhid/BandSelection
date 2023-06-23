import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import glob
import numpy as np


def generate_individual(num_features):
    """
    Generate a random individual
    """
    individual = [random.randint(0, 1) for _ in range(num_features)]
    while sum(individual) == 0:
        individual = [random.randint(0, 1) for _ in range(num_features)]

    return individual


def generate_population(population_size, num_features):
    """
    Generate a population of 'population_size' individuals.
    """
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
    """
    Evaluate the fitness of each individual in the population.
    """
    return [
        calculate_fitness(individual, X_train, X_test, y_train, y_test)
        for individual in population
    ]


def select_individuals(population, scores, num_parents):
    """
    Select the top 'num_parents' individuals based on their fitness scores.
    """
    selected_parents = []
    sorted_indices = sorted(range(len(scores)),
                            key=lambda k: scores[k],
                            reverse=True)

    for i in range(num_parents):
        selected_parents.append(population[sorted_indices[i]])

    return selected_parents


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
        mutated_offspring.append(individual)

    return mutated_offspring


def genetic_algorithm(X_train, X_test, y_train, y_test, population_size,
                      num_generations, num_parents, num_offspring):
    num_features = X_train[0].shape[0]
    population = generate_population(population_size, num_features)

    for generation in range(num_generations):
        print("Generation", generation + 1)

        # Evaluate fitness
        scores = evaluate_population(population, X_train, X_test, y_train,
                                     y_test)

        # Select parents
        parents = select_individuals(population, scores, num_parents)

        # Create offspring through crossover
        offspring = crossover(parents, num_offspring)

        # Perform mutation
        offspring = mutation(offspring)

        # Replace the population with offspring
        population = parents + offspring

    # Get the final fitness scores
    scores = evaluate_population(population, X_train, X_test, y_train, y_test)

    # print(f'scores: {scores}')

    # Select the best individual
    best_individual_index = max(range(len(scores)), key=lambda k: scores[k])
    best_individual = population[best_individual_index]
    best_fitness = scores[best_individual_index]

    return best_individual, best_fitness


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
population_size = 50
num_generations = 10
num_parents = 15
num_offspring = 20

best_individual, best_fitness = genetic_algorithm(X_train, X_val, y_train,
                                                  y_val, population_size,
                                                  num_generations, num_parents,
                                                  num_offspring)

print(f"Best individual: {best_individual}")
print(f"With fitness: {best_fitness :.3f}")

print(f'\nCalculating fitness for {best_individual}')
test_result = calculate_fitness(best_individual, X_train, X_test, y_train,
                                y_test)
print(f'test result = {test_result :.3f}')
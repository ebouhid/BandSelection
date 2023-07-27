import random
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


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

    clf = SVC(C=100, gamma='scale', kernel='rbf', class_weight='balanced')
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_test_sel)
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
            individual = generate_individual(len(individual))
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


def generate_offspring(parents, num_offspring, distribution, inf_lim, sup_lim,
                       unique_individuals):
    offspring = []
    for _ in range(num_offspring):
        individual = [int(inf_lim < prob < sup_lim) for prob in distribution]
        if sum(individual) == 0:
            individual = generate_individual(len(individual))

        individual_tuple = tuple(individual)
        if individual_tuple not in unique_individuals:
            offspring.append(individual)
            unique_individuals.add(individual_tuple)

    return mutation(offspring)


def genetic_algorithm(X_train, X_test, y_train, y_test, population_size,
                      num_generations, num_parents, num_offspring, inf_lim,
                      sup_lim):
    num_features = X_train[0].shape[0]
    population = generate_population(population_size, num_features)

    loop = tqdm(range(num_generations))

    # Keep track of unique individuals across generations
    unique_individuals = set(map(tuple, population))

    for generation in loop:
        # Evaluate fitness
        scores = evaluate_population(population, X_train, X_test, y_train,
                                     y_test)

        # Select parents
        parents = select_individuals(population, scores, num_parents)

        distribution = estimate_distribution(parents, num_features)

        # Generate offspring while ensuring uniqueness
        offspring = generate_offspring(parents, num_offspring, distribution,
                                       inf_lim, sup_lim, unique_individuals)

        # Replace the population with parents and unique offspring
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
for path in glob.glob('data/dataset_v3-467/forest/*'):
    X_all.append(np.load(path))
    y_all.append(0)

for path in glob.glob('data/dataset_v3-467/non_forest/*'):
    X_all.append(np.load(path))
    y_all.append(1)

# perform split
X_train, X_val, y_train, y_val = train_test_split(X_all,
                                                  y_all,
                                                  test_size=0.3,
                                                  random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val,
                                                y_val,
                                                test_size=0.5,
                                                random_state=43)

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

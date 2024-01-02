import random
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import logging

# Get command line arguments
exp_name = str(sys.argv[1])
seed = int(sys.argv[2])
gpu_id = int(sys.argv[3])

# Set random seed
np.random.seed(seed)


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
    selected_bands = list(np.nonzero(individual)[0])

    X_train_sel = [
        segment[selected_bands, :, :].reshape(-1) for segment in X_train
    ]
    X_test_sel = [
        segment[selected_bands, :, :].reshape(-1) for segment in X_test
    ]

    clf = SVC(C=100,
              kernel='rbf',
              class_weight='balanced',
              random_state=seed,
              )
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_test_sel)
    return balanced_accuracy_score(y_test, y_pred)


def evaluate_population(population, X_train, X_test, y_train, y_test, generation=None):
    msg = f'Generation {generation}' if generation else 'First Evaluation'
    return [
        calculate_fitness(individual, X_train, X_test, y_train, y_test)
        for individual in tqdm(population, desc=msg)
    ]


def crossover(parents, offspring_size, xover_prob):
    """
    Perform crossover to create 'offspring_size' new individuals.
    """
    offspring = []

    for _ in range(offspring_size):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)

        half = len(parent1) // 2
        spawn = parent1[:half] + parent2[half:]
        if random.random() < xover_prob:
            offspring.append(spawn)
        else:
            offspring.append(random.choice([parent1, parent2]))

    return offspring


def mutation(offspring, mut_prob):
    """
    Perform mutation by flipping a random bit in each individual.
    """
    mutated_offspring = []

    for individual in offspring:
        if random.random() < mut_prob:
            index = random.randint(0, len(individual) - 1)
            individual[index] = 1 - individual[index]
            # preventing null individuals (all zeroes)
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

    return selected_individuals


def estimate_distribution(selected_individuals, num_features):
    distribution = np.zeros(num_features)
    for individual in selected_individuals:
        distribution += np.array(individual)
    return distribution / len(selected_individuals)


def generate_offspring(parents, num_offspring, distribution, inf_lim, sup_lim,
                       mut_prob, xover_prob):
    offspring = []
    for _ in range(num_offspring):
        for i in range(len(distribution)):
            distribution[i] = distribution[i] if (
                inf_lim < distribution[i] < sup_lim) else 0
        individual = [
            random.choices([0, 1], weights=[1 - p, p])[0] for p in distribution
        ]
        # preventing null individual (all zeroes)
        if sum(individual) == 0:
            individual = generate_individual(len(individual))
        offspring.append(individual)
    if xover_prob > 0:
        offspring = crossover(offspring, num_offspring, xover_prob)
    
    if mut_prob > 0:
        offspring = mutation(offspring, mut_prob)
    return offspring


def genetic_algorithm(X_train, X_test, y_train, y_test, population_size,
                      num_generations, num_parents, num_offspring, inf_lim,
                      sup_lim, mut_prob, xover_prob):
    num_features = X_train[0].shape[0]
    population = generate_population(population_size, num_features)

    logging.basicConfig(filename=f'results/{exp_name}/logfile-{seed}.out',
                        level=logging.INFO)
    # Logging experiment settings
    logging.info(f'Seed: {seed}')
    logging.info(f'Population size: {population_size}')
    logging.info(f'Number of generations: {num_generations}')
    logging.info(f'Number of parents: {num_parents}')
    logging.info(f'Number of offspring: {num_offspring}')
    logging.info(f'Inferior limit: {inf_lim}')
    logging.info(f'Superior limit: {sup_lim}')
    logging.info(f'Mutation probability: {mut_prob}')
    logging.info(f'Crossover probability: {xover_prob}')

    loop = range(num_generations)

    for generation in loop:
        # Evaluate fitness
        scores = evaluate_population(population, X_train, X_test, y_train,
                                     y_test, generation)

        # Log the 3 best individuals
        scores_df = pd.DataFrame()
        scores_df['Individual'] = population
        scores_df['Val accuracy'] = scores
        scores_df = scores_df.sort_values(by='Val accuracy', ascending=False)
        logging.info(f'Generation {generation}')
        logging.info(scores_df.head(3))
        logging.info('')

        # Select parents
        parents = select_individuals(population, scores, num_parents)

        distribution = estimate_distribution(parents, num_features)
        offspring = generate_offspring(parents, num_offspring, distribution,
                                       inf_lim, sup_lim, mut_prob, xover_prob)

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

if __name__ == '__main__':
    # Loading dataset
    # Training set
    X_train = []
    y_train = []
    for filename in glob.glob('data/classification_dataset/train/forest/*'):
        X_train.append(np.load(filename))
        y_train.append(0)
    for filename in glob.glob('data/classification_dataset/train/non_forest/*'):
        X_train.append(np.load(filename))
        y_train.append(1)

    # Validation set
    X_val = []
    y_val = []
    for filename in glob.glob('data/classification_dataset/val/forest/*'):
        X_val.append(np.load(filename))
        y_val.append(0)
    for filename in glob.glob('data/classification_dataset/val/non_forest/*'):
        X_val.append(np.load(filename))
        y_val.append(1)
    
    # Test set
    X_test = []
    y_test = []
    for filename in glob.glob('data/classification_dataset/test/forest/*'):
        X_test.append(np.load(filename))
        y_test.append(0)
    for filename in glob.glob('data/classification_dataset/test/non_forest/*'):
        X_test.append(np.load(filename))
        y_test.append(1)

    # call the genetic algorithm
    num_best = 10
    population_size = 10
    num_generations = 10
    num_parents = 5
    num_offspring = 5
    inf_lim = 0.00
    sup_lim = 1.00
    mut_prob = 0
    xover_prob = 0

    results = genetic_algorithm(X_train, X_val, y_train, y_val, population_size,
                                num_generations, num_parents, num_offspring,
                                inf_lim, sup_lim, mut_prob, xover_prob)

    results['Test accuracy'] = results['Individual'].apply(
        lambda ind: calculate_fitness(ind, X_train, X_test, y_train, y_test))

    print(results.sort_values(by='Test accuracy', ascending=False).head(num_best))

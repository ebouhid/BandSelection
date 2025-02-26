import random
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import logging
import argparse
from umda_dataset import UMDADataset
from concurrent.futures import ThreadPoolExecutor


def generate_individual(num_features):
    individual = [random.randint(0, 1) for _ in range(num_features)]
    while sum(individual) == 0:
        individual = [random.randint(0, 1) for _ in range(num_features)]
    return individual


def generate_population(population_size, num_features):
    # Generating a set of unique individuals
    population = set()
    while len(population) < population_size:
        individual = tuple(generate_individual(num_features))  # Note the conversion to tuple
        population.add(individual)
    return list(population)  # Convert back to list for further processing

def generate_offspring(parents, num_offspring, distribution, inf_lim, sup_lim):
    offspring = set()
    while len(offspring) < num_offspring:
        for i in range(len(distribution)):
            distribution[i] = distribution[i] if (inf_lim <= distribution[i] <= sup_lim) else 0
        individual = tuple([
            random.choices([0, 1], weights=[1 - p, p])[0] for p in distribution
        ])
        # Preventing null individual (all zeroes)
        if sum(individual) == 0:
            individual = tuple(generate_individual(len(individual)))
        offspring.add(individual)
    return list(offspring)



def calculate_fitness(individual, X_train, X_test, y_train, y_test, seed):
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

    logging.info(f'Evaluating individual: {individual}')

    clf = SVC(C=100,
              kernel='rbf',
              class_weight='balanced',
              random_state=seed,
              )
    clf.fit(X_train_sel, y_train)
    logging.info(f"Fit classifier with {len(selected_bands)} bands (individual {individual})")
    y_pred = clf.predict(X_test_sel)
    logging.info(f"Predicted test set (individual {individual})")
    return balanced_accuracy_score(y_test, y_pred)


def evaluate_population(population, X_train, X_test, y_train, y_test, seed, generation=None):
    if generation is not None:
        if generation == 0:
            msg = "Evaluating initial population"
        else:
            msg = f"Generation {generation}"
    else:
        msg = "Final evaluation"

    def evaluate_individual(individual):
        return calculate_fitness(individual, X_train, X_test, y_train, y_test, seed)

    # Using ThreadPoolExecutor to parallelize evaluations
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(evaluate_individual, population), total=len(population), desc=msg))

    return results



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


def umda(X_train, X_test, y_train, y_test, population_size,
         num_generations, num_parents, num_offspring, inf_lim,
         sup_lim, seed):
    num_features = X_train[0].shape[0]
    population = generate_population(population_size, num_features)

    # Logging experiment settings
    logging.info(f'Seed: {seed}')
    logging.info(f'Number of features (bands): {num_features}')
    logging.info(f'Population size: {population_size}')
    logging.info(f'Number of generations: {num_generations}')
    logging.info(f'Number of parents: {num_parents}')
    logging.info(f'Number of offspring: {num_offspring}')
    logging.info(f'Inferior limit: {inf_lim}')
    logging.info(f'Superior limit: {sup_lim}')

    loop = range(num_generations)

    for generation in loop:
        # Evaluate fitness
        scores = evaluate_population(population, X_train, X_test, y_train,
                                     y_test, seed, generation)

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
                                       inf_lim, sup_lim)

        # Replace the population with offspring
        population = parents + offspring

    # Get the final fitness scores
    scores_df = pd.DataFrame()
    scores = evaluate_population(
        population, X_train, X_test, y_train, y_test, seed=seed)
    scores_df['Individual'] = population
    scores_df['Val accuracy'] = scores

    # Select the best individuals
    final_best_individuals = scores_df.sort_values(by='Val accuracy',
                                                   ascending=False)

    return final_best_individuals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default='lsat')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    exp_name = args.exp_name
    seed = args.seed

    # Set random seed
    np.random.seed(seed)
    random.seed(seed)

    # Set up logging
    logging.basicConfig(filename=f'results/{exp_name}/logfile-{seed}.out',
                        level=logging.INFO)

    # Defining regions
    train_regions = ["x01", "x02", "x06", "x09", "x10"]
    val_regions = ["x07", "x08"]
    test_regions = ["x03", "x04"]

    # Log regions
    logging.info(f'Train regions: {train_regions}')
    logging.info(f'Validation regions: {val_regions}')
    logging.info(f'Test regions: {test_regions}')

    # Loading datasets
    train_ds = UMDADataset(
        'data/classification_datasets/landsat_SLIC_puc_4000', train_regions)
    val_ds = UMDADataset(
        'data/classification_datasets/landsat_SLIC_puc_4000', val_regions)
    test_ds = UMDADataset(
        'data/classification_datasets/landsat_SLIC_puc_4000', test_regions)

    X_train, y_train = train_ds.get_set()
    X_val, y_val = val_ds.get_set()
    X_test, y_test = test_ds.get_set()

    # Call UMDA
    num_best = 15
    population_size = 60
    num_generations = 2
    num_parents = 10
    num_offspring = 50
    inf_lim = 1/8
    sup_lim = 7/8

    results = umda(X_train, X_val, y_train, y_val, population_size,
                   num_generations, num_parents, num_offspring,
                   inf_lim, sup_lim, args.seed)

    results['Test accuracy'] = results['Individual'].apply(
        lambda ind: calculate_fitness(ind, X_train, X_test, y_train, y_test, seed=seed))

    print(results.sort_values(by='Test accuracy', ascending=False).head(num_best))

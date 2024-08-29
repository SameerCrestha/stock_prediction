import numpy as np
from create_evaluate_model import create_model, evaluate_model, get_train_val_test_data

X_train, y_train, _, _, X_test, y_test = get_train_val_test_data()

# Define the fitness function (e.g., model accuracy)
def fitness_function(params):
    print(f"Training model with params: {params}")
    
    model = create_model(params, X_train, y_train)
    mae = evaluate_model(model, X_test, y_test)
    print(f"Got mae: {mae} \n")
    accuracy = -mae
    return accuracy

# Initialize the population
def initialize_population(pop_size, param_choices):
    population = []
    for _ in range(pop_size):
        chromosome = {}
        for key, values in param_choices.items():
            chosen_value = np.random.choice(values)
            chromosome[key] = int(chosen_value) if isinstance(chosen_value, np.integer) else chosen_value
        population.append(chromosome)
    return population

# Selection
def selection(population, fitness_scores):
    probabilities = fitness_scores / np.sum(fitness_scores)
    selected = np.random.choice(population, size=len(population), p=probabilities, replace=True)
    return selected

# Crossover
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child = {key: (parent1[key] if i < crossover_point else parent2[key]) for i, key in enumerate(parent1)}
    return child

# Mutation
def mutate(chromosome, param_choices, mutation_rate=0.01):
    if np.random.rand() < mutation_rate:
        param_to_mutate = np.random.choice(list(chromosome.keys()))
        possible_values = param_choices[param_to_mutate]
        new_value = np.random.choice([val for val in possible_values if val != chromosome[param_to_mutate]])
        chromosome[param_to_mutate] = int(new_value) if isinstance(new_value, np.integer) else new_value
    return chromosome

def genetic_algorithm(param_choices, pop_size=5, generations=5):
    population = initialize_population(pop_size, param_choices)

    for generation in range(generations):
        print(f"---------------Generation-{generation+1}---------------")
        fitness_scores = np.array([fitness_function(chromosome) for chromosome in population])
        selected_population = selection(population, fitness_scores)

        next_generation = []
        for i in range(0, pop_size, 2):
            parent1 = selected_population[i]
            if i + 1 < len(selected_population):
                parent2 = selected_population[i + 1]
            else:
                parent2 = selected_population[0]  # Fallback to the first individual or handle differently
            
            child1 = mutate(crossover(parent1, parent2), param_choices)
            child2 = mutate(crossover(parent2, parent1), param_choices)
            next_generation.extend([child1, child2])

        # If next_generation has more individuals than pop_size, trim it
        if len(next_generation) > pop_size:
            next_generation = next_generation[:pop_size]

        population = next_generation

        print(f'Generation {generation+1}, Best Fitness had mae of {-np.max(fitness_scores)}')

    best_chromosome = population[np.argmax(fitness_scores)]
    return best_chromosome

# Example usage with sets of possible values
param_choices = {
    'num_epochs': [20, 30, 50, 80],       # List of possible values
    'learning_rate': [0.01, 0.05, 0.1],   # List of possible values
    'hidden_size': [2, 16, 32, 64, 128],  # List of possible values
    'num_layers': [1, 2, 3]               # List of possible values
}

best_params = genetic_algorithm(param_choices,pop_size=10,generations=3)
print(f'\n ---------------Best Parameters: {best_params}')

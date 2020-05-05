#!/usr/bin/env python
###############################################################################
# Version: 1.1
# Last modified on: 3 April, 2016 
# Developers: Michael G. Epitropakis
#      email: m_(DOT)_epitropakis_(AT)_lancaster_(DOT)_ac_(DOT)_uk 
###############################################################################
from cec2013.cec2013 import *
import numpy as np
import random

POPU_SIZE = 10
KIDS_NUM = 10
RUN_REPEAT = 50
MUTATE_RATE = 0.5
TARGET_FUNC = 1
MAX_EVALUATE_COUNT = 10000
CROSSOVER_ALPHA = 0.6
MUTATE_GAMMA = 1
TOURNAMENT_RATE = 1
SHARING_RADIUS = 5
SHARING_SIGMA = 1

EVALUATE_COUNT = 0
DIM = 0
ub = 0
lb = 0

def init(ub,lb):
    population = np.zeros((POPU_SIZE,DIM))
    for i in range(POPU_SIZE):
        population[i] = lb + (ub - lb) * np.random.rand(1, DIM)
    return population

def crossover(A,B):
    kid = np.zeros(DIM)
    for i in range(DIM):
        kid[i] = CROSSOVER_ALPHA * A[i] + (1-CROSSOVER_ALPHA) * B[i]
    return kid

def mutate(kid):
    pob = random.random()
    if pob > MUTATE_RATE:
        return kid
    for i in range(DIM):
        kid[i] += np.random.standard_cauchy(1) * MUTATE_GAMMA
    return kid

def select(population,fitness):
    total_popu_size = population.shape[0]
    modified_fitness = fitness_share(population,fitness)
    win_table = np.zeros(total_popu_size)
    for i in range(total_popu_size):
        for j in range(0,int(total_popu_size * TOURNAMENT_RATE)):
            another = random.randint(0,total_popu_size-1)
            if( modified_fitness[i] > modified_fitness[another]):
                win_table[i] += 1
    order = np.argsort(win_table)
    sorted_popu = population[-order]
    sorted_fitness = fitness[-order]
    return sorted_popu[0:POPU_SIZE,:],sorted_fitness[0:POPU_SIZE]

def generate(population):
    kids = np.zeros((KIDS_NUM,DIM))
    kids_count = 0
    while kids_count < KIDS_NUM:
        kid = crossover(population[random.randint(0,population.shape[0]-1)],population[random.randint(0,population.shape[0]-1)])
        kid = mutate(kid)
        if is_invalid(kid):
            continue
        ## check if kid is exist in kids
        a = (kids == kid)
        duplicate = False
        for c in a:
            if c.all():
                duplicate = True
                break
        if not duplicate:
            kids[kids_count] = kid
            kids_count += 1

    return kids


def is_invalid(indiv):
    invalid = False
    for j in range(DIM):
        if indiv[j]>ub[j] or indiv[j]<lb[j]:
            invalid = True
            break
    return invalid

def fitness_share(population,fitness):
    new_fitness = np.zeros(fitness.shape)
    for i in range(population.shape[0]):
        sh = 0
        for j in range(population.shape[0]):
            if i != j:
                dis = distance(population[i],population[j])
                if dis < SHARING_RADIUS:
                    sh += 1 - math.pow((dis/SHARING_RADIUS),SHARING_SIGMA)
        if sh == 0:
            sh = 1
        new_fitness[i] = fitness[i] / sh
        print("----point: ",population[i])
        print("before sharing: ",fitness[i])
        print("sh: ",sh)
    return new_fitness

def distance(indiv1, indiv2):
    sum = 0
    for i in range(DIM):
        sum += math.pow(indiv1[i] - indiv2[i],2)
    return math.pow(sum,0.5)


def evaluate(f,x):
    global EVALUATE_COUNT
    size = x.shape[0]
    fitness = np.zeros(size)
    for i in range(size):
        fitness[i] = f.evaluate(x[i])
        EVALUATE_COUNT += 1
    return fitness

def main():
    print (70*"=")
    # Demonstration of all functions
    for i in range(1,21):
        # Create function
        f = CEC2013(i)

        # Create position vectors
        x = np.ones( f.get_dimension() )

        # Evaluate :-)
        value = f.evaluate(x)
        print ("f", i, "(", x, ") = ", f.evaluate(x))

    print (70*"=")
    # Demonstration of using how_many_goptima function
    for i in range(1,21):
        # Create function
        f = CEC2013(i)
        dim = f.get_dimension()

        # Create population of position vectors
        pop_size = 10
        X = np.zeros( (pop_size, dim) )
        ub =np.zeros( dim )
        lb =np.zeros( dim )
        # Get lower, upper bounds
        for k in range(dim):
            ub[k] = f.get_ubound(k)
            lb[k] = f.get_lbound(k)
        ub = [15]
        lb = [10]
        # Create population within bounds
        fitness = np.zeros( pop_size )
        for j in range(pop_size):
            X[j] = lb + (ub - lb) * np.random.rand( 1 , dim )
            fitness[j] = f.evaluate(X[j])

        # Calculate how many global optima are in the population
        accuracy = 0.001
        count, seeds = how_many_goptima(X, f, accuracy)
        print ("In the current population there exist", count, "global optimizers.")
        print ("Global optimizers:", seeds)

    print (70*"=")



if __name__ == "__main__":
    ## intialization
    file = open("log.txt","w")
    f = CEC2013(TARGET_FUNC)
    DIM = f.get_dimension()
    ub = np.zeros(DIM)
    lb = np.zeros(DIM)
    # Get lower, upper bounds
    for k in range(DIM):
        ub[k] = f.get_ubound(k)
        lb[k] = f.get_lbound(k)

    population = init(ub,lb)
    fitness = evaluate(f,population)
    print("init population: ",population)
    print("init fitness: ", fitness)
    print(30*"*", ", init over")
    ## iteration
    while(EVALUATE_COUNT < MAX_EVALUATE_COUNT):
        print("evaluate count: ",EVALUATE_COUNT)
        kids = generate(population)
        print("kids : ", kids)
        kids_fitness = evaluate(f, kids)
        print("kids_fitness : ", kids_fitness)
        total_popu = np.r_[population,kids]
        total_fitness = np.r_[fitness,kids_fitness]
        population,fitness = select(total_popu,total_fitness)
        print(np.mean(fitness))
        accuracy = 0.1
        for i in range(population.shape[0]):
            file.write(str(population[i][0]) + " " + str(fitness[i])+ " " + str(EVALUATE_COUNT) + '\n')
        count, seeds = how_many_goptima(population, f, accuracy)
        print("In the current population there exist", count, "global optimizers.")
        print("Global optimizers:", seeds)
        print(70*"=")
    file.close()
    accuracy = 1
    count, seeds = how_many_goptima(population, f, accuracy)
    print("In the current population there exist", count, "global optimizers.")
    print("Global optimizers:", seeds)


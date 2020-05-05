
from cec2013.cec2013 import *
import numpy as np
import random

POPU_SIZE = 1000
KIDS_NUM = 1000
RUN_REPEAT = 50
MUTATE_RATE = 0.6
TARGET_FUNC = 6
MAX_EVALUATE_COUNT = 200000
CROSSOVER_ALPHA = 0.6
MUTATE_GAMMA = 2
TOURNAMENT_RATE = 0.4
SHARING_RADIUS = 6
SHARING_SIGMA = 0.5
SHARING_BETA = 2
LOCAL_SEARCH_LENGTH_RATE = 0.05

EVALUATE_COUNT = 0
DIM = 0
ub = 0
lb = 0
f = None
local_search_improve_count = 0

def local_search1(indiv,fitness):
    global EVALUATE_COUNT,local_search_improve_count
    step_size = LOCAL_SEARCH_LENGTH_RATE * (1 - tanh(EVALUATE_COUNT / MAX_EVALUATE_COUNT * 3))
    best_indiv = indiv
    best_fitness = fitness
    for j in range(indiv.shape[0]):
        new_indiv = indiv
        new_indiv[j] += step_size * (ub[j] - lb[j])
        if not is_invalid(new_indiv):
            new_fitness = f.evaluate(new_indiv)
            EVALUATE_COUNT += 1
            if new_fitness > best_fitness:
                best_indiv = new_indiv
                best_fitness = new_fitness
        new_indiv = indiv
        new_indiv[j] -= step_size * (ub[j] - lb[j])
        if not is_invalid(new_indiv):
            EVALUATE_COUNT += 1
            new_fitness = f.evaluate(new_indiv)
            if new_fitness > best_fitness:
                best_indiv = new_indiv
                best_fitness = new_fitness
    if best_fitness > fitness:
        local_search_improve_count += 1
    return best_indiv,best_fitness

def local_search(population,fitness):
    global EVALUATE_COUNT,local_search_improve_count
    step_size = LOCAL_SEARCH_LENGTH_RATE * (1 - tanh(EVALUATE_COUNT/MAX_EVALUATE_COUNT * 3))
    for i in range(population.shape[0]):
        indiv = population[i]
        indiv_fitness = fitness[i]
        best_indiv = indiv
        best_fitness = indiv_fitness
        for j in range(population.shape[1]):
            new_indiv = indiv
            new_indiv[j] += step_size * (ub[j]-lb[j])
            if not is_invalid(new_indiv):
                new_fitness = f.evaluate(new_indiv)
                EVALUATE_COUNT += 1
                if new_fitness > best_fitness:
                    best_indiv = new_indiv
                    best_fitness = new_fitness
            new_indiv = indiv
            new_indiv[j] -= step_size * (ub[j] - lb[j])
            if not is_invalid(new_indiv):
                EVALUATE_COUNT += 1
                new_fitness = f.evaluate(new_indiv)
                if new_fitness > best_fitness:
                    best_indiv = new_indiv
                    best_fitness = new_fitness
        population[i] = best_indiv
        if (fitness[i] < best_fitness):
            local_search_improve_count += 1
        fitness[i] = best_fitness
    return population,fitness

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

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

def select_fitness_sharing(parents,parents_fitness,kids,kids_fitness):
    keep_rate = 0.1
    edge = int(keep_rate*POPU_SIZE)
    buffer_popu = parents[0:edge,:]
    buffer_fitness = parents_fitness[0:edge]
    population = np.r_[parents[edge+1:,:], kids]
    fitness = np.r_[parents_fitness[edge+1:], kids_fitness]
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
    sorted_popu =sorted_popu[0:POPU_SIZE,:]
    sorted_fitness = sorted_fitness[0:POPU_SIZE]

    sorted_popu = np.r_[buffer_popu,sorted_popu]
    sorted_fitness = np.r_[buffer_fitness, sorted_fitness]

    order = np.argsort(sorted_fitness)
    print("order: ",order)
    sorted_popu = sorted_popu[-order]
    sorted_fitness = sorted_fitness[-order]
    return sorted_popu,sorted_fitness

def exist(indiv,population):
    a = (population == indiv)
    duplicate = False
    for c in a:
        if c.all():
            duplicate = True
            break
    return duplicate

def generate(population):
    kids = np.zeros((KIDS_NUM,DIM))
    kids_count = 0
    while kids_count < KIDS_NUM:
        kid = crossover(population[random.randint(0,population.shape[0]-1)],population[random.randint(0,population.shape[0]-1)])
        kid = mutate(kid)
        if is_invalid(kid):
            continue
        if not exist(kid,kids):
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
            sh = 0.0001
        new_fitness[i] = math.pow(fitness[i],SHARING_BETA) / sh
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

def EA_fitness_sharing():
    global DIM
    global ub
    global lb
    global EVALUATE_COUNT
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

        population,fitness = select_fitness_sharing(population,fitness,kids,kids_fitness)
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

def generate_crowding(file ,f,population,fitness):
    kids_count = 0
    global EVALUATE_COUNT
    # no local search : 10 11 13
    # has local search : 11 13 11
    # population,fitness = local_search(population,fitness)
    while kids_count < POPU_SIZE:
        while True:
            indexA = random.randint(0,population.shape[0]-1)
            indexB = random.randint(0,population.shape[0]-1)
            parentA = population[indexA]
            parentB = population[indexB]
            kidA= crossover(parentA,parentB)
            kidB = crossover(parentB,parentA)
            kidA = mutate(kidA)
            kidB = mutate(kidB)
            #if (not is_invalid(kidA)) and (not is_invalid(kidB)) and (not exist(kidA,population)) and (not exist(kidB,population)) :
            break

        if  is_invalid(kidA) or is_invalid(kidB):
            continue
        ## check if kid is exist in kids
        kidA_fitness = f.evaluate(kidA)
        kidB_fitness = f.evaluate(kidB)
        EVALUATE_COUNT += 2

        #kidA,kidA_fitness = local_search1(kidA,kidA_fitness)
        #kidB, kidB_fitness = local_search1(kidB, kidB_fitness)
        if((distance(parentA,kidA)+distance(parentB,kidB)) > (distance(parentA,kidB)+distance(parentB,kidA))):
            if(kidA_fitness > fitness[indexA]):
                population[indexA] = kidA
                fitness[indexA] = kidA_fitness
                kids_count += 1
                for i in kidA:
                    file.write(str(i) + " ")
                file.write(str(kidA_fitness) + " ")
                file.write( str(EVALUATE_COUNT) + '\n')
                if (EVALUATE_COUNT >= MAX_EVALUATE_COUNT):
                    break
            if (kidB_fitness > fitness[indexB]):
                population[indexB] = kidB
                fitness[indexB] = kidB_fitness
                kids_count += 1
                for i in kidA:
                    file.write(str(i) + " ")
                file.write(str(kidA_fitness) + " ")
                file.write( str(EVALUATE_COUNT) + '\n')
                if (EVALUATE_COUNT >= MAX_EVALUATE_COUNT):
                    break
        else:
            if (kidA_fitness > fitness[indexB]):
                population[indexB] = kidA
                fitness[indexB] = kidA_fitness
                kids_count += 1
                for i in kidA:
                    file.write(str(i) + " ")
                file.write(str(kidA_fitness) + " ")
                file.write( str(EVALUATE_COUNT) + '\n')
                if (EVALUATE_COUNT >= MAX_EVALUATE_COUNT):
                    break
            if (kidB_fitness > fitness[indexA]):
                population[indexA] = kidB
                fitness[indexA] = kidB_fitness
                kids_count += 1
                for i in kidA:
                    file.write(str(i) + " ")
                file.write(str(kidA_fitness) + " ")
                file.write( str(EVALUATE_COUNT) + '\n')
                if (EVALUATE_COUNT >= MAX_EVALUATE_COUNT):
                    break
        print("EVALUATE_COUNT: ",EVALUATE_COUNT)
    return population,fitness


def EA_crowding():
    global DIM
    global ub
    global lb
    global EVALUATE_COUNT
    global f
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
    print(30*"*", ", init over")
    ## iteration
    while(EVALUATE_COUNT < MAX_EVALUATE_COUNT):
        print("evaluate count: ",EVALUATE_COUNT)
        population,fitness = generate_crowding(file,f,population,fitness)
        accuracy = 0.1
        count, seeds = how_many_goptima(population, f, accuracy)
        print("In the current population there exist", count, "global optimizers.")
        print("Global optimizers:", seeds)
        print(70*"=")
    file.close()
    accuracys = [1,0.1,0.01,0.001]
    count, seeds = how_many_goptima(population, f, accuracy)
    for accuracy in accuracys:
        print("accurcy = ", accuracy)
        print("In the current population there exist", count, "global optimizers.")
        print("Global optimizers:", seeds)

if __name__ == "__main__":
    EA_crowding()
    print(local_search_improve_count)


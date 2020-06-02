import time

from cec2013.cec2013 import *
import numpy as np
import random

# tech:  Speciation, crowding, local search


###
# thought  Speciation : 0.001 acc on funiton 6, nopt =
###

POPU_SIZE = 1000
KIDS_NUM = 1000
RUN_REPEAT = 1
MUTATE_RATE = 0.6
TARGET_FUNC = 1
CROSSOVER_ALPHA = 0.6
MUTATE_GAMMA = 2
TOURNAMENT_RATE = 0.4
SHARING_SIGMA = 0.5
SHARING_BETA = 2
LOCAL_SEARCH_LENGTH_RATE = 0.1
CKPT = 1000


RADIUS = 0
MAX_EVALUATE_COUNT = 0
EVALUATE_COUNT = 0
DIM = 0
ub = 0
lb = 0
f = None
local_search_improve_count = 0
OPT_LIST = []
OPT_FE_LIST = []
START_TIME = 0



def diff_mutate(target,target_fitness,population,fitness):
    global EVALUATE_COUNT
    found = False
    for i in range(POPU_SIZE):
        refer_index = i
        reference = population[refer_index]
        if distance(reference, target) < RADIUS * math.pow(2, 1 - (EVALUATE_COUNT / MAX_EVALUATE_COUNT)):
            found = True
            break
    if found:
        reference_fitness = fitness[refer_index]
        if(reference_fitness < target_fitness):
            for i in range(DIM):
                target[i] += random.random() * (reference[i] - target[i])
        target_fitness = f.evaluate(target)
        EVALUATE_COUNT += 1
        return target,target_fitness
    return target,target_fitness


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

def random_local_search(indiv,fitness):
    global  EVALUATE_COUNT
    new_indiv = indiv
    c = EVALUATE_COUNT / MAX_EVALUATE_COUNT * 0.9  + 0.1
    for i in range(10):
        for i in range(DIM):
            new_indiv[i] += np.random.standard_cauchy(1) * c
        EVALUATE_COUNT += 1
        if f.evaluate(new_indiv) > fitness:
            return new_indiv,fitness
    return indiv,fitness


def sort(targets,marks): # sort targets based on marks
    order = np.argsort(marks)
    return targets[-order]

def exist(indiv,population):
    a = (population == indiv)
    duplicate = False
    for c in a:
        if c.all():
            duplicate = True
            break
    return duplicate

def is_invalid(indiv):
    invalid = False
    for j in range(DIM):
        if indiv[j]>ub[j] or indiv[j]<lb[j]:
            invalid = True
            break
    return invalid

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


def EA_crowding():
    global DIM
    global ub
    global lb
    global EVALUATE_COUNT,MAX_EVALUATE_COUNT
    global f
    global RADIUS
    global START_TIME
    START_TIME = time.time()
    ## intialization

    f = CEC2013(TARGET_FUNC)
    DIM = f.get_dimension()
    MAX_EVALUATE_COUNT = f.get_maxfes()
    RADIUS = f.get_rho()
    ub = np.zeros(DIM)
    lb = np.zeros(DIM)
    # Get lower, upper bounds
    for k in range(DIM):
        ub[k] = f.get_ubound(k)
        lb[k] = f.get_lbound(k)
    opt_log = [[0,0,0,0] for _ in range(int(MAX_EVALUATE_COUNT/CKPT))]
    for RUN_COUNT in range(1, RUN_REPEAT+1):
        EVALUATE_COUNT = 0
        generation_log_filename = "logs\\problem%03drun%03d_generation_log.txt"% (TARGET_FUNC, RUN_COUNT)
        print(generation_log_filename)
        generation_log_file = open(generation_log_filename, "w")
        #opts_log_filename = "logs\\problem%03drun%03d_opts_log.txt"% (TARGET_FUNC, RUN_COUNT)
        #opts_log_file = open(opts_log_filename, "w")

        population = init(ub,lb)
        fitness = evaluate(f,population)
        fes_list = np.array([EVALUATE_COUNT for i in range(POPU_SIZE)])
        init_time = (time.time()-START_TIME)*1000
        time_list = np.array([init_time for i in range(POPU_SIZE)])
        ## iteration
        unselected = False
        while(EVALUATE_COUNT < MAX_EVALUATE_COUNT):
            if EVALUATE_COUNT%CKPT == 0:
                print(70*"=")
                print(EVALUATE_COUNT)
                accuracys = [ 0.1, 0.01, 0.001,0.0001]
                #opts_log_file.write(str(EVALUATE_COUNT) + " " )
                for k in range(4):
                    accuracy = accuracys[k]
                    count, seeds = how_many_goptima(population, f, accuracy)
                    #opts_log_file.write(str(count) + " ")
                    opt_log[int(EVALUATE_COUNT/CKPT)][k] += count
                    print(count)
                #opts_log_file.write('\n')

                # local search of top
                #population = sort(population,fitness)
                #fitness = sort(fitness,fitness)
                #for i in range(int(0.05 * POPU_SIZE)):
                #    population[i],fitness[i] = random_local_search(population[i],fitness[i])


            # at the point of 0.8 evaluation is used, remove all low fitness individual.
            if unselected and  EVALUATE_COUNT >= 0.8 * MAX_EVALUATE_COUNT:
                unselected = False
                population = sort(population,fitness)
                fitness = sort(fitness,fitness)
                cut = 0
                for i in range(POPU_SIZE):
                    if fitness[i] < fitness[0]:
                        cut = i
                        break
                population = population[:cut]
                fitness = fitness[:cut]
                while population.shape[0] < POPU_SIZE:
                    population = np.r_[population, population]
                    fitness = np.r_[fitness, fitness]
                population = population[:POPU_SIZE]
                fitness = fitness[:POPU_SIZE]

            # generate valid and unduplicated indiv
            while True:
                indexA = random.randint(0, population.shape[0] - 1)
                indexB = random.randint(0, population.shape[0] - 1)

                parentA = population[indexA]
                parentB = population[indexB]
                if distance(parentA,parentB) > RADIUS * math.pow(2,1-(EVALUATE_COUNT/MAX_EVALUATE_COUNT)): # Speciation , to avoid the population get closer and closer to center of solution space
                    continue
                kidA = crossover(parentA, parentB)
                kidB = crossover(parentB, parentA)
                kidA = mutate(kidA)
                kidB = mutate(kidB)
                if (not is_invalid(kidA)) and (not is_invalid(kidB)) and (not exist(kidA, population)) and (
                not exist(kidB, population)):
                    break

            # evaluate
            kidA_fitness = f.evaluate(kidA)
            kidB_fitness = f.evaluate(kidB)
            EVALUATE_COUNT += 2

            # diff mutate
            #kidA,kidA_fitness = diff_mutate(kidA,kidA_fitness,population,fitness)
            #kidB,kidB_fitness= diff_mutate(kidB, kidB_fitness, population, fitness)

            #kidA, kidA_fitness = local_search2(kidA,kidA_fitness)
            #kidB, kidB_fitness = local_search2(kidB, kidB_fitness)

            # compare for replacement
            if ((distance(parentA, kidA) + distance(parentB, kidB)) > (
                    distance(parentA, kidB) + distance(parentB, kidA))):
                if (kidA_fitness > fitness[indexA]):
                    population[indexA] = kidA
                    fitness[indexA] = kidA_fitness
                    fes_list[indexA] = EVALUATE_COUNT
                    time_list[indexA] = (time.time() - START_TIME)*1000
                    generation_log_file.write(str(kidA_fitness) + " ")
                    generation_log_file.write(str(EVALUATE_COUNT) + " " )
                    for i in kidA:
                        generation_log_file.write(str(i) + " ")
                    generation_log_file.write('\n')

                if (kidB_fitness > fitness[indexB]):
                    population[indexB] = kidB
                    fitness[indexB] = kidB_fitness
                    fes_list[indexB] = EVALUATE_COUNT
                    time_list[indexB] = (time.time() - START_TIME)*1000
                    generation_log_file.write(str(kidA_fitness) + " ")
                    generation_log_file.write(str(EVALUATE_COUNT) + " " )

                    for i in kidA:
                        generation_log_file.write(str(i) + " ")
                    generation_log_file.write('\n')

            else:
                if (kidA_fitness > fitness[indexB]):
                    population[indexB] = kidA
                    fitness[indexB] = kidA_fitness
                    fes_list[indexB] = EVALUATE_COUNT
                    time_list[indexB] = (time.time() - START_TIME)*1000
                    generation_log_file.write(str(kidA_fitness) + " ")
                    generation_log_file.write(str(EVALUATE_COUNT) + " " )

                    for i in kidA:
                        generation_log_file.write(str(i) + " ")
                    generation_log_file.write('\n')

                if (kidB_fitness > fitness[indexA]):
                    population[indexA] = kidB
                    fitness[indexA] = kidB_fitness
                    fes_list[indexA] = EVALUATE_COUNT
                    time_list[indexA] = (time.time() - START_TIME)*1000
                    generation_log_file.write(str(kidA_fitness) + " ")
                    generation_log_file.write(str(EVALUATE_COUNT) + " " )

                    for i in kidA:
                        generation_log_file.write(str(i) + " ")
                    generation_log_file.write('\n')

        opts_log_filename = "average\\problem%03d_opts_log.txt" % TARGET_FUNC
        opts_log_file = open(opts_log_filename, "w")
        for x in range(int(MAX_EVALUATE_COUNT/CKPT)):
            opts_log_file.write(str(x) + " ")
            for k in range(4):
                opts_log_file.write(str(opt_log[x][k]/(RUN_REPEAT)) + " ")
            opts_log_file.write("\n")

        generation_log_file.close()
        output_filename = "output\\problem%03drun%03d.dat" % (TARGET_FUNC, RUN_COUNT)
        output_file = open(output_filename, "w")
        for i in range(POPU_SIZE):
            for j in range(DIM):
                output_file.write(str(population[i][j]) + " ")
            output_file.write("= ")
            output_file.write(str(fitness[i]))
            output_file.write(" @ ")
            output_file.write(str(fes_list[i]) + " " + str(time_list[i]) + " " + "1" + '\n')
        output_file.close()

        accuracys = [1,0.1,0.01,0.001]
        for accuracy in accuracys:
            count, seeds = how_many_goptima(population, f, accuracy)
            print("accurcy = ", accuracy)
            print("In the current population there exist", count, "global optimizers.")

if __name__ == "__main__":

    EA_crowding()


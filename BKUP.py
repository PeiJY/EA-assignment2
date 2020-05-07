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
TARGET_FUNC = 6
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


def calculate_tour_rate():
    return TOURNAMENT_RATE * ( tanh(EVALUATE_COUNT / MAX_EVALUATE_COUNT * 2))

# 每个维度，正负2侧生成
def local_search1(indiv,fitness):
    global EVALUATE_COUNT,local_search_improve_count
    step_size =  RADIUS * LOCAL_SEARCH_LENGTH_RATE * (1 - tanh(EVALUATE_COUNT / MAX_EVALUATE_COUNT * 3))
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
    return best_indiv,best_fitness

# 一侧有提升则不再生成
def local_search2(indiv,fitness):
    global EVALUATE_COUNT,local_search_improve_count
    step_size =  RADIUS * LOCAL_SEARCH_LENGTH_RATE * (1 - tanh(EVALUATE_COUNT / MAX_EVALUATE_COUNT * 2))
    best_indiv = indiv
    best_fitness = fitness
    for j in range(indiv.shape[0]):
        new_indiv = indiv
        new_indiv[j] += step_size * (ub[j] - lb[j])
        if not is_invalid(new_indiv):
            new_fitness = f.evaluate(new_indiv)
            EVALUATE_COUNT += 1
            if new_fitness > fitness:
                best_indiv = new_indiv
                best_fitness = new_fitness
                break
        new_indiv = indiv
        new_indiv[j] -= step_size * (ub[j] - lb[j])
        if not is_invalid(new_indiv):
            EVALUATE_COUNT += 1
            new_fitness = f.evaluate(new_indiv)
            if new_fitness > best_fitness:
                best_indiv = new_indiv
                best_fitness = new_fitness
                break
    return best_indiv,best_fitness

def local_search(population,fitness):
    global EVALUATE_COUNT,local_search_improve_count
    step_size = RADIUS *  LOCAL_SEARCH_LENGTH_RATE * (1 - tanh(EVALUATE_COUNT/MAX_EVALUATE_COUNT * 3))
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
                if dis < RADIUS:
                    sh += 1 - math.pow((dis / RADIUS), SHARING_SIGMA)
        if sh == 0:
            sh = 0.0001
        new_fitness[i] = math.pow(fitness[i],SHARING_BETA) / sh
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
    global EVALUATE_COUNT,MAX_EVALUATE_COUNT
    global RADIUS

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
    RADIUS = f.get_rho()
    MAX_EVALUATE_COUNT = f.get_maxfes()
    population = init(ub,lb)
    fitness = evaluate(f,population)
    print(30*"*", ", init over")
    ## iteration
    while(EVALUATE_COUNT < MAX_EVALUATE_COUNT):
        print("evaluate count: ",EVALUATE_COUNT)
        kids = generate(population)
        kids_fitness = evaluate(f, kids)

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
    for RUN_COUNT in range(1, RUN_REPEAT):
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

            # generate valid and unduplicated indiv
            while True:
                indexA = random.randint(0, population.shape[0] - 1)
                indexB = random.randint(0, population.shape[0] - 1)
                parentA = population[indexA]
                parentB = population[indexB]
                if distance(parentA,parentB) > RADIUS: # Speciation , to avoid the population get closer and closer to center of solution space
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
                opts_log_file.write(str(opt_log[x][k]/(MAX_EVALUATE_COUNT/CKPT)) + " ")
            opts_log_file.write("\n")

        generation_log_file.close()
        output_filename = "output\\problem%03drun%03d.dat" % (TARGET_FUNC, RUN_COUNT)
        output_file = open(output_filename, "w")  ##problem001run001.dat
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
            # print("Global optimizers:", seeds)


# not used
def find_opts(population,fitness):
    global OPTS,OPTS_FITNESS,OPTS_FES,OPTS_TIME
    opts = []
    opts_fitness = []
    opts_fes = []
    opts_time = []
    accept_range = 0.9
    population = sort(population,fitness)
    fitness = sort(fitness,fitness)
    if OPTS_FITNESS.size > 0:
        best_fitness = OPTS_FITNESS[0]
    else:
        best_fitness = 0

    # find the optimal indiv in current generation
    while opts_fitness[-1] > (best_fitness * accept_range):
        optimal_indiv = population[0]
        optimal_fit = fitness[0]
        if optimal_fit > best_fitness:
            best_fitness = optimal_fit
        opts.append(optimal_indiv)
        opts_fitness.append(optimal_fit)
        opts_fes.append(EVALUATE_COUNT)
        opts_time.append((time.time()-START_TIME)*1000)
        # delete all near indiv of optimal indiv
        for i in range(population.shape[0]):
            if distance(population[i],optimal_indiv) <= RADIUS:
                np.delete(population,i,axis=0)
                np.delete(fitness,i,axis=0)
                i -= 1

    sorted = sort(np.array([np.r_[OPTS_TIME, opts_time],np.r_[OPTS_FES ,opts_fes],np.r_[ OPTS_FITNESS,opts_fitness],np.r_[ OPTS,opts]]),np.r_[ OPTS_FITNESS,opts_fitness])

    opts_fitness = sorted[:, 2]
    ckpt = 0
    for i in range(opts_fitness):
        if opts_fitness[i] < best_fitness * accept_range:
            ckpt = i
            break

    OPTS_TIME = sorted[:ckpt,0]
    OPTS_FES = sorted[:ckpt,1]
    OPTS_FITNESS = sorted[:ckpt,2]
    OPTS = sorted[:ckpt, 3:]


if __name__ == "__main__":
    for TARGET_FUNC in range(2,6):
        EA_crowding()


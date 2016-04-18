# Computational Molecular Medicine
# HW5
# Author: Wei Jiang
import numpy as np
import random as rd
import math
from HW5_CMM import GWBPsimulator
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
from time import clock
class GWBPsimulator2(GWBPsimulator):

    def __init__(self,N_initial, prob_die, NO_generation, NO_cell_type,cancer_size):
        GWBPsimulator.__init__(self,N_initial, prob_die, NO_generation, NO_cell_type, cancer_size)
        self.cancer_size = cancer_size
        self.cancer_times = []
    def simulate_one(self):
        for i in range(0, self.NO_generation):
            probs = [self.unifrand(self.population_size[i_type, i]) for i_type in range(0,self.NO_cell_type)]
            for i_type in range(0, self.NO_cell_type):
                if i_type ==0:
                    temp = self.prob_die[i_type]
                    # print temp
                    die = probs[i_type] < temp[0]
                    NO_die = np.sum(die)
                    NO_divide = die.size - NO_die
                    prob_mutate = self.unifrand(2*NO_divide)
                    mutate = prob_mutate< self.prob_die[i_type][1]
                    NO_mutate = np.sum(mutate)
                    NO_normal = 2*NO_divide - NO_mutate
                    self.population_size[i_type, i+1] = NO_normal
                    self.population_size[i_type+1, i+1] = NO_mutate
                else:
                    die = probs[i_type] < self.prob_die[i_type]
                    NO_die = np.sum(die)
                    NO_divide = die.size - NO_die
                    self.population_size[i_type,i+1] += 2*NO_divide
            if self.population_size[1,i+1] >= self.cancer_size:
                break

    def cancer_incidence(self):
        for i in range(0, self.num_simulation):
            population = self.pop_size_multi_simulation[i]
            cancer_occur = population[1,:]> self.cancer_size
            if np.sum(cancer_occur) >=1:
                first_cancer_generation = [i for i, x in enumerate(cancer_occur) if x][0]
            else:
                first_cancer_generation = 10*self.NO_generation
            self.cancer_times.append(first_cancer_generation)
        cdf = []
        for j in range(0, self.NO_generation):
            temp = np.asarray(self.cancer_times) < j # number of simulations T<= j th generation
            cdf.append(np.sum(temp)/float(self.num_simulation))
        plt.figure()
        plt.plot(2*np.arange(0, self.NO_generation), cdf, 'b-')
        plt.xlabel('Months')
        plt.ylabel('P(T<t)')
        plt.title('Cancer incidence')
        plt.savefig('cancer_incidence.pdf')
# self_testing code
if __name__ == '__main__':
    start_time = clock()
    prob_death = [[100/float(201),1/float(5000)],20/float(41)]
    GWBP = GWBPsimulator2(N_initial=[100,0],prob_die=prob_death,NO_generation=500,NO_cell_type=2,cancer_size=math.pow(10,6))
    GWBP.simulate_multiple(1000)
    GWBP.cancer_incidence()
    print 'running time(seconds):', clock() - start_time
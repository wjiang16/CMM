# Computational Molecular Medicine
# HW5
# Author: Wei Jiang
import numpy as np
import random as rd
import math

class GWBPsimulator():
    def __init__(self, N_initial, prob_die, NO_generation, NO_cell_type, cancer_size = math.pow(10,6)):
        '''

        :param N_initial: a list
        :param prob_die: a list of probability of die for different cell types
        :param NO_generation: int
        :param NO_cell_type: int
        :return:
        '''
        self.N_initial = N_initial
        self.prob_die = prob_die
        self.NO_generation = NO_generation
        self.NO_cell_type = NO_cell_type
        self.population_size = np.zeros((self.NO_cell_type, self.NO_generation + 1))
        self.population_size[:,0] = self.N_initial
        self.pop_size_multi_simulation = []
        self.cancer_size = cancer_size
    def simulate_one(self):
        for i in range(0, self.NO_generation):
            probs = [self.unifrand(self.population_size[i_type, i]) for i_type in range(0,self.NO_cell_type)]
            die_or_live = [probs[i_type] < self.prob_die[i_type] for i_type in range(0,self.NO_cell_type)]
            # print die_or_live
            for ind,j in enumerate(die_or_live):
                NO_die = np.sum(j) # number of cell dies
                # print NO_die, j.size
                NO_divide = j.size - NO_die
                # print NO_divide
                self.population_size[ind,i+1] = 2*NO_divide
    def unifrand(self,size):
        size = int(size)
        unif_rand = np.zeros((1,size))
        for i in range(0,size):
            unif_rand[0, i] = rd.uniform(0,1)
        return unif_rand
    def get_population_size(self,ith_generation):
        return self.population_size[:,ith_generation]

    def simulate_multiple(self, num_simulation):
        self.num_simulation = num_simulation
        for i in range(0,num_simulation):
            GWBP = self.__class__(N_initial=self.N_initial,prob_die=self.prob_die,NO_generation=self.NO_generation,
                                  NO_cell_type=self.NO_cell_type,cancer_size= self.cancer_size)
            GWBP.simulate_one()
            self.pop_size_multi_simulation.append(GWBP.population_size)

    def get_expected_popsize(self, ith_generation):
        ith_pop_size = np.asarray([i[:,ith_generation-1] for i in self.pop_size_multi_simulation])
        expected_popsize = np.mean(ith_pop_size,0)
        return expected_popsize

    def get_prob_extinction(self,ith_generation):
        ith_pop_size = np.asarray([i[:,ith_generation-1] for i in self.pop_size_multi_simulation])
        num_extinct = np.sum(ith_pop_size == 0, axis=0)
        prob_extinct = num_extinct/float(self.num_simulation)
        return prob_extinct

    def get_expected_popsize_given_no_extinct(self, ith_generation):
        ith_pop_size = np.asarray([i[:,ith_generation-1] for i in self.pop_size_multi_simulation])
        num_extinct = np.sum(ith_pop_size, axis=0)
        size_no_extinct = np.sum(ith_pop_size > 0, axis=0)
        expected_condition_size = np.divide(1.0*num_extinct, size_no_extinct)
        return expected_condition_size

def prob_extinct(ith_generation,q1=0.5):
    '''

    :param ith_generation: int
    :param q1: float, the probability that N(1)=0, first generation extinct
    :return: float
    '''
    q = q1
    if ith_generation == 1:
        return q
    else:
        for i in range(0,ith_generation-1):
            q = q1 + (1-q1)*q*q
        return q
def expected_size(ith_generation, q):
    u = (1-q)*2
    return math.pow(u, ith_generation)

# self_testing code
if __name__ == '__main__':
    prob_death = 20/float(41)
    GWBP = GWBPsimulator(N_initial=[1],prob_die=[prob_death],NO_generation=100,NO_cell_type=1)
    GWBP.simulate_multiple(500)
    print GWBP.get_expected_popsize(14)
    print 'probability of extinction from simulation',GWBP.get_prob_extinction(15)
    print 'from analytic expression', prob_extinct(15,prob_death )
    print expected_size(14,prob_death)
    print 'conditional expected population size', GWBP.get_expected_popsize_given_no_extinct(100)
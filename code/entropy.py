# entropy.py
# v 1.0  28 April 2016 [GD]

######################
#
# Submission by Gioia Dominedo (Harvard ID: 40966234) for
# AM 207 - Stochastic Methods for Data Analysis, Inference and Optimization
#
# Course Project
#
######################

# runs entropy solver for MasterMind

'''
Example:

>>> python entropy.py
Code successfully initialized to  [1 0 2 2]

guess #1 of 10: you guessed  [2, 1, 4, 3]
You have 0 right item(s) in the right place, and
  2 right item(s) but in the wrong place

guess #2 of 10: you guessed  [3, 0, 2, 5]
You have 2 right item(s) in the right place, and
  0 right item(s) but in the wrong place

guess #3 of 10: you guessed  [1, 0, 2, 0]
You have 3 right item(s) in the right place, and
  0 right item(s) but in the wrong place

guess #4 of 10: you guessed  [1, 0, 2, 2]
You have 4 right item(s) in the right place
You win!
'''

import MMboard as mm

import numpy as np
import itertools
import time
import copy
import matplotlib.pyplot as plt

######################
#
# GLOBAL SEARCH: MAXIMIZE ENTROPY AT EVERY STEP
#
######################

def entropy_all(cl=4, nc=6, code=None, silent=False):
    '''
    Selects the move with the highest Shannon entropy at each step.

    Mathematical reference: http://link.springer.com/article/10.1007%2FBF01917147
    Implementation reference: http://www.geometer.org/mathcircles
    '''

    # initialize game board
    # if code is provided, it must be in the form of a list or 1-D numpy array
    game = mm.MMboard(codelength=cl, numcolors=nc, suppress_output=silent)
    if code is None:
        game.set_code()
    else:
        game.set_code(code)

    # create a set of all possible hidden codes
    possible_guesses = {}  # index is the string sequence, value is a cl-element tuple
    digits = list(np.arange(nc))
    for i in itertools.product(digits, repeat=cl):
        possible_guesses[''.join(map(str, list(i)))] = list(i)
    n_possible_guesses = len(possible_guesses)
    assert n_possible_guesses == nc**cl

    # create a list of all possible (b, w) responses
    responses = list()
    for l in range(cl, -1, -1):
        responses.extend(list(itertools.combinations_with_replacement('BW', l)))
    responses = [''.join(r) for r in responses]
    responses.remove('B' * (cl - 1) + 'W') # not possible
    responses = [(r.count('B'), r.count('W')) for r in responses] # consistent with game response format
    n_responses = len(responses)
    # response_idx = dict(zip(responses, range(n_responses))) # for numpy implementation

    # keep track of the number of guesses
    n_guesses = 0

    while True:
        # this is not an infinite loop -
        # maximum iterations governed by max number of possible guesses

        # if only one guess remains, use that one
        if len(possible_guesses) == 1:
            guess = possible_guesses.values()[0]

        # otherwise choose (one of) the guesses that maximize the shannon entropy
        else:

            # APPROACH #1: DICTIONARIES (FASTER) -->

            # keep track of entropy for each possible guess
            entropy = dict()

            for g in possible_guesses: # possible guesses

                # calculate probability of each response category
                response_prob = dict(zip(responses, [0.0] * n_responses))
                for c in possible_guesses: # possible real codes
                    response_prob[game.check_guess(guess=possible_guesses[g], answer=possible_guesses[c])] += 1

                # calculate entropy of guess
                # assumes log(0) = 0
                entropy[g] = 0.0
                for r in responses:
                    prob = response_prob[r] / n_possible_guesses
                    if prob > 0:
                        entropy[g] -= prob * np.log2(prob)

            # identify the guess(es) that maximize entropy, then pick one randomly
            max_entropy = max(entropy.values())
            best_guesses = [k for k in entropy.keys() if entropy[k] == max_entropy]
            guess = possible_guesses[np.random.choice(best_guesses)]

            # APPROACH #2: NUMPY ARRAYS (SLOWER) -->

            # guess_probs = np.zeros((n_possible_guesses, n_responses), dtype=np.float)
            # guess_idx = possible_guesses.keys()

            # # calculate distribution across response categories
            # for gidx, g in enumerate(guess_idx): # possible guesses
            #     for c in guess_idx: # possible real codes
            #         ridx = response_idx[game.check_guess(guess=possible_guesses[g], answer=possible_guesses[c])]
            #         guess_probs[gidx, ridx] += 1

            # # normalize
            # row_sums = np.sum(guess_probs, axis=1)
            # guess_probs /= row_sums[:, np.newaxis]
            # guess_probs_log = np.log2(guess_probs)
            # guess_probs_log[guess_probs_log == -np.inf] = 0 # assume log(0) = 0

            # # identify the guess(es) that maximize entropy, then pick one randomly
            # entropy = - np.sum(guess_probs * guess_probs_log, axis=1)
            # best_guesses = np.argwhere(entropy == np.max(entropy))
            # guess = possible_guesses[guess_idx[np.random.choice(best_guesses.flatten())]]

        n_guesses += 1

        # delete current guess from possible guesses
        del possible_guesses[''.join(map(str, guess))]

        # play the guess to get a response of colored (b) and white pegs
        response = game.guess_code(guess)

        # terminate if the guess is correct or the maximum number of tries is reaches
        if ((response is not None) and response[0] == cl) or game.gameover:
            return n_guesses

        # remove any codes that wouldn't give the same response
        for code in possible_guesses.keys():
            # uses class' built-in check_guess function
            if game.check_guess(guess=possible_guesses[code], answer=guess) != response:
                del possible_guesses[code]
        n_possible_guesses = len(possible_guesses)

######################
#
# GLOBAL SEARCH: CHOOSE RANDOMLY AT FIRST STEP
#
######################

def entropy_minusone(cl=4, nc=6, code=None, silent=False):
    '''
    Selects the move with the highest Shannon entropy at each step, except for the first step,
    where a random code is selected instead. The entropy of most guesses is very similar at the
    first step, so we trade off very little or no performance for significant improvements in runtime.

    Mathematical reference: http://link.springer.com/article/10.1007%2FBF01917147
    Implementation reference: http://www.geometer.org/mathcircles
    '''

    # initialize game board
    # if code is provided, it must be in the form of a list or 1-D numpy array
    game = mm.MMboard(codelength=cl, numcolors=nc, suppress_output=silent)
    if code is None:
        game.set_code()
    else:
        game.set_code(code)

    # create a set of all possible hidden codes
    possible_guesses = {}  # index is the string sequence, value is a cl-element tuple
    digits = list(np.arange(nc))
    for i in itertools.product(digits, repeat=cl):
        possible_guesses[''.join(map(str, list(i)))] = list(i)
    n_possible_guesses = len(possible_guesses)
    assert n_possible_guesses == nc**cl

    # create a list of all possible (b, w) responses
    responses = list()
    for l in range(cl, -1, -1):
        responses.extend(list(itertools.combinations_with_replacement('BW', l)))
    responses = [''.join(r) for r in responses]
    responses.remove('B' * (cl - 1) + 'W') # not possible
    responses = [(r.count('B'), r.count('W')) for r in responses] # consistent with game response format
    n_responses = len(responses)
    # response_idx = dict(zip(responses, range(n_responses))) # for numpy implementation

    # keep track of the number of guesses
    n_guesses = 0

    # initial random guess
    guess = list(np.random.randint(0, nc, cl))

    while True:
        # this is not an infinite loop -
        # maximum iterations governed by max number of possible guesses

        # delete current guess from possible guesses
        del possible_guesses[''.join(map(str, guess))]

        # play the guess to get a response of colored (b) and white pegs
        response = game.guess_code(guess)
        n_guesses += 1

        # terminate if the guess is correct or the maximum number of tries is reaches
        if ((response is not None) and response[0] == cl) or game.gameover:
            return n_guesses

        # remove any codes that wouldn't give the same response
        for code in possible_guesses.keys():
            # uses class' built-in check_guess function
            if game.check_guess(guess=possible_guesses[code], answer=guess) != response:
                del possible_guesses[code]
        n_possible_guesses = len(possible_guesses)

        # if only one guess remains, use that one
        if len(possible_guesses) == 1:
            guess = possible_guesses.values()[0]

        # otherwise choose (one of) the guesses that maximize the shannon entropy
        else:

            # APPROACH #1: DICTIONARIES (FASTER) -->

            # keep track of entropy for each possible guess
            entropy = dict()

            for g in possible_guesses: # possible guesses

                # calculate probability of each response category
                response_prob = dict(zip(responses, [0.0] * n_responses))
                for c in possible_guesses: # possible real codes
                    response_prob[game.check_guess(guess=possible_guesses[g], answer=possible_guesses[c])] += 1

                # calculate entropy of guess
                # assumes log(0) = 0
                entropy[g] = 0.0
                for r in responses:
                    prob = response_prob[r] / n_possible_guesses
                    if prob > 0:
                        entropy[g] -= prob * np.log2(prob)

            # identify the guess(es) that maximize entropy, then pick one randomly
            max_entropy = max(entropy.values())
            best_guesses = [k for k in entropy.keys() if entropy[k] == max_entropy]
            guess = possible_guesses[np.random.choice(best_guesses)]

            # APPROACH #2: NUMPY ARRAYS (SLOWER) -->

            # guess_probs = np.zeros((n_possible_guesses, n_responses), dtype=np.float)
            # guess_idx = possible_guesses.keys()

            # # calculate distribution across response categories
            # for gidx, g in enumerate(guess_idx): # possible guesses
            #     for c in guess_idx: # possible real codes
            #         ridx = response_idx[game.check_guess(guess=possible_guesses[g], answer=possible_guesses[c])]
            #         guess_probs[gidx, ridx] += 1

            # # normalize
            # row_sums = np.sum(guess_probs, axis=1)
            # guess_probs /= row_sums[:, np.newaxis]
            # guess_probs_log = np.log2(guess_probs)
            # guess_probs_log[guess_probs_log == -np.inf] = 0 # assume log(0) = 0

            # # identify the guess(es) that maximize entropy, then pick one randomly
            # entropy = - np.sum(guess_probs * guess_probs_log, axis=1)
            # best_guesses = np.argwhere(entropy == np.max(entropy))
            # guess = possible_guesses[guess_idx[np.random.choice(best_guesses.flatten())]]

######################
#
# LOCAL SEARCH: SIMULATED ANNEALING
#
# adapted from annealing.py
#
######################

class SAentropy():

    '''
    Uses entropy as the objective function to maximize in the simulated annealing
    local search algorithm.
    '''

    def __init__(self, show_graph=False):
        # store previous guesses, along with response
        # key is the string sequence, values is the (b, w) tuple as returned
        # by guess_code;
        # supports up to N=10 digits (0-indexed) per position
        self._prev_guesses = {}

        self._best_guess = None  # stores best guess so far
        self._best_resp = None  # stores best response so far
        self._show_graph = False

    def _objective_function(self, guess):
        '''
        Objective function: maximize shannon entropy (i.e. wider probability distribution)
        '''

        # check that guess matches previous responses
        assert self._sa._codeOK(guess)

        # calculate shannon entropy for the guess, given the remaining possible guesses

        # calculate probability of each response category
        response_prob = dict(zip(self.responses, [0.0] * self.n_responses))
        for c in self.possible_guesses: # possible real codes
            response_prob[self._sa.check_guess(guess=guess, answer=self.possible_guesses[c])] += 1

        # calculate entropy of guess
        entropy = 0.0
        for r in self.responses:
            prob = response_prob[r] / self.n_possible_guesses
            if prob > 0: # assumes log(0) = 0
                entropy -= prob * np.log2(prob)

        return -entropy # SA is minimizing

    def _change_guess(self, guess, repeats=5):
        '''
        Pick randomly from the set of possible guesses.
        '''

        guess = np.random.choice(list(self.possible_guesses))
        return self.possible_guesses[guess]

    def _closeness_score(self, resp):
        # the higher score, the more correct the answer
        # we cannot really use this as the objective function, since we need
        # to actually make a guess before getting this response (it is based
        # on the actual code and only attainable by committing a guess);
        # but we can use this to identify the final best answer for reporting
        # if we did not guess the correct answer in the limited number of
        # tries/time

        # inspired by the score for the genetic algorithm, we start with the
        # following:
        # score: 10 * black + 1 * white
        # weights correct digit in correct position more heavily

        # response is in the form of a (b, w) tuple
        # however, we modify this score so that if all the correct digits have
        # been found (but answer is wrong), this is just as good as finding all
        # but one correct digits as we can reduce the
        # dimension of the search to just one type (mutation OR permutation)

        if (resp[0] != self._sa._L) and ((resp[0] + resp[1]) == self._sa._L):
            return (self._sa._L - 1) * 10

        return resp[0] * 10 + resp[1]

    def sa(self, guess, resp, init_temp, thermostat, ftol, itol, otol, k):
        '''Perform simulated annealing.

        Inputs:
        guess: the initial guess
        resp: the response returned from the MasterMind solver for that guess
        init_temp: the initial temperature for the SA algorithm
        thermostat: the rate at which temperature is decreased
        itol: a proxy for the maximum time SA can run to solve the game
              in SA iterations (which is larger than number of guesses made)
        ftol, otol: other stopping criteria, not currently used
        k: temperature tuning parameter

        Returns:
        A tuple containing the (best guess before time expired,
                                the (b, w) response for the best guess,
                                the number of iterations<=itol performed) '''

        obj_values = []  # stores objective function values from successive runs
        curr_guess = copy.deepcopy(guess)

        prev_E = 3 * self._sa._L  # arbitrarily large initial "energy"
        obj_values.append(prev_E)
        temperature = init_temp

        best_score = self._closeness_score(resp)  # store best objective function seen so far
        self._best_guess = guess  # store initial guess as corresponding best guess
        self._best_resp = resp  # store its response as corresponding best response

        it = 1  # acceptances counter (i.e. actual guesses made)
        atp = 1  # total number of iterations

        while True:

            L = np.floor(temperature).astype(int)  # step size
            # L = np.floor(np.exp(temperature)).astype(int)  # alternative step size
            # L = np.floor(np.sqrt(temperature)).astype(int)  # alternative step size
            # L = np.floor(np.log(temperature)).astype(int)  # alternative step size

            propose_guess = self._change_guess(curr_guess, L)

            # check whether proposed guess has already been guessed
            pg_str = ''.join(map(str, propose_guess))
            while pg_str in self._prev_guesses:
                propose_guess = self._change_guess(curr_guess)  # guess again
                pg_str = ''.join(map(str, propose_guess))

            new_E = self._objective_function(propose_guess)
            delta_E = new_E - prev_E

            # if we find something of equal (not just strictly lower) cost,
            # we try that as well as the corresponding combination could be good
            if ((delta_E <= 0) or
                (np.random.rand() < np.exp(- 1.0 * delta_E / (k * temperature)))):
                # how choice was made
                if delta_E <= 0:
                    if not(self._sa._nooutput):
                        print "accepted better or equal, temperature:", temperature
                else:
                    if not(self._sa._nooutput):
                        print "accepted worse (random), temperature:", temperature

                # accept proposal from which to make next guess
                curr_guess = propose_guess
                obj_values.append(new_E)
                prev_E = new_E
                it += 1

                # delete current guess from possible guesses
                del self.possible_guesses[''.join(map(str, propose_guess))]

                # actually make the guess
                response = self._sa.guess_code(propose_guess)

                # keep track of best guess
                if self._closeness_score(response) >= best_score:
                    best_score = self._closeness_score(response)
                    self._best_guess = curr_guess
                    self._best_resp = response

                # If the response is four colored pegs, the game is won,
                # the algorithm terminates.
                # If maximum number of tries reached, the algorithm terminates.
                if (((response is not None) and response[0]==self._sa._L) or
                    self._sa.gameover):
                    # print details of best guess when game has ended
                    if response[0] != self._sa._L:
                        if not(self._sa._nooutput):
                            print 'best guess', self._best_guess, self._best_resp, it
                            print 'actual', self._sa._code
                    break

                # Otherwise, cache the response
                self._prev_guesses[pg_str] = response

                # remove any codes that wouldn't give the same response
                for code in self.possible_guesses.keys():
                    # uses class' built-in check_guess function
                    if self._sa.check_guess(guess=self.possible_guesses[code], answer=propose_guess) != response:
                        del self.possible_guesses[code]
                self.n_possible_guesses = len(self.possible_guesses)

            atp += 1

            # temperature adjustment and reannealing
            temperature = thermostat * temperature
            if ((temperature <= 2) and
                (self._closeness_score(response) < best_score)):
                # we implemented a variation of reannealing here
                # if we jumped to a new guess that has a score that is
                # worse than the current best score, then we have to jump
                # around more to further explore the space (i.e. "reheat")
                # so we reset to a higher fraction of the initial temperature
                temperature = init_temp * 0.8

            # termination conditions
            if atp > itol:
                if not(self._sa._nooutput):
                    print 'itol: maximum iterations reached'
                    print 'best guess', self._best_guess, self._best_resp, it
                    print 'actual', self._sa._code
                break

        if not(self._sa._nooutput) and self._show_graph:
            print "plotted"
            plt.figure()
            plt.plot(obj_values)
            plt.title("Objective function")
            plt.show()

        return self._best_guess, self._best_resp, it

    def runSA(self, cl=4, nc=6, code=None, silent=False):
        '''
        Driver: Set up board and run SA algorithm.
        Modified to keep track of state space
        '''

        # if code is provided, it must be in the form of a list or 1-D numpy array
        self._sa = mm.MMboard(codelength=cl, numcolors=nc, suppress_output=silent)
        if code is None:
            self._sa.set_code()
        else:
            self._sa.set_code(code)

        # create a set of all possible hidden codes
        self.possible_guesses = {}  # index is the string sequence, value is a cl-element tuple
        digits = list(np.arange(nc))
        for i in itertools.product(digits, repeat=cl):
            self.possible_guesses[''.join(map(str, list(i)))] = list(i)
        self.n_possible_guesses = len(self.possible_guesses)
        assert self.n_possible_guesses == nc**cl

        # create a list of all possible (b, w) responses
        self.responses = list()
        for l in range(cl, -1, -1):
            self.responses.extend(list(itertools.combinations_with_replacement('BW', l)))
        self.responses = [''.join(r) for r in self.responses]
        self.responses.remove('B' * (cl - 1) + 'W') # not possible
        self.responses = [(r.count('B'), r.count('W')) for r in self.responses] # consistent with game response format
        self.n_responses = len(self.responses)

        # initial guess
        sa_guess = list(np.random.randint(0, nc, cl))

        # delete current guess from possible guesses
        del self.possible_guesses[''.join(map(str, sa_guess))]

        # play the guess to get a response of colored (b) and white pegs
        response = self._sa.guess_code(sa_guess)

        # If the response is four colored pegs, the game is won, the algorithm
        # terminates.
        # If maximum number of tries reached, the algorithm terminates.
        if ((response is not None) and response[0] == cl) or self._sa.gameover:
            return 1

        # Otherwise, cache the response
        self._prev_guesses[''.join(map(str, sa_guess))] = response

        # remove any codes that wouldn't give the same response
        for code in self.possible_guesses.keys():
            # uses class' built-in check_guess function
            if self._sa.check_guess(guess=self.possible_guesses[code], answer=sa_guess) != response:
                del self.possible_guesses[code]
        self.n_possible_guesses = len(self.possible_guesses)

        # we fiddled around with these parameters
        init_temp = 6
        thermostat = 0.98
        k = 0.2  # try slightly bigger fractions for larger spaces
        itol = 10000  # this is a proxy for allowable execution time

        # output is in the form of best guess, best response, number of guesses
        output = self.sa(sa_guess, response, init_temp, thermostat, 0, itol, 0, k)

        return output[2]  # number of guesses

######################
#
# LOCAL SEARCH: GENETIC ALGORITHMS
#
# adapted from genetic.py
#
######################

class GAentropy():
    '''
    Uses entropy as the objective function to maximize in the genetic algorithms
    local search algorithm.
    '''

    def __init__(self):
        # store previous guesses, along with response in the form of a tuple
        # (code, response) where code is the python array and response is the
        # (b, w) tuple as returned by guess_code;
        self._prev_guesses = []

        self.game = None  # game board instance

        # experiment with these parameters
        self.max_pop = 60
        self.max_gen = 100

        self.crossover_points = 2
        self.crossover_prob = 0.5
        self.crossover_mutation_prob = 0.03
        self.permutation_prob = 0.03
        self.inversion_prob = 0.02

    def _objective_function(self, guess):
        '''
        Objective function: maximize shannon entropy (i.e. wider probability distribution)
        '''

        # check that guess matches previous responses
        assert self.game._codeOK(guess)

        # convert to string
        guess_string = ''.join(map(str, guess))

        # return negative value if the code is not a possible guess
        # minimum entropy for valid codes = 0
        if guess_string not in self.possible_guesses:
            return -1

        # calculate shannon entropy for the guess, given the remaining possible guesses

        # calculate probability of each response category
        response_prob = dict(zip(self.responses, [0.0] * self.n_responses))
        for c in self.possible_guesses: # possible real codes
            response_prob[self.game.check_guess(guess=guess, answer=self.possible_guesses[c])] += 1

        # calculate entropy of guess
        entropy = 0.0
        for r in self.responses:
            prob = response_prob[r] / self.n_possible_guesses
            if prob > 0: # assumes log(0) = 0
                entropy -= prob * np.log2(prob)

        return entropy

    def evolve_population(self, pop_size, gen_size):
        '''
        :param pop_size: maximum size for a given population
        :param gen_size: maximum number of population generations
        '''
        # initialize population
        population = np.random.choice(self.game._C, (pop_size, self.game._L), replace=True)
        population = [list(item) for item in population]

        elite = []
        h = 1

        while not elite:

            children = []

            for i in range(pop_size-1):
                child = self.crossover(population[i], population[i+1])

                if np.random.rand() <= self.crossover_mutation_prob:
                    child = self.mutate(child)

                child = self.permute(child)

                child = self.invert(child)

                children.append(child)

            # append last one
            children.append(population[pop_size-1])

            # Calculate fitness score for each child. The closer to 0, the better
            pop_score = []
            for c in children:
                pop_score.append((self._objective_function(c), c))

            # sort population based on fitness in ascending order
            # pop_score = sorted(pop_score, key=lambda x: x[0])

            # Pick ones where score is >0
            eligibles = [e for (score, e) in pop_score if score>0]

            # no good ones, move on to next generation to try again
            if len(eligibles) == 0:
                h += 1
                continue

            # remove duplicates
            for code in eligibles:
                if code in elite:
                    elite.remove(code)

                    # replace the removed duplicate elite code with a random one
                    elite.append(list(np.random.choice(self.game._C, self.game._L, replace=True)))

            # add eligible to elite
            for eligible in eligibles:
                if len(elite) == pop_size:
                    break

                if eligible not in elite:
                    elite.append(eligible)

            # Prepare the parent population for the next generation based
            # on the current generation
            population = []
            population.extend(eligibles)

            # fill the rest of the population with random codes up to pop_size
            j = len(eligibles)
            while j < pop_size:
                population.append(list(np.random.choice(self.game._C, self.game._L, replace=True)))
                j += 1

            h += 1

        return elite

    def crossover(self, code1, code2):
        '''
        Up to number of crossovers equal to self.crossover_points, each with probability of self.crossover_prob,
        crosses code1 with code2
        '''
        result = copy.copy(code1)

        for i in range(self.game._L):
            if np.random.rand() <= self.crossover_prob:
                result[i] = code2[i]

        return result

    def mutate(self, code):
        '''
        With a probability of self.crossover_mutation_prob, crossover is followed by a mutation that replaces
        the color of one randomly chosen position by a random other color
        '''
        result = copy.copy(code)

        pos = np.random.choice(self.game._L)
        color = np.random.choice(self.game._C)
        # try again until mutate to a different color
        while result[pos] == color:
            pos = np.random.choice(self.game._L)
            color = np.random.choice(self.game._C)

        result[pos] = color

        return result

    def permute(self, code):
        '''
        With a probability of self.permutation_prob, colors of two random positions are switched
        '''
        result = copy.copy(code)

        if np.random.rand() <= self.permutation_prob:
            pos_a, pos_b = np.random.choice(self.game._L, 2, replace=False)

            result[pos_a], result[pos_b] = result[pos_b], result[pos_a]

        return result

    def invert(self, code):
        '''
        with probability of self.inversion_prob, two randomly chosen positions have their sequence of colors
        between these two positions inverted
        '''
        result = copy.copy(code)

        if np.random.rand() <= self.inversion_prob:
            pos_range = np.random.choice(self.game._L, 2, replace=False)
            pos1, pos2 = np.sort(pos_range)
            flipped = np.fliplr([result[pos1:pos2]])[0]
            result[pos1:pos2] = flipped

        return result

    def runGA(self, cl=4, nc=6, code=None, silent=False, initial=None):
        '''
        Set up board and run GA algorithm
        '''
        # if code is provided, it must be in the form of a list or 1-D numpy array
        self.game = mm.MMboard(codelength=cl, numcolors=nc, suppress_output=silent)
        if code is None:
            self.game.set_code()
        else:
            self.game.set_code(code)

        # create a set of all possible hidden codes
        self.possible_guesses = {}  # index is the string sequence, value is a cl-element tuple
        digits = list(np.arange(nc))
        for i in itertools.product(digits, repeat=cl):
            self.possible_guesses[''.join(map(str, list(i)))] = list(i)
        self.n_possible_guesses = len(self.possible_guesses)
        assert self.n_possible_guesses == nc**cl

        # create a list of all possible (b, w) responses
        self.responses = list()
        for l in range(cl, -1, -1):
            self.responses.extend(list(itertools.combinations_with_replacement('BW', l)))
        self.responses = [''.join(r) for r in self.responses]
        self.responses.remove('B' * (cl - 1) + 'W') # not possible
        self.responses = [(r.count('B'), r.count('W')) for r in self.responses] # consistent with game response format
        self.n_responses = len(self.responses)

        # initial guess
        if initial:
            ga_guess = initial
        elif cl == 4 and nc == 6:
            ga_guess = [1, 1, 2, 3]
        else:
            ga_guess = list(np.random.randint(0, nc, cl))

        # delete current guess from possible guesses
        del self.possible_guesses[''.join(map(str, ga_guess))]

        # play the guess to get a response of colored (b) and white pegs.
        response = self.game.guess_code(ga_guess)
        self._prev_guesses.append((ga_guess, response))

        # remove any codes that wouldn't give the same response
        for code in self.possible_guesses.keys():
            # uses class' built-in check_guess function
            if self.game.check_guess(guess=self.possible_guesses[code], answer=ga_guess) != response:
                del self.possible_guesses[code]
        self.n_possible_guesses = len(self.possible_guesses)

        while self.game.gameover is False:

            eligibles = self.evolve_population(self.max_pop, self.max_gen)

            ga_guess = eligibles.pop()

            while ga_guess in [c for (c, r) in self._prev_guesses]:
                if not eligibles:
                    continue
                else:
                    ga_guess = eligibles.pop()

            # delete current guess from possible guesses
            del self.possible_guesses[''.join(map(str, ga_guess))]

            response = self.game.guess_code(ga_guess)
            self._prev_guesses.append((ga_guess, response))

            # remove any codes that wouldn't give the same response
            for code in self.possible_guesses.keys():
                # uses class' built-in check_guess function
                if self.game.check_guess(guess=self.possible_guesses[code], answer=ga_guess) != response:
                    del self.possible_guesses[code]
            self.n_possible_guesses = len(self.possible_guesses)

        return self.game.n_guessed

if __name__ == "__main__":

    np.seterr(divide='ignore') # supress warnings related to log(0)
    
    print '** METHOD 1: MAXIMIZE ENTROPY AT ALL STEPS **'
    timer = time.time()
    entropy_all() # maximize entropy at all steps
    print 'Runtime: %0.2f seconds' % (time.time() - timer)

    print
    print '** METHOD 1: CHOOSE RANDOMLY AT FIRST STEP **'
    timer = time.time()
    entropy_minusone() # choose randomly at first step
    print 'Runtime: %0.2f seconds' % (time.time() - timer)

    print
    print '** METHOD 3: LOCAL SEARCH WITH SIMULATED ANNEALING **'
    timer = time.time()
    s = SAentropy()
    s.runSA()
    print 'Runtime: %0.2f seconds' % (time.time() - timer)

    print
    print '** METHOD 4: LOCAL SEARCH WITH GENETIC ALGORITHMS **'
    timer = time.time()
    g = GAentropy()
    g.runGA()
    print 'Runtime: %0.2f seconds' % (time.time() - timer)

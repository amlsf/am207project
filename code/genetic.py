# genetic.py
# v 1.0  27 April 2016 [AL]

######################
#
# Submission by Amy Lee (Harvard ID: 60984077) for
# AM 207 - Stochastic Methods for Data Analysis, Inference and Optimization
#
# Course Project
#
######################

# runs Genetic Algorithm for MasterMind
# Reference: Berghmann et al. "Efficient solutions for Mastermind using genetic algorithms"

'''
Example:

>>> python genetic.py

Code successfully initialized to  [1 5 2 2]

guess #1 of 10: you guessed  [1, 3, 1, 4]
You have 1 right item(s) in the right place, and
  0 right item(s) but in the wrong place

accepted better or equal, temperature: 6
guess #2 of 10: you guessed  [4, 4, 3, 0]
You have 0 right item(s) in the right place, and
  0 right item(s) but in the wrong place

accepted worse (random), temperature: 5.7624
guess #3 of 10: you guessed  [5, 2, 4, 1]
You have 0 right item(s) in the right place, and
  3 right item(s) but in the wrong place

accepted better or equal, temperature: 5.20875319948
guess #4 of 10: you guessed  [2, 4, 2, 5]
You have 1 right item(s) in the right place, and
  2 right item(s) but in the wrong place

accepted worse (random), temperature: 4.08739574544
guess #5 of 10: you guessed  [4, 2, 5, 5]
You have 0 right item(s) in the right place, and
  2 right item(s) but in the wrong place

accepted better or equal, temperature: 2.89663070294
guess #6 of 10: you guessed  [2, 1, 5, 2]
You have 1 right item(s) in the right place, and
  3 right item(s) but in the wrong place

accepted worse (random), temperature: 2.7819241271
guess #7 of 10: you guessed  [5, 1, 2, 2]
You have 2 right item(s) in the right place, and
  2 right item(s) but in the wrong place

accepted better or equal, temperature: 2.56595823837
guess #8 of 10: you guessed  [1, 5, 2, 2]
You have 4 right item(s) in the right place
You win!
'''

import MMboard as mm

import numpy as np
import copy


class GAsim():

    """An instance of a genetic algorithm to solve Mastermind."""

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
        '''Implement Bernier objective function.

        `guess` is in the form of a list or numpy array.

        "Each combination is compared with the previously played guess, the
        number of different white and black pegs are computed and added to the
        cost... In this way, the cost increases with the number of unsatisfied
        rules."

        The idea here is that given a new guess, would all the different
        responses for the previous attempts be satisfied if the new guess were
        actually the correct answer?  If so, then the new guess could possibly
        be the secret code. Otherwise, it cannot possibly be the new code,
        but may be close to it.'''

        assert self.game._codeOK(guess)
        C = 0

        for (c, r) in self._prev_guesses:
            # get what response for a code would have been if the
            # guess is actually the right code, for each previous guess
            b, w = self.game.check_guess(guess=guess, answer=c)
            assert (b + w) <= self.game._L

            diffw = r[1] - w
            diffb = r[0] - b
            C += abs(diffw) + abs(diffw + diffb)

        return C

    def evolve_population(self, pop_size, gen_size):
        """
        :param pop_size: maximum size for a given population
        :param gen_size: maximum number of population generations
        """
        # initialize population
        population = np.random.choice(self.game._C, (pop_size, self.game._L), replace=True)
        population = [list(item) for item in population]

        elite = []
        h = 1
        # TODO
        # len(elite) <= pop_size and h <= gen_size
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

            # Pick ones where score is 0
            eligibles = [e for (score, e) in pop_score if score == 0]

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
        """
        Up to number of crossovers equal to self.crossover_points, each with probability of self.crossover_prob,
        crosses code1 with code2
        """
        result = copy.copy(code1)

        for i in range(self.game._L):
            if np.random.rand() <= self.crossover_prob:
                result[i] = code2[i]

        return result

    def mutate(self, code):
        """
        With a probability of self.crossover_mutation_prob, crossover is followed by a mutation that replaces
        the color of one randomly chosen position by a random other color
        """
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
        """
        With a probability of self.permutation_prob, colors of two random positions are switched
        """
        result = copy.copy(code)

        if np.random.rand() <= self.permutation_prob:
            pos_a, pos_b = np.random.choice(self.game._L, 2, replace=False)

            result[pos_a], result[pos_b] = result[pos_b], result[pos_a]

        return result

    def invert(self, code):
        """
        with probability of self.inversion_prob, two randomly chosen positions have their sequence of colors
        between these two positions inverted
        """
        result = copy.copy(code)

        if np.random.rand() <= self.inversion_prob:
            pos_range = np.random.choice(self.game._L, 2, replace=False)
            pos1, pos2 = np.sort(pos_range)
            flipped = np.fliplr([result[pos1:pos2]])[0]
            result[pos1:pos2] = flipped

        return result

    def runGA(self, cl=4, nc=6, code=None, silent=False, initial=None):
        """
        Set up board and run GA algorithm
        """
        # if code is provided, it must be in the form of a list or 1-D numpy array
        self.game = mm.MMboard(codelength=cl, numcolors=nc, suppress_output=silent)
        if code is None:
            self.game.set_code()
        else:
            self.game.set_code(code)

        # initial guess
        if initial:
            init_guess = initial
        elif cl == 4 and nc == 6:
            init_guess = [1, 1, 2, 3]
        else:
            init_guess = list(np.random.randint(0, nc, cl))

        # play the guess to get a response of colored (b) and white pegs.
        response = self.game.guess_code(init_guess)
        self._prev_guesses.append((init_guess, response))

        while self.game.gameover is False:
            eligibles = self.evolve_population(self.max_pop, self.max_gen)

            code = eligibles.pop()
            while code in [c for (c, r) in self._prev_guesses]:
                if not eligibles:
                    continue
                else:
                    code = eligibles.pop()

            response = self.game.guess_code(code)
            self._prev_guesses.append((code, response))

        return self.game.n_guesseds

if __name__ == "__main__":
    s = GAsim()
    s.runGA() # same as runGA(cl=4, nc=6, silent=False)

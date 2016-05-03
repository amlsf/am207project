# mle.py
# v 1.0  30 April 2016 [JRM]

######################
#
# Submission by Reinier Maat (Harvard ID: 10985439) for
# AM 207 - Stochastic Methods for Data Analysis, Inference and Optimization
#
# Course Project
#
######################

# runs maximum likelihood estimator for Mastermind. Amounts to constrained random search

'''
Example:

>>> python mle.py

Code successfully initialized to  [1 4 4 4]

guess #1 of 10: you guessed  [4 4 0 0]
You have 1 right item(s) in the right place, and
  1 right item(s) but in the wrong place

guess #2 of 10: you guessed  [4 0 2 5]
You have 0 right item(s) in the right place, and
  1 right item(s) but in the wrong place

guess #3 of 10: you guessed  [0 3 0 3]
You have 0 right item(s) in the right place, and
  0 right item(s) but in the wrong place

guess #4 of 10: you guessed  [1 4 4 4]
You have 4 right item(s) in the right place
You win!
'''

from MMboard import MMboard 
import numpy as np
import itertools

# Maximum likelihood estimation
def maximum_likelihood(cl=4, nc=6, code=None, silent=False):
    
        # Get board ready to play
    board = MMboard(codelength=cl, numcolors=nc, suppress_output=silent)
    board.set_code() if not code else board.set_code(code)
    
    # Value stores
    solutions = np.array(list(itertools.product(range(nc), repeat=cl)))
    guesses = []
    responses = []
    
    # Emulates response given some hidden code
    def response(guess, code):
        equals = guess == code
        in_both = np.intersect1d(guess[equals == False], code[equals == False])
        guess_count = np.array(map(lambda a: np.count_nonzero(code[equals == False] == a), in_both))
        code_count = np.array(map(lambda a: np.count_nonzero(guess[equals == False] == a), in_both))
        wrong_places = np.array(map(lambda (a, b): a if a <= b else b, zip(guess_count, code_count)))
        return len(equals.nonzero()[0]), int(wrong_places.sum())
    
    # Likelihood defined as above
    likelihood = lambda guess, outcome, code: 1 if response(guess, code) == outcome else 0
    
    n_guesses = 0
    
    # Run as long we're not game over
    while not board.gameover:
        
        # Make an informed guess from set of solutions with maximum likelihood
        guess = solutions[np.random.randint(0, len(solutions))]
        outcome = board.guess_code(guess)
        # print outcome, response(guess, board._code)
        guesses.append(guess)
        responses.append(outcome)
        
        # Compute feasible solutions left
        valid_indices = []
        for i in xrange(len(solutions)):
            solution = solutions[i]
            total_likelihood = 1
            
            # Compute likelihood over past outcomes
            for j in xrange(len(guesses)):
                guess = guesses[j]
                outcome = responses[j]
                if likelihood(guess, outcome, solution) == 0:
                    total_likelihood = 0
                    break
            
            # Keep only solutions with likelihood > 0
            if total_likelihood > 0:
                valid_indices.append(i)
        
        # Set solutions
        solutions = solutions[valid_indices]
        
        # Set guess count
        n_guesses += 1
        
    return n_guesses

if __name__ == "__main__":
    maximum_likelihood()  # same as random_search(cl=4, nc=6, silent=False)
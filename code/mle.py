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

from MMboard import MMboard 
import numpy as np
import itertools

# Maximum likelihood estimation
def maximum_likelihood(n_code, n_colors, code=None, silent=True):
    
        # Get board ready to play
    board = MMboard(codelength=n_code, numcolors=n_colors, suppress_output=silent)
    board.set_code() if not code else board.set_code(code)
    
    # Value stores
    solutions = np.array(list(itertools.product(range(n_colors), repeat=n_code)))
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
        print outcome, response(guess, board._code)
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

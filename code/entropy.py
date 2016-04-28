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

def entropy(cl=4, nc=6, code=None, silent=False):
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
    response_idx = dict(zip(responses, range(n_responses)))

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
                response_prob = dict(zip(responses, [0.0] * len(responses)))
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

        # play the guess to get a response of colored (b) and white pegs.
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

if __name__ == "__main__":
    np.seterr(divide='ignore') # supress warnings related to log(0)
    entropy()

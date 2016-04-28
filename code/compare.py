# compare.py
# v 1.0  27 April 2016 [KL]

######################
#
# Submission by Kendrick Lo (Harvard ID: 70984997) for
# AM 207 - Stochastic Methods for Data Analysis, Inference and Optimization
#
# Course Project
#
######################

# runs a comparison of the different algorithms for a particular
# set of game parameters
# to run, execute: python compare.py

import numpy as np
import time
import matplotlib.pyplot as plt

# main game functions
# import MMboard as mm

# individual algorithms
import knuth as kn  # Knuth algorithm
import randomsearch as rs  # Random search with constraints algorithm
import annealing as sa  # Simulated annealing
# import __________ as ga  # Genetic Algorithms

# variable alphabet sizes (cardinality)
alphabet_sizes = [4, 5, 6, 7, 8, 9, 10]  # [4, 6, 8, 10]

# code length size (anything bigger than 6 can take very long to run)
length_sizes = [4]

# output modes
silent_mode = True  # detailed output for each iteration
summary_stats = True  # print number of guesses and time for each iteration

# store results of number of guesses and time, indexed by algorithm
# value is a list, first element is a tuple (digits, positions)
results = {}  # Algoname: [((digits, positions), guesses, time) ... ]

# number of simulations to average over
nsims = 20

for c in alphabet_sizes:

    for l in length_sizes:

        print "\n----------------------------------------------"
        print "possible digits: %i, number of positions: %i" % (c, l)
        print "----------------------------------------------"

        # generate random code for use in all algorithms
        secret = np.random.randint(0, c, size=l)

        ################
        print "\n*** KNUTH'S ALGORITHM ***"

        cnt = np.zeros(nsims)
        runtime = np.zeros(nsims)

        for i in range(nsims):
            start = time.time()
            cnt[i] = kn.knuth(cl=l, nc=c, code=secret, silent=silent_mode)
            runtime[i] = time.time() - start

        meancnt = np.mean(cnt)
        cnterror = np.std(cnt)
        meanrun = np.mean(runtime)
        runerror = np.std(runtime)

        if "knuth" not in results:
            results["knuth"] = [((c, l), meancnt, meanrun, cnterror, runerror)]
        else:
            results["knuth"] += [((c, l), meancnt, meanrun, cnterror, runerror)]
        if summary_stats:
            print "avg number of guesses: %.1f (std: %.3f), avg run time: %.3f (std: %.3f)" % (meancnt, cnterror, meanrun, runerror)
        ################

        ################
        print "\n*** RANDOM SEARCH UNDER CONSTRAINTS ***"

        cnt = np.zeros(nsims)
        runtime = np.zeros(nsims)

        for i in range(nsims):
            start = time.time()
            cnt[i] = rs.random_search(cl=l, nc=c, code=secret, silent=silent_mode)
            runtime[i] = time.time() - start

        meancnt = np.mean(cnt)
        cnterror = np.std(cnt)
        meanrun = np.mean(runtime)
        runerror = np.std(runtime)

        if "random_search" not in results:
            results["random_search"] = [((c, l), meancnt, meanrun, cnterror, runerror)]
        else:
            results["random_search"] += [((c, l), meancnt, meanrun, cnterror, runerror)]
        if summary_stats:
            print "avg number of guesses: %.1f (std: %.3f), avg run time: %.3f (std: %.3f)" % (meancnt, cnterror, meanrun, runerror)
        #################

        #################
        print "\n*** SIMULATED ANNEALING ***"

        cnt = np.zeros(nsims)
        runtime = np.zeros(nsims)

        for i in range(nsims):
            start = time.time()
            cnt[i] = sa.SAsim().runSA(cl=l, nc=c, code=secret, silent=silent_mode)
            runtime[i] = time.time() - start

        meancnt = np.mean(cnt)
        cnterror = np.std(cnt)
        meanrun = np.mean(runtime)
        runerror = np.std(runtime)

        if "SA" not in results:
            results["SA"] = [((c, l), meancnt, meanrun, cnterror, runerror)]
        else:
            results["SA"] += [((c, l), meancnt, meanrun, cnterror, runerror)]
        if summary_stats:
            print "avg number of guesses: %.1f (std: %.3f), avg run time: %.3f (std: %.3f)" % (meancnt, cnterror, meanrun, runerror)
        #################

        #################
        # print "\n*** GENETIC ALGORITHMS ***"
        #
        # cnt = np.zeros(nsims)
        # runtime = np.zeros(nsims)
        #
        # for i in range(nsims):
        #     start = time.time()
        #     cnt[i] = ga.##########(cl=l, nc=c, code=secret, silent=silent_mode)
        #     runtime[i] = time.time() - start
        #
        # meancnt = np.mean(cnt)
        # cnterror = np.std(cnt)
        # meanrun = np.mean(runtime)
        # runerror = np.std(runtime)
        #
        # if "Genetic" not in results:
        #     results["Genetic"] = [((c, l), meancnt, meanrun, cnterror, runerror)]
        # else:
        #     results["Genetic"] += [((c, l), meancnt, meanrun, cnterror, runerror)]
        # if summary_stats:
        #     print "avg number of guesses: %.1f (std: %.3f), avg run time: %.3f (std: %.3f)" % (meancnt, cnterror, meanrun, runerror)
        #################

# print results dictionary
print results

##########
# plots
##########

print "printing plot of number of guesses by character set (code length 4)"
plt.figure()
plt.title("Number of Guesses (fixed code length 4)")
plt.xlabel("characters")
plt.ylabel("guesses")
plt.xlim(3, 11, 1)
plt.ylim(1, 11, 1)
for model in results.keys():
    # ((c, l), cnt, runtime)
    points_x, points_y, err = [], [], []
    for point in results[model]:
        if point[0][1] == 4:  # fix to standard length
            points_x += [point[0][0]]
            points_y += [point[1]]
            err += [point[3]]
    plt.errorbar(points_x, points_y, yerr=err, label=model)
plt.legend(loc="best")
plt.show()

print "printing plot of execution time by character set (code length 4)"
plt.figure()
plt.title("Execution Time (fixed code length 4)")
plt.xlabel("characters")
plt.ylabel("execution time (seconds)")
plt.xlim(3, 11, 1)
for model in results.keys():
    # ((c, l), cnt, runtime)
    points_x, points_y, err = [], [], []
    for point in results[model]:
        if point[0][1] == 4:  # fix to standard length
            points_x += [point[0][0]]
            points_y += [point[2]]
            err += [point[4]]
    plt.ylim(0, 5)
    plt.errorbar(points_x, points_y, yerr=err, label=model)
plt.legend(loc="best")
plt.show()

# print "printing plot of number of guesses by word length (6 possible digits)"
# plt.figure()
# plt.title("Number of Guesses by word length (6 possible digits)")
# plt.xlabel("word length")
# plt.ylabel("guesses")
# plt.xlim(3, 11, 1)
# plt.ylim(1, 11, 1)
# for model in results.keys():
#     # ((c, l), cnt, runtime)
#     points_x, points_y, err = [], [], []
#     for point in results[model]:
#         if point[0][0] == 6:  # fix to standard length
#             points_x += [point[0][1]]
#             points_y += [point[1]]
#             err += [point[3]]
#     plt.errorbar(points_x, points_y, yerr=err, label=model)
# plt.legend(loc="best")
# plt.show()
#
# print "printing plot of number of execution time by word length (6 possible digits)"
# plt.figure()
# plt.title("Execution Time by word length (6 possible digits)")
# plt.xlabel("word length")
# plt.ylabel("execution time (seconds)")
# plt.xlim(3, 11, 1)
# for model in results.keys():
#     # ((c, l), cnt, runtime)
#     points_x, points_y, err = [], [], []
#     for point in results[model]:
#         if point[0][0] == 6:  # fix to standard length
#             points_x += [point[0][1]]
#             points_y += [point[2]]
#             err += [point[4]]
#     plt.ylim(0, 5)
#     plt.errorbar(points_x, points_y, yerr=err, label=model)
# plt.legend(loc="best")
# plt.show()

'''
Sample output: (silent_mode = True)

----------------------------------------------
possible digits: 4, number of positions: 4
----------------------------------------------

*** KNUTH'S ALGORITHM ***
avg number of guesses: 3.0 (std: 0.000), avg run time: 0.035 (std: 0.002)

*** RANDOM SEARCH UNDER CONSTRAINTS ***
avg number of guesses: 3.6 (std: 0.792), avg run time: 0.002 (std: 0.000)

*** SIMULATED ANNEALING ***
avg number of guesses: 6.7 (std: 1.584), avg run time: 0.016 (std: 0.022)

----------------------------------------------
possible digits: 5, number of positions: 4
----------------------------------------------

*** KNUTH'S ALGORITHM ***
avg number of guesses: 4.0 (std: 0.000), avg run time: 0.261 (std: 0.006)

*** RANDOM SEARCH UNDER CONSTRAINTS ***
avg number of guesses: 4.2 (std: 0.748), avg run time: 0.006 (std: 0.000)

*** SIMULATED ANNEALING ***
avg number of guesses: 8.7 (std: 1.824), avg run time: 0.015 (std: 0.021)

----------------------------------------------
possible digits: 6, number of positions: 4
----------------------------------------------

*** KNUTH'S ALGORITHM ***
avg number of guesses: 4.0 (std: 0.000), avg run time: 0.181 (std: 0.006)

*** RANDOM SEARCH UNDER CONSTRAINTS ***
avg number of guesses: 4.8 (std: 0.766), avg run time: 0.013 (std: 0.003)

*** SIMULATED ANNEALING ***
avg number of guesses: 7.1 (std: 2.022), avg run time: 0.078 (std: 0.271)

----------------------------------------------
possible digits: 7, number of positions: 4
----------------------------------------------

*** KNUTH'S ALGORITHM ***
avg number of guesses: 5.0 (std: 0.000), avg run time: 4.792 (std: 0.019)

*** RANDOM SEARCH UNDER CONSTRAINTS ***
avg number of guesses: 5.2 (std: 0.994), avg run time: 0.023 (std: 0.001)

*** SIMULATED ANNEALING ***
avg number of guesses: 8.4 (std: 2.156), avg run time: 0.109 (std: 0.234)

----------------------------------------------
possible digits: 8, number of positions: 4
----------------------------------------------

*** KNUTH'S ALGORITHM ***
avg number of guesses: 5.0 (std: 0.000), avg run time: 7.582 (std: 0.034)

*** RANDOM SEARCH UNDER CONSTRAINTS ***
avg number of guesses: 5.3 (std: 0.654), avg run time: 0.040 (std: 0.004)

*** SIMULATED ANNEALING ***
avg number of guesses: 8.8 (std: 1.824), avg run time: 0.171 (std: 0.335)

----------------------------------------------
possible digits: 9, number of positions: 4
----------------------------------------------

*** KNUTH'S ALGORITHM ***
avg number of guesses: 6.0 (std: 0.000), avg run time: 39.936 (std: 1.903)

*** RANDOM SEARCH UNDER CONSTRAINTS ***
avg number of guesses: 6.0 (std: 1.023), avg run time: 0.071 (std: 0.011)

*** SIMULATED ANNEALING ***
avg number of guesses: 8.6 (std: 1.855), avg run time: 0.280 (std: 0.517)

----------------------------------------------
possible digits: 10, number of positions: 4
----------------------------------------------

*** KNUTH'S ALGORITHM ***
avg number of guesses: 6.0 (std: 0.000), avg run time: 141.457 (std: 2.085)

*** RANDOM SEARCH UNDER CONSTRAINTS ***
avg number of guesses: 5.7 (std: 1.229), avg run time: 0.095 (std: 0.004)

*** SIMULATED ANNEALING ***
avg number of guesses: 8.8 (std: 2.308), avg run time: 0.142 (std: 0.308)

'''

# Exploring Optimization Strategies for *Mastermind*

## Harvard AM207: Stochastic Methods for Data Analysis, Inference and Optimization (Spring 2016)

This repository includes all our working files for our final project. The full codebase is included in the `code` folder. The relevant files are:

- `MMboard.py`: Mastermind game interface
- `knuth.py`: Knuth's five-guess algorithm
- `randomsearch.py`: Random search with constraints, aka random sampling from posterior
- `entropy.py`: Shannon entropy maximization, including 2 global search implementations and 2 local search implementations (simulated annealing and genetic algorithms)
- `annealing.py`: Simulated annealing, using Bernier objective function
- `genetic.py`: Genetic algorithms, using Bernier objective function
- `mle.py`: Maximum likelihood estimator, equivalent to random search with constraints
- `compare_colors.py`: Runs all methods for codes of length 4 and 4-10 possible colors
- `compare_lengths.py`: Runs all methods for 2 possible colors and codes of length 4-10

Individual optimization methods can be tested by running the relevant script (e.g. `python entropy.py`). The entire set of optimization methods can be tested by running one of the comparison scripts (e.g. `python compare_colors.py`).

All code was developed using Python 2.7 and may not run as expected on other builds.



Please also refer to our screencast: https://youtu.be/9VpXru8dRGA



**Team Members:**

Gioia Dominedo  |  [@dominedo](https://github.com/dominedo)  |  dominedo@g.harvard.edu

Amy Lee  |  [@amlsf](https://github.com/amlsf)  |  amymaelee@g.harvard.edu

Kendrick Lo  |  [@ppgmg](https://github.com/ppgmg)  |  klo@g.harvard.edu

Reinier Maat  |  [@1Reinier](https://github.com/1Reinier)  |  maat@g.harvard.edu
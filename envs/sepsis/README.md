Sepsis domain is modified based on https://github.com/clinicalml/gumbel-max-scm

## Dependencies

This code was run using Python 3.7 in a `conda` environment:  Running the following commands should cover all the dependencies of the code (e.g., installing `pandas` will install `numpy`, and so on)
```
conda install jupyter
conda install pandas
conda install seaborn
conda install tqdm
pip install pymdptoolbox
```

## Updated Simulator

As we receive suggestions for improving the realism of the sepsis simulator, we will collect them in the `sim-v2` branch of this repository, in case it is useful for others.  The `master` branch will remain unchanged to facilitate reproduction of the original paper.

## Acknowledgements

First, we would like to thank Christina X Ji and [Fredrik D. Johansson](http://www.mit.edu/~fredrikj/) for their work on an earlier version of the sepsis simulator we use in this paper.

Second, for some of the code used in the posterior inference over Gumbel variables, we borrowed from Chris Maddison's blog post [here](https://cmaddis.github.io/gumbel-machinery)

Finally, in this repository (in `pymdptoolbox/`) we have the source code for the `pymdptoolbox` package from [sawcordwell/pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox), which is in turn based the toolset described in `Chades I, Chapron G, Cros M-J, Garcia F & Sabbadin R (2014) 'MDPtoolbox: a multi-platform toolbox to solve stochastic dynamic programming problems', Ecography, vol. 37, no. 9, pp. 916–920, doi 10.1111/ecog.00888.`  We reproduce it here because we needed to make a slight modification to the `mdp` class to bypass certain checks; in particular, it checks for whether or not the rows of the transition matrix sum to one, but can fail due to floating-point inaccuraries - we replace this check in the main code with an assertion using `np.allclose` instead of checking for strict equality.

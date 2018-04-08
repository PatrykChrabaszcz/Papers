### An Efficient Approach for Assessing Hyperparameter Importance (ICML 2014) Hutter et al. (University of Freiburg)
- Builds a model that predicts performance of a given hyperparameters set (configuration). 
- Uses this model to asses the influence of hyperparameters.
- Checks how performance of one parameter varies based on all possible values of other parameters (has to marginalize over them).
- Provides a linear algorithm for marginalization of hyperparameters (when using random forest as a model).

From the paper:
- "In low-dimensional problems with numerical hyperparameters, the best available hyperparameter optimization methods use Bayesian optimization based on Gaussian process models, whereas in high-dimensional and discrete spaces, tree-based models, and in particular random forests, are more successful"
- "However, not much work has been done on quantifying the relative importance of the hyperparameters that do matter."
- "Here, we provide the first efficient and exact method for deriving functional ANOVA sensitivity indices for random forests."
- "Optimally, algorithm designers would like to know how their hyperparameters affect performance in general, not just in the context of a single fixed instantiation of the remaining hyperparameters, but across all their instantiations."

TODO: Finish this paper


### Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization (Mar 2016) Li et.al (University of California)
https://arxiv.org/abs/1603.06560

- Claims that Hyperband gives 5x, 30x speedups compared to Bayesian Optimization

TODO:Finish this paper

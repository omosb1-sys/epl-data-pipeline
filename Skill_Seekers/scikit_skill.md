---
name: scikit-learn
description: Scikit-learn machine learning library for Python.
---

# Scikit-Learn Skill

Scikit-learn machine learning library for python., generated from official documentation.

## When to Use This Skill

This skill should be triggered when:
- Working with scikit-learn
- Asking about scikit-learn features or APIs
- Implementing scikit-learn solutions
- Debugging scikit-learn code
- Learning scikit-learn best practices

## Quick Reference

### Common Patterns

**Pattern 1:** This function generates a GraphViz representation of the decision tree, which is then written into out_file. Once exported, graphical renderings can be generated using, for example:

```
out_file
```

**Pattern 2:** Note that this type is the most specific type that can be inferred. For example:

```
binary
```

**Pattern 3:** Usage example:

```
>>> from sklearn.linear_model import TweedieRegressor
>>> reg = TweedieRegressor(power=1, alpha=0.5, link='log')
>>> reg.fit([[0, 0], [0, 1], [2, 2]], [0, 1, 2])
TweedieRegressor(alpha=0.5, link='log', power=1)
>>> reg.coef_
array([0.2463, 0.4337])
>>> reg.intercept_
np.float64(-0.7638)
```

**Pattern 4:** Here is an example:

```
>>> from sklearn.metrics.cluster import contingency_matrix
>>> x = ["a", "a", "a", "b", "b", "b"]
>>> y = [0, 0, 1, 1, 2, 2]
>>> contingency_matrix(x, y)
array([[2, 1, 0],
       [0, 1, 2]])
```

**Pattern 5:** The functional form of the G function used in the approximation to neg-entropy. Could be either ‘logcosh’, ‘exp’, or ‘cube’. You can also provide your own function. It should return a tuple containing the value of the function, and of its derivative, in the point. The derivative should be averaged along its last dimension. Example:

```
def my_g(x):
    return x ** 3, (3 * x ** 2).mean(axis=-1)
```

**Pattern 6:** To perform the train and test split, use the indices for the train and test subsets yielded by the generator output by the split() method of the cross-validation splitter. For example:

```
split()
```

**Pattern 7:** Fictitious Example: Let’s make the above arguments more tangible. Consider a setting in network reliability engineering, such as maintaining stable internet or Wi-Fi connections. As provider of the network, you have access to the dataset of log entries of network connections containing network load over time and many interesting features. Your goal is to improve the reliability of the connections. In fact, you promise your customers that on at least 99% of all days there are no connection discontinuities larger than 1 minute. Therefore, you are interested in a prediction of the 99% quantile (of longest connection interruption duration per day) in order to know in advance when to add more bandwidth and thereby satisfy your customers. So the target functional is the 99% quantile. From the table above, you choose the pinball loss as scoring function (fair enough, not much choice given), for model training (e.g. HistGradientBoostingRegressor(loss="quantile", quantile=0.99)) as well as model evaluation (mean_pinball_loss(..., alpha=0.99) - we apologize for the different argument names, quantile and alpha) be it in grid search for finding hyperparameters or in comparing to other models like QuantileRegressor(quantile=0.99).

```
HistGradientBoostingRegressor(loss="quantile", quantile=0.99)
```

**Pattern 8:** One typical use case is to wrap an existing metric function from the library with non-default values for its parameters, such as the beta parameter for the fbeta_score function:

```
beta
```

### Example Code Patterns

**Example 1** (json):
```json
>>> print(regr.named_steps['linearsvr'].coef_)
[18.582 27.023 44.357 64.522]
>>> print(regr.named_steps['linearsvr'].intercept_)
[-4.]
>>> print(regr.predict([[0, 0, 0, 0]]))
[-2.384]
```

## Reference Files

This skill includes comprehensive documentation in `references/`:

- **modules.md** - Modules documentation
- **other.md** - Other documentation

Use `view` to read specific reference files when detailed information is needed.

## Working with This Skill

### For Beginners
Start with the getting_started or tutorials reference files for foundational concepts.

### For Specific Features
Use the appropriate category reference file (api, guides, etc.) for detailed information.

### For Code Examples
The quick reference section above contains common patterns extracted from the official docs.

## Resources

### references/
Organized documentation extracted from official sources. These files contain:
- Detailed explanations
- Code examples with language annotations
- Links to original documentation
- Table of contents for quick navigation

### scripts/
Add helper scripts here for common automation tasks.

### assets/
Add templates, boilerplate, or example projects here.

## Notes

- This skill was automatically generated from official documentation
- Reference files preserve the structure and examples from source docs
- Code examples include language detection for better syntax highlighting
- Quick reference patterns are extracted from common usage examples in the docs

## Updating

To refresh this skill with updated documentation:
1. Re-run the scraper with the same configuration
2. The skill will be rebuilt with the latest information

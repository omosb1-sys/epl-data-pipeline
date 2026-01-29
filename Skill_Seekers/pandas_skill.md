---
name: pandas
description: Pandas data analysis and manipulation library. Use for DataFrames, Series, cleaning, and transformation.
---

# Pandas Skill

Pandas data analysis and manipulation library. use for dataframes, series, cleaning, and transformation., generated from official documentation.

## When to Use This Skill

This skill should be triggered when:
- Working with pandas
- Asking about pandas features or APIs
- Implementing pandas solutions
- Debugging pandas code
- Learning pandas best practices

## Quick Reference

### Common Patterns

**Pattern 1:** Expressions that would result in an object dtype or involve datetime operations because of NaT must be evaluated in Python space, but part of an expression can still be evaluated with numexpr. For example:

```
NaT
```

**Pattern 2:** the necessary format to pass styles to .set_table_styles() is as a list of dicts, each with a CSS-selector tag and CSS-properties. Properties can either be a list of 2-tuples, or a regular CSS-string, for example:

```
[13]:
```

**Pattern 3:** For example:

```
In [32]: pd.Period("2011-01")
Out[32]: Period('2011-01', 'M')

In [33]: pd.Period("2012-05", freq="D")
Out[33]: Period('2012-05-01', 'D')
```

**Pattern 4:** By default, each row has an equal probability of being selected, but if you want rows to have different probabilities, you can pass the sample function sampling weights as weights. These weights can be a list, a NumPy array, or a Series, but they must be of the same length as the object you are sampling. Missing values will be treated as a weight of zero, and inf values are not allowed. If weights do not sum to 1, they will be re-normalized by dividing all weights by the sum of the weights. For example:

```
sample
```

**Pattern 5:** You can get the value of the frame where column b has values between the values of columns a and c. For example:

```
b
```

**Pattern 6:** To format values Styler.format() should be used prior to calling Styler.to_latex, as well as other methods such as Styler.hide() for example:

```
Styler.format()
```

**Pattern 7:** The MultiIndex keeps all the defined levels of an index, even if they are not actually used. When slicing an index, you may notice this. For example:

```
MultiIndex
```

**Pattern 8:** A RangeIndex will behave similarly to a Index with an int64 dtype and operations on a RangeIndex, whose result cannot be represented by a RangeIndex, but should have an integer dtype, will be converted to an Index with int64. For example:

```
RangeIndex
```

### Example Code Patterns

**Example 1** (json):
```json
>>> df = pd.DataFrame({'name': ['Raphael', 'Donatello'],
...                    'mask': ['red', 'purple'],
...                    'weapon': ['sai', 'bo staff']})
>>> df.to_csv('out.csv', index=False)
```

**Example 2** (json):
```json
In [15]: pd.Timedelta(pd.offsets.Second(2))
Out[15]: Timedelta('0 days 00:00:02')
```

**Example 3** (typescript):
```typescript
In [1]: s = pd.Series(["a", "b", "c", "a"], dtype="category")

In [2]: s
Out[2]: 
0    a
1    b
2    c
3    a
dtype: category
Categories (3, object): ['a', 'b', 'c']
```

**Example 4** (json):
```json
In [3]: df = pd.DataFrame({"A": ["a", "b", "c", "a"]})

In [4]: df["B"] = df["A"].astype("category")

In [5]: df
Out[5]: 
   A  B
0  a  a
1  b  b
2  c  c
3  a  a
```

## Reference Files

This skill includes comprehensive documentation in `references/`:

- **api.md** - Api documentation
- **reference.md** - Reference documentation
- **user_guide.md** - User Guide documentation

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

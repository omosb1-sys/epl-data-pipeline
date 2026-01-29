---
name: scipy
description: SciPy library for scientific computing and technical computing.
---

# Scipy Skill

Scipy library for scientific computing and technical computing., generated from official documentation.

## When to Use This Skill

This skill should be triggered when:
- Working with scipy
- Asking about scipy features or APIs
- Implementing scipy solutions
- Debugging scipy code
- Learning scipy best practices

## Quick Reference

### Common Patterns

**Pattern 1:** The type of arrays returned is dependent on the type of operation, but it is, in most cases, equal to the type of the input. If, however, the output argument is used, the type of the result is equal to the type of the specified output argument. If no output argument is given, it is still possible to specify what the result of the output should be. This is done by simply assigning the desired numpy type object to the output argument. For example:

```
numpy
```

**Pattern 2:** Sometimes, it is convenient to choose a different origin for the kernel. For this reason, most functions support the origin parameter, which gives the origin of the filter relative to its center. For example:

```
>>> a = [0, 0, 0, 1, 0, 0, 0]
>>> correlate1d(a, [1, 1, 1], origin = -1)
array([0, 1, 1, 1, 0, 0, 0])
```

**Pattern 3:** The result is that the object (marker = 2) is smaller because the second marker was processed earlier. This may not be the desired effect if the first marker was supposed to designate a background object. Therefore, watershed_ift treats markers with a negative value explicitly as background markers and processes them after the normal markers. For instance, replacing the first marker by a negative marker gives a result similar to the first example:

```
watershed_ift
```

**Pattern 4:** The function find_objects returns slices for all objects, unless the max_label parameter is larger then zero, in which case only the first max_label objects are returned. If an index is missing in the label array, None is return instead of a slice. For example:

```
find_objects
```

**Pattern 5:** This is three fewer links than our initial example: the path from “ape” to “man” is only five steps.

```
>>> from scipy.sparse.csgraph import connected_components
>>> N_components, component_list = connected_components(graph)
>>> print(N_components)
15    # may vary
```

**Pattern 6:** For example:

```
"void (double)"
"double (double, int *, void *)"
```

**Pattern 7:** Here is an example:

```
>>> from scipy.optimize import minimize_scalar
>>> f = lambda x: (x - 2) * (x + 1)**2
>>> res = minimize_scalar(f, method='brent')
>>> print(res.x)
1.0
```

**Pattern 8:** There are several important points to note about this example:

```
slow_func
```

## Reference Files

This skill includes comprehensive documentation in `references/`:

- **doc.md** - Doc documentation

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

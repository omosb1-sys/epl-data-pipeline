---
name: seaborn
description: Seaborn statistical data visualization library based on matplotlib.
---

# Seaborn Skill

Seaborn statistical data visualization library based on matplotlib., generated from official documentation.

## When to Use This Skill

This skill should be triggered when:
- Working with seaborn
- Asking about seaborn features or APIs
- Implementing seaborn solutions
- Debugging seaborn code
- Learning seaborn best practices

## Quick Reference

### Common Patterns

**Pattern 1:** When you want to represent multiple categories in a plot, you typically should vary the color of the elements. Consider this simple example: in which of these two plots is it easier to count the number of triangular points?

```
color_palette()
```

**Pattern 2:** The seaborn.objects namespace will provide access to all of the relevant classes. The most important is Plot. You specify plots by instantiating a Plot object and calling its methods. Let’s see a simple example:

```
seaborn.objects
```

**Pattern 3:** Once you’ve drawn a plot using FacetGrid.map() (which can be called multiple times), you may want to adjust some aspects of the plot. There are also a number of methods on the FacetGrid object for manipulating the figure at a higher level of abstraction. The most general is FacetGrid.set(), and there are other more specialized methods like FacetGrid.set_axis_labels(), which respects the fact that interior facets do not have axis labels. For example:

```
FacetGrid.map()
```

## Reference Files

This skill includes comprehensive documentation in `references/`:

- **api.md** - Api documentation
- **generated.md** - Generated documentation
- **tutorial.md** - Tutorial documentation

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

# Seaborn - Tutorial

**Pages:** 14

---

## Overview of seaborn plotting functions#

**URL:** https://seaborn.pydata.org/tutorial/function_overview.html

**Contents:**
- Overview of seaborn plotting functions#
- Similar functions for similar tasks#
- Figure-level vs. axes-level functions#
  - Axes-level functions make self-contained plots#
  - Figure-level functions own their figure#
  - Customizing plots from a figure-level function#
  - Specifying figure sizes#
  - Relative merits of figure-level functions#
- Combining multiple views on the data#

Most of your interactions with seaborn will happen through a set of plotting functions. Later chapters in the tutorial will explore the specific features offered by each function. This chapter will introduce, at a high-level, the different kinds of functions that you will encounter.

The seaborn namespace is flat; all of the functionality is accessible at the top level. But the code itself is hierarchically structured, with modules of functions that achieve similar visualization goals through different means. Most of the docs are structured around these modules: you’ll encounter names like “relational”, “distributional”, and “categorical”.

For example, the distributions module defines functions that specialize in representing the distribution of datapoints. This includes familiar methods like the histogram:

Along with similar, but perhaps less familiar, options such as kernel density estimation:

Functions within a module share a lot of underlying code and offer similar features that may not be present in other components of the library (such as multiple="stack" in the examples above). They are designed to facilitate switching between different visual representations as you explore a dataset, because different representations often have complementary strengths and weaknesses.

In addition to the different modules, there is a cross-cutting classification of seaborn functions as “axes-level” or “figure-level”. The examples above are axes-level functions. They plot data onto a single matplotlib.pyplot.Axes object, which is the return value of the function.

In contrast, figure-level functions interface with matplotlib through a seaborn object, usually a FacetGrid, that manages the figure. Each module has a single figure-level function, which offers a unitary interface to its various axes-level functions. The organization looks a bit like this:

For example, displot() is the figure-level function for the distributions module. Its default behavior is to draw a histogram, using the same code as histplot() behind the scenes:

To draw a kernel density plot instead, using the same code as kdeplot(), select it using the kind parameter:

You’ll notice that the figure-level plots look mostly like their axes-level counterparts, but there are a few differences. Notably, the legend is placed outside the plot. They also have a slightly different shape (more on that shortly).

The most useful feature offered by the figure-level functions is that they can easily create figures with multiple subplots. For example, instead of stacking the three distributions for each species of penguins in the same axes, we can “facet” them by plotting each distribution across the columns of the figure:

The figure-level functions wrap their axes-level counterparts and pass the kind-specific keyword arguments (such as the bin size for a histogram) down to the underlying function. That means they are no less flexible, but there is a downside: the kind-specific parameters don’t appear in the function signature or docstrings. Some of their features might be less discoverable, and you may need to look at two different pages of the documentation before understanding how to achieve a specific goal.

The axes-level functions are written to act like drop-in replacements for matplotlib functions. While they add axis labels and legends automatically, they don’t modify anything beyond the axes that they are drawn into. That means they can be composed into arbitrarily-complex matplotlib figures with predictable results.

The axes-level functions call matplotlib.pyplot.gca() internally, which hooks into the matplotlib state-machine interface so that they draw their plots on the “currently-active” axes. But they additionally accept an ax= argument, which integrates with the object-oriented interface and lets you specify exactly where each plot should go:

In contrast, figure-level functions cannot (easily) be composed with other plots. By design, they “own” their own figure, including its initialization, so there’s no notion of using a figure-level function to draw a plot onto an existing axes. This constraint allows the figure-level functions to implement features such as putting the legend outside of the plot.

Nevertheless, it is possible to go beyond what the figure-level functions offer by accessing the matplotlib axes on the object that they return and adding other elements to the plot that way:

The figure-level functions return a FacetGrid instance, which has a few methods for customizing attributes of the plot in a way that is “smart” about the subplot organization. For example, you can change the labels on the external axes using a single line of code:

While convenient, this does add a bit of extra complexity, as you need to remember that this method is not part of the matplotlib API and exists only when using a figure-level function.

To increase or decrease the size of a matplotlib plot, you set the width and height of the entire figure, either in the global rcParams, while setting up the plot (e.g. with the figsize parameter of matplotlib.pyplot.subplots()), or by calling a method on the figure object (e.g. matplotlib.Figure.set_size_inches()). When using an axes-level function in seaborn, the same rules apply: the size of the plot is determined by the size of the figure it is part of and the axes layout in that figure.

When using a figure-level function, there are several key differences. First, the functions themselves have parameters to control the figure size (although these are actually parameters of the underlying FacetGrid that manages the figure). Second, these parameters, height and aspect, parameterize the size slightly differently than the width, height parameterization in matplotlib (using the seaborn parameters, width = height * aspect). Most importantly, the parameters correspond to the size of each subplot, rather than the size of the overall figure.

To illustrate the difference between these approaches, here is the default output of matplotlib.pyplot.subplots() with one subplot:

A figure with multiple columns will have the same overall size, but the axes will be squeezed horizontally to fit in the space:

In contrast, a plot created by a figure-level function will be square. To demonstrate that, let’s set up an empty plot by using FacetGrid directly. This happens behind the scenes in functions like relplot(), displot(), or catplot():

When additional columns are added, the figure itself will become wider, so that its subplots have the same size and shape:

And you can adjust the size and shape of each subplot without accounting for the total number of rows and columns in the figure:

The upshot is that you can assign faceting variables without stopping to think about how you’ll need to adjust the total figure size. A downside is that, when you do want to change the figure size, you’ll need to remember that things work a bit differently than they do in matplotlib.

Here is a summary of the pros and cons that we have discussed above:

Easy faceting by data variables

Many parameters not in function signature

Legend outside of plot by default

Cannot be part of a larger matplotlib figure

Easy figure-level customization

Different API from matplotlib

Different figure size parameterization

Different figure size parameterization

On balance, the figure-level functions add some additional complexity that can make things more confusing for beginners, but their distinct features give them additional power. The tutorial documentation mostly uses the figure-level functions, because they produce slightly cleaner plots, and we generally recommend their use for most applications. The one situation where they are not a good choice is when you need to make a complex, standalone figure that composes multiple different plot kinds. At this point, it’s recommended to set up the figure using matplotlib directly and to fill in the individual components using axes-level functions.

Two important plotting functions in seaborn don’t fit cleanly into the classification scheme discussed above. These functions, jointplot() and pairplot(), employ multiple kinds of plots from different modules to represent multiple aspects of a dataset in a single figure. Both plots are figure-level functions and create figures with multiple subplots by default. But they use different objects to manage the figure: JointGrid and PairGrid, respectively.

jointplot() plots the relationship or joint distribution of two variables while adding marginal axes that show the univariate distribution of each one separately:

pairplot() is similar — it combines joint and marginal views — but rather than focusing on a single relationship, it visualizes every pairwise combination of variables simultaneously:

Behind the scenes, these functions are using axes-level functions that you have already met (scatterplot() and kdeplot()), and they also have a kind parameter that lets you quickly swap in a different representation:

**Examples:**

Example 1 (unknown):
```unknown
penguins = sns.load_dataset("penguins")
sns.histplot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")
```

Example 2 (unknown):
```unknown
sns.kdeplot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")
```

Example 3 (unknown):
```unknown
sns.displot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")
```

Example 4 (unknown):
```unknown
sns.displot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack", kind="kde")
```

---

## Estimating regression fits#

**URL:** https://seaborn.pydata.org/tutorial/regression.html

**Contents:**
- Estimating regression fits#
- Functions for drawing linear regression models#
- Fitting different kinds of models#
- Conditioning on other variables#
- Plotting a regression in other contexts#

Many datasets contain multiple quantitative variables, and the goal of an analysis is often to relate those variables to each other. We previously discussed functions that can accomplish this by showing the joint distribution of two variables. It can be very helpful, though, to use statistical models to estimate a simple relationship between two noisy sets of observations. The functions discussed in this chapter will do so through the common framework of linear regression.

In the spirit of Tukey, the regression plots in seaborn are primarily intended to add a visual guide that helps to emphasize patterns in a dataset during exploratory data analyses. That is to say that seaborn is not itself a package for statistical analysis. To obtain quantitative measures related to the fit of regression models, you should use statsmodels. The goal of seaborn, however, is to make exploring a dataset through visualization quick and easy, as doing so is just as (if not more) important than exploring a dataset through tables of statistics.

The two functions that can be used to visualize a linear fit are regplot() and lmplot().

In the simplest invocation, both functions draw a scatterplot of two variables, x and y, and then fit the regression model y ~ x and plot the resulting regression line and a 95% confidence interval for that regression:

These functions draw similar plots, but regplot() is an axes-level function, and lmplot() is a figure-level function. Additionally, regplot() accepts the x and y variables in a variety of formats including simple numpy arrays, pandas.Series objects, or as references to variables in a pandas.DataFrame object passed to data. In contrast, lmplot() has data as a required parameter and the x and y variables must be specified as strings. Finally, only lmplot() has hue as a parameter.

The core functionality is otherwise similar, though, so this tutorial will focus on lmplot():.

It’s possible to fit a linear regression when one of the variables takes discrete values, however, the simple scatterplot produced by this kind of dataset is often not optimal:

One option is to add some random noise (“jitter”) to the discrete values to make the distribution of those values more clear. Note that jitter is applied only to the scatterplot data and does not influence the regression line fit itself:

A second option is to collapse over the observations in each discrete bin to plot an estimate of central tendency along with a confidence interval:

The simple linear regression model used above is very simple to fit, however, it is not appropriate for some kinds of datasets. The Anscombe’s quartet dataset shows a few examples where simple linear regression provides an identical estimate of a relationship where simple visual inspection clearly shows differences. For example, in the first case, the linear regression is a good model:

The linear relationship in the second dataset is the same, but the plot clearly shows that this is not a good model:

In the presence of these kind of higher-order relationships, lmplot() and regplot() can fit a polynomial regression model to explore simple kinds of nonlinear trends in the dataset:

A different problem is posed by “outlier” observations that deviate for some reason other than the main relationship under study:

In the presence of outliers, it can be useful to fit a robust regression, which uses a different loss function to downweight relatively large residuals:

When the y variable is binary, simple linear regression also “works” but provides implausible predictions:

The solution in this case is to fit a logistic regression, such that the regression line shows the estimated probability of y = 1 for a given value of x:

Note that the logistic regression estimate is considerably more computationally intensive (this is true of robust regression as well). As the confidence interval around the regression line is computed using a bootstrap procedure, you may wish to turn this off for faster iteration (using ci=None).

An altogether different approach is to fit a nonparametric regression using a lowess smoother. This approach has the fewest assumptions, although it is computationally intensive and so currently confidence intervals are not computed at all:

The residplot() function can be a useful tool for checking whether the simple regression model is appropriate for a dataset. It fits and removes a simple linear regression and then plots the residual values for each observation. Ideally, these values should be randomly scattered around y = 0:

If there is structure in the residuals, it suggests that simple linear regression is not appropriate:

The plots above show many ways to explore the relationship between a pair of variables. Often, however, a more interesting question is “how does the relationship between these two variables change as a function of a third variable?” This is where the main differences between regplot() and lmplot() appear. While regplot() always shows a single relationship, lmplot() combines regplot() with FacetGrid to show multiple fits using hue mapping or faceting.

The best way to separate out a relationship is to plot both levels on the same axes and to use color to distinguish them:

Unlike relplot(), it’s not possible to map a distinct variable to the style properties of the scatter plot, but you can redundantly code the hue variable with marker shape:

To add another variable, you can draw multiple “facets” with each level of the variable appearing in the rows or columns of the grid:

A few other seaborn functions use regplot() in the context of a larger, more complex plot. The first is the jointplot() function that we introduced in the distributions tutorial. In addition to the plot styles previously discussed, jointplot() can use regplot() to show the linear regression fit on the joint axes by passing kind="reg":

Using the pairplot() function with kind="reg" combines regplot() and PairGrid to show the linear relationship between variables in a dataset. Take care to note how this is different from lmplot(). In the figure below, the two axes don’t show the same relationship conditioned on two levels of a third variable; rather, PairGrid() is used to show multiple relationships between different pairings of the variables in a dataset:

Conditioning on an additional categorical variable is built into both of these functions using the hue parameter:

**Examples:**

Example 1 (unknown):
```unknown
tips = sns.load_dataset("tips")
sns.regplot(x="total_bill", y="tip", data=tips);
```

Example 2 (unknown):
```unknown
sns.lmplot(x="total_bill", y="tip", data=tips);
```

Example 3 (unknown):
```unknown
sns.lmplot(x="size", y="tip", data=tips);
```

Example 4 (unknown):
```unknown
sns.lmplot(x="size", y="tip", data=tips, x_jitter=.05);
```

---

## Visualizing categorical data#

**URL:** https://seaborn.pydata.org/tutorial/categorical.html

**Contents:**
- Visualizing categorical data#
- Categorical scatterplots#
- Comparing distributions#
  - Boxplots#
  - Violinplots#
- Estimating central tendency#
  - Bar plots#
  - Point plots#
- Showing additional dimensions#

In the relational plot tutorial we saw how to use different visual representations to show the relationship between multiple variables in a dataset. In the examples, we focused on cases where the main relationship was between two numerical variables. If one of the main variables is “categorical” (divided into discrete groups) it may be helpful to use a more specialized approach to visualization.

In seaborn, there are several different ways to visualize a relationship involving categorical data. Similar to the relationship between relplot() and either scatterplot() or lineplot(), there are two ways to make these plots. There are a number of axes-level functions for plotting categorical data in different ways and a figure-level interface, catplot(), that gives unified higher-level access to them.

It’s helpful to think of the different categorical plot kinds as belonging to three different families, which we’ll discuss in detail below. They are:

Categorical scatterplots:

stripplot() (with kind="strip"; the default)

swarmplot() (with kind="swarm")

Categorical distribution plots:

boxplot() (with kind="box")

violinplot() (with kind="violin")

boxenplot() (with kind="boxen")

Categorical estimate plots:

pointplot() (with kind="point")

barplot() (with kind="bar")

countplot() (with kind="count")

These families represent the data using different levels of granularity. When deciding which to use, you’ll have to think about the question that you want to answer. The unified API makes it easy to switch between different kinds and see your data from several perspectives.

In this tutorial, we’ll mostly focus on the figure-level interface, catplot(). Remember that this function is a higher-level interface each of the functions above, so we’ll reference them when we show each kind of plot, keeping the more verbose kind-specific API documentation at hand.

The default representation of the data in catplot() uses a scatterplot. There are actually two different categorical scatter plots in seaborn. They take different approaches to resolving the main challenge in representing categorical data with a scatter plot, which is that all of the points belonging to one category would fall on the same position along the axis corresponding to the categorical variable. The approach used by stripplot(), which is the default “kind” in catplot() is to adjust the positions of points on the categorical axis with a small amount of random “jitter”:

The jitter parameter controls the magnitude of jitter or disables it altogether:

The second approach adjusts the points along the categorical axis using an algorithm that prevents them from overlapping. It can give a better representation of the distribution of observations, although it only works well for relatively small datasets. This kind of plot is sometimes called a “beeswarm” and is drawn in seaborn by swarmplot(), which is activated by setting kind="swarm" in catplot():

Similar to the relational plots, it’s possible to add another dimension to a categorical plot by using a hue semantic. (The categorical plots do not currently support size or style semantics). Each different categorical plotting function handles the hue semantic differently. For the scatter plots, it is only necessary to change the color of the points:

Unlike with numerical data, it is not always obvious how to order the levels of the categorical variable along its axis. In general, the seaborn categorical plotting functions try to infer the order of categories from the data. If your data have a pandas Categorical datatype, then the default order of the categories can be set there. If the variable passed to the categorical axis looks numerical, the levels will be sorted. But, by default, the data are still treated as categorical and drawn at ordinal positions on the categorical axes (specifically, at 0, 1, …) even when numbers are used to label them:

As of v0.13.0, all categorical plotting functions have a native_scale parameter, which can be set to True when you want to use numeric or datetime data for categorical grouping without changing the underlying data properties:

The other option for choosing a default ordering is to take the levels of the category as they appear in the dataset. The ordering can also be controlled on a plot-specific basis using the order parameter. This can be important when drawing multiple categorical plots in the same figure, which we’ll see more of below:

We’ve referred to the idea of “categorical axis”. In these examples, that’s always corresponded to the horizontal axis. But it’s often helpful to put the categorical variable on the vertical axis (particularly when the category names are relatively long or there are many categories). To do this, swap the assignment of variables to axes:

As the size of the dataset grows, categorical scatter plots become limited in the information they can provide about the distribution of values within each category. When this happens, there are several approaches for summarizing the distributional information in ways that facilitate easy comparisons across the category levels.

The first is the familiar boxplot(). This kind of plot shows the three quartile values of the distribution along with extreme values. The “whiskers” extend to points that lie within 1.5 IQRs of the lower and upper quartile, and then observations that fall outside this range are displayed independently. This means that each value in the boxplot corresponds to an actual observation in the data.

When adding a hue semantic, the box for each level of the semantic variable is made narrower and shifted along the categorical axis:

This behavior is called “dodging”, and it is controlled by the dodge parameter. By default (as of v0.13.0), elements dodge only if they would otherwise overlap:

A related function, boxenplot(), draws a plot that is similar to a box plot but optimized for showing more information about the shape of the distribution. It is best suited for larger datasets:

A different approach is a violinplot(), which combines a boxplot with the kernel density estimation procedure described in the distributions tutorial:

This approach uses the kernel density estimate to provide a richer description of the distribution of values. Additionally, the quartile and whisker values from the boxplot are shown inside the violin. The downside is that, because the violinplot uses a KDE, there are some other parameters that may need tweaking, adding some complexity relative to the straightforward boxplot:

It’s also possible to “split” the violins, which can allow for a more efficient use of space:

Finally, there are several options for the plot that is drawn on the interior of the violins, including ways to show each individual observation instead of the summary boxplot values:

It can also be useful to combine swarmplot() or stripplot() with a box plot or violin plot to show each observation along with a summary of the distribution:

For other applications, rather than showing the distribution within each category, you might want to show an estimate of the central tendency of the values. Seaborn has two main ways to show this information. Importantly, the basic API for these functions is identical to that for the ones discussed above.

A familiar style of plot that accomplishes this goal is a bar plot. In seaborn, the barplot() function operates on a full dataset and applies a function to obtain the estimate (taking the mean by default). When there are multiple observations in each category, it also uses bootstrapping to compute a confidence interval around the estimate, which is plotted using error bars:

The default error bars show 95% confidence intervals, but (starting in v0.12), it is possible to select from a number of other representations:

A special case for the bar plot is when you want to show the number of observations in each category rather than computing a statistic for a second variable. This is similar to a histogram over a categorical, rather than quantitative, variable. In seaborn, it’s easy to do so with the countplot() function:

Both barplot() and countplot() can be invoked with all of the options discussed above, along with others that are demonstrated in the detailed documentation for each function:

An alternative style for visualizing the same information is offered by the pointplot() function. This function also encodes the value of the estimate with height on the other axis, but rather than showing a full bar, it plots the point estimate and confidence interval. Additionally, pointplot() connects points from the same hue category. This makes it easy to see how the main relationship is changing as a function of the hue semantic, because your eyes are quite good at picking up on differences of slopes:

While the categorical functions lack the style semantic of the relational functions, it can still be a good idea to vary the marker and/or linestyle along with the hue to make figures that are maximally accessible and reproduce well in black and white:

Just like relplot(), the fact that catplot() is built on a FacetGrid means that it is easy to add faceting variables to visualize higher-dimensional relationships:

For further customization of the plot, you can use the methods on the FacetGrid object that it returns:

**Examples:**

Example 1 (unknown):
```unknown
tips = sns.load_dataset("tips")
sns.catplot(data=tips, x="day", y="total_bill")
```

Example 2 (unknown):
```unknown
sns.catplot(data=tips, x="day", y="total_bill", jitter=False)
```

Example 3 (unknown):
```unknown
sns.catplot(data=tips, x="day", y="total_bill", kind="swarm")
```

Example 4 (unknown):
```unknown
sns.catplot(data=tips, x="day", y="total_bill", hue="sex", kind="swarm")
```

---

## Controlling figure aesthetics#

**URL:** https://seaborn.pydata.org/tutorial/aesthetics.html

**Contents:**
- Controlling figure aesthetics#
- Seaborn figure styles#
- Removing axes spines#
- Temporarily setting figure style#
- Overriding elements of the seaborn styles#
- Scaling plot elements#

Drawing attractive figures is important. When making figures for yourself, as you explore a dataset, it’s nice to have plots that are pleasant to look at. Visualizations are also central to communicating quantitative insights to an audience, and in that setting it’s even more necessary to have figures that catch the attention and draw a viewer in.

Matplotlib is highly customizable, but it can be hard to know what settings to tweak to achieve an attractive plot. Seaborn comes with a number of customized themes and a high-level interface for controlling the look of matplotlib figures.

Let’s define a simple function to plot some offset sine waves, which will help us see the different stylistic parameters we can tweak.

This is what the plot looks like with matplotlib defaults:

To switch to seaborn defaults, simply call the set_theme() function.

(Note that in versions of seaborn prior to 0.8, set_theme() was called on import. On later versions, it must be explicitly invoked).

Seaborn splits matplotlib parameters into two independent groups. The first group sets the aesthetic style of the plot, and the second scales various elements of the figure so that it can be easily incorporated into different contexts.

The interface for manipulating these parameters are two pairs of functions. To control the style, use the axes_style() and set_style() functions. To scale the plot, use the plotting_context() and set_context() functions. In both cases, the first function returns a dictionary of parameters and the second sets the matplotlib defaults.

There are five preset seaborn themes: darkgrid, whitegrid, dark, white, and ticks. They are each suited to different applications and personal preferences. The default theme is darkgrid. As mentioned above, the grid helps the plot serve as a lookup table for quantitative information, and the white-on grey helps to keep the grid from competing with lines that represent data. The whitegrid theme is similar, but it is better suited to plots with heavy data elements:

For many plots, (especially for settings like talks, where you primarily want to use figures to provide impressions of patterns in the data), the grid is less necessary.

Sometimes you might want to give a little extra structure to the plots, which is where ticks come in handy:

Both the white and ticks styles can benefit from removing the top and right axes spines, which are not needed. The seaborn function despine() can be called to remove them:

Some plots benefit from offsetting the spines away from the data, which can also be done when calling despine(). When the ticks don’t cover the whole range of the axis, the trim parameter will limit the range of the surviving spines.

You can also control which spines are removed with additional arguments to despine():

Although it’s easy to switch back and forth, you can also use the axes_style() function in a with statement to temporarily set plot parameters. This also allows you to make figures with differently-styled axes:

If you want to customize the seaborn styles, you can pass a dictionary of parameters to the rc argument of axes_style() and set_style(). Note that you can only override the parameters that are part of the style definition through this method. (However, the higher-level set_theme() function takes a dictionary of any matplotlib parameters).

If you want to see what parameters are included, you can just call the function with no arguments, which will return the current settings:

You can then set different versions of these parameters:

A separate set of parameters control the scale of plot elements, which should let you use the same code to make plots that are suited for use in settings where larger or smaller plots are appropriate.

First let’s reset the default parameters by calling set_theme():

The four preset contexts, in order of relative size, are paper, notebook, talk, and poster. The notebook style is the default, and was used in the plots above.

Most of what you now know about the style functions should transfer to the context functions.

You can call set_context() with one of these names to set the parameters, and you can override the parameters by providing a dictionary of parameter values.

You can also independently scale the size of the font elements when changing the context. (This option is also available through the top-level set() function).

Similarly, you can temporarily control the scale of figures nested under a with statement.

Both the style and the context can be quickly configured with the set() function. This function also sets the default color palette, but that will be covered in more detail in the next section of the tutorial.

**Examples:**

Example 1 (typescript):
```typescript
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

Example 2 (python):
```python
def sinplot(n=10, flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, n + 1):
        plt.plot(x, np.sin(x + i * .5) * (n + 2 - i) * flip)
```

Example 3 (unknown):
```unknown
sns.set_theme()
sinplot()
```

Example 4 (unknown):
```unknown
sns.set_style("whitegrid")
data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
sns.boxplot(data=data);
```

---

## User guide and tutorial#

**URL:** https://seaborn.pydata.org/tutorial.html

**Contents:**
- User guide and tutorial#
- API Overview#
- Objects interface#
- Plotting functions#
- Statistical operations#
- Multi-plot grids#
- Figure aesthetics#

---

## An introduction to seaborn#

**URL:** https://seaborn.pydata.org/tutorial/introduction.html

**Contents:**
- An introduction to seaborn#
- A high-level API for statistical graphics#
  - Statistical estimation#
  - Distributional representations#
  - Plots for categorical data#
- Multivariate views on complex datasets#
  - Lower-level tools for building figures#
- Opinionated defaults and flexible customization#
  - Relationship to matplotlib#
  - Next steps#

Seaborn is a library for making statistical graphics in Python. It builds on top of matplotlib and integrates closely with pandas data structures.

Seaborn helps you explore and understand your data. Its plotting functions operate on dataframes and arrays containing whole datasets and internally perform the necessary semantic mapping and statistical aggregation to produce informative plots. Its dataset-oriented, declarative API lets you focus on what the different elements of your plots mean, rather than on the details of how to draw them.

Here’s an example of what seaborn can do:

A few things have happened here. Let’s go through them one by one:

Seaborn is the only library we need to import for this simple example. By convention, it is imported with the shorthand sns.

Behind the scenes, seaborn uses matplotlib to draw its plots. For interactive work, it’s recommended to use a Jupyter/IPython interface in matplotlib mode, or else you’ll have to call matplotlib.pyplot.show() when you want to see the plot.

This uses the matplotlib rcParam system and will affect how all matplotlib plots look, even if you don’t make them with seaborn. Beyond the default theme, there are several other options, and you can independently control the style and scaling of the plot to quickly translate your work between presentation contexts (e.g., making a version of your figure that will have readable fonts when projected during a talk). If you like the matplotlib defaults or prefer a different theme, you can skip this step and still use the seaborn plotting functions.

Most code in the docs will use the load_dataset() function to get quick access to an example dataset. There’s nothing special about these datasets: they are just pandas dataframes, and we could have loaded them with pandas.read_csv() or built them by hand. Most of the examples in the documentation will specify data using pandas dataframes, but seaborn is very flexible about the data structures that it accepts.

This plot shows the relationship between five variables in the tips dataset using a single call to the seaborn function relplot(). Notice how we provided only the names of the variables and their roles in the plot. Unlike when using matplotlib directly, it wasn’t necessary to specify attributes of the plot elements in terms of the color values or marker codes. Behind the scenes, seaborn handled the translation from values in the dataframe to arguments that matplotlib understands. This declarative approach lets you stay focused on the questions that you want to answer, rather than on the details of how to control matplotlib.

There is no universally best way to visualize data. Different questions are best answered by different plots. Seaborn makes it easy to switch between different visual representations by using a consistent dataset-oriented API.

The function relplot() is named that way because it is designed to visualize many different statistical relationships. While scatter plots are often effective, relationships where one variable represents a measure of time are better represented by a line. The relplot() function has a convenient kind parameter that lets you easily switch to this alternate representation:

Notice how the size and style parameters are used in both the scatter and line plots, but they affect the two visualizations differently: changing the marker area and symbol in the scatter plot vs the line width and dashing in the line plot. We did not need to keep those details in mind, letting us focus on the overall structure of the plot and the information we want it to convey.

Often, we are interested in the average value of one variable as a function of other variables. Many seaborn functions will automatically perform the statistical estimation that is necessary to answer these questions:

When statistical values are estimated, seaborn will use bootstrapping to compute confidence intervals and draw error bars representing the uncertainty of the estimate.

Statistical estimation in seaborn goes beyond descriptive statistics. For example, it is possible to enhance a scatterplot by including a linear regression model (and its uncertainty) using lmplot():

Statistical analyses require knowledge about the distribution of variables in your dataset. The seaborn function displot() supports several approaches to visualizing distributions. These include classic techniques like histograms and computationally-intensive approaches like kernel density estimation:

Seaborn also tries to promote techniques that are powerful but less familiar, such as calculating and plotting the empirical cumulative distribution function of the data:

Several specialized plot types in seaborn are oriented towards visualizing categorical data. They can be accessed through catplot(). These plots offer different levels of granularity. At the finest level, you may wish to see every observation by drawing a “swarm” plot: a scatter plot that adjusts the positions of the points along the categorical axis so that they don’t overlap:

Alternately, you could use kernel density estimation to represent the underlying distribution that the points are sampled from:

Or you could show only the mean value and its confidence interval within each nested category:

Some seaborn functions combine multiple kinds of plots to quickly give informative summaries of a dataset. One, jointplot(), focuses on a single relationship. It plots the joint distribution between two variables along with each variable’s marginal distribution:

The other, pairplot(), takes a broader view: it shows joint and marginal distributions for all pairwise relationships and for each variable, respectively:

These tools work by combining axes-level plotting functions with objects that manage the layout of the figure, linking the structure of a dataset to a grid of axes. Both elements are part of the public API, and you can use them directly to create complex figures with only a few more lines of code:

Seaborn creates complete graphics with a single function call: when possible, its functions will automatically add informative axis labels and legends that explain the semantic mappings in the plot.

In many cases, seaborn will also choose default values for its parameters based on characteristics of the data. For example, the color mappings that we have seen so far used distinct hues (blue, orange, and sometimes green) to represent different levels of the categorical variables assigned to hue. When mapping a numeric variable, some functions will switch to a continuous gradient:

When you’re ready to share or publish your work, you’ll probably want to polish the figure beyond what the defaults achieve. Seaborn allows for several levels of customization. It defines multiple built-in themes that apply to all figures, its functions have standardized parameters that can modify the semantic mappings for each plot, and additional keyword arguments are passed down to the underlying matplotlib artists, allowing even more control. Once you’ve created a plot, its properties can be modified through both the seaborn API and by dropping down to the matplotlib layer for fine-grained tweaking:

Seaborn’s integration with matplotlib allows you to use it across the many environments that matplotlib supports, including exploratory analysis in notebooks, real-time interaction in GUI applications, and archival output in a number of raster and vector formats.

While you can be productive using only seaborn functions, full customization of your graphics will require some knowledge of matplotlib’s concepts and API. One aspect of the learning curve for new users of seaborn will be knowing when dropping down to the matplotlib layer is necessary to achieve a particular customization. On the other hand, users coming from matplotlib will find that much of their knowledge transfers.

Matplotlib has a comprehensive and powerful API; just about any attribute of the figure can be changed to your liking. A combination of seaborn’s high-level interface and matplotlib’s deep customizability will allow you both to quickly explore your data and to create graphics that can be tailored into a publication quality final product.

You have a few options for where to go next. You might first want to learn how to install seaborn. Once that’s done, you can browse the example gallery to get a broader sense for what kind of graphics seaborn can produce. Or you can read through the rest of the user guide and tutorial for a deeper discussion of the different tools and what they are designed to accomplish. If you have a specific plot in mind and want to know how to make it, you could check out the API reference, which documents each function’s parameters and shows many examples to illustrate usage.

**Examples:**

Example 1 (markdown):
```markdown
# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

# Load an example dataset
tips = sns.load_dataset("tips")

# Create a visualization
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)
```

Example 2 (markdown):
```markdown
# Import seaborn
import seaborn as sns
```

Example 3 (markdown):
```markdown
# Apply the default theme
sns.set_theme()
```

Example 4 (markdown):
```markdown
# Load an example dataset
tips = sns.load_dataset("tips")
```

---

## Data structures accepted by seaborn#

**URL:** https://seaborn.pydata.org/tutorial/data_structure.html

**Contents:**
- Data structures accepted by seaborn#
- Long-form vs. wide-form data#
  - Long-form data#
  - Wide-form data#
  - Messy data#
  - Further reading and take-home points#
- Options for visualizing long-form data#
- Options for visualizing wide-form data#

As a data visualization library, seaborn requires that you provide it with data. This chapter explains the various ways to accomplish that task. Seaborn supports several different dataset formats, and most functions accept data represented with objects from the pandas or numpy libraries as well as built-in Python types like lists and dictionaries. Understanding the usage patterns associated with these different options will help you quickly create useful visualizations for nearly any dataset.

As of current writing (v0.13.0), the full breadth of options covered here are supported by most, but not all, of the functions in seaborn. Namely, a few older functions (e.g., lmplot() and regplot()) are more limited in what they accept.

Most plotting functions in seaborn are oriented towards vectors of data. When plotting x against y, each variable should be a vector. Seaborn accepts data sets that have more than one vector organized in some tabular fashion. There is a fundamental distinction between “long-form” and “wide-form” data tables, and seaborn will treat each differently.

A long-form data table has the following characteristics:

Each variable is a column

Each observation is a row

As a simple example, consider the “flights” dataset, which records the number of airline passengers who flew in each month from 1949 to 1960. This dataset has three variables (year, month, and number of passengers):

With long-form data, columns in the table are given roles in the plot by explicitly assigning them to one of the variables. For example, making a monthly plot of the number of passengers per year looks like this:

The advantage of long-form data is that it lends itself well to this explicit specification of the plot. It can accommodate datasets of arbitrary complexity, so long as the variables and observations can be clearly defined. But this format takes some getting used to, because it is often not the model of the data that one has in their head.

For simple datasets, it is often more intuitive to think about data the way it might be viewed in a spreadsheet, where the columns and rows contain levels of different variables. For example, we can convert the flights dataset into a wide-form organization by “pivoting” it so that each column has each month’s time series over years:

Here we have the same three variables, but they are organized differently. The variables in this dataset are linked to the dimensions of the table, rather than to named fields. Each observation is defined by both the value at a cell in the table and the coordinates of that cell with respect to the row and column indices.

With long-form data, we can access variables in the dataset by their name. That is not the case with wide-form data. Nevertheless, because there is a clear association between the dimensions of the table and the variable in the dataset, seaborn is able to assign those variables roles in the plot.

Seaborn treats the argument to data as wide form when neither x nor y are assigned.

This plot looks very similar to the one before. Seaborn has assigned the index of the dataframe to x, the values of the dataframe to y, and it has drawn a separate line for each month. There is a notable difference between the two plots, however. When the dataset went through the “pivot” operation that converted it from long-form to wide-form, the information about what the values mean was lost. As a result, there is no y axis label. (The lines also have dashes here, because relplot() has mapped the column variable to both the hue and style semantic so that the plot is more accessible. We didn’t do that in the long-form case, but we could have by setting style="month").

Thus far, we did much less typing while using wide-form data and made nearly the same plot. This seems easier! But a big advantage of long-form data is that, once you have the data in the correct format, you no longer need to think about its structure. You can design your plots by thinking only about the variables contained within it. For example, to draw lines that represent the monthly time series for each year, simply reassign the variables:

To achieve the same remapping with the wide-form dataset, we would need to transpose the table:

(This example also illustrates another wrinkle, which is that seaborn currently considers the column variable in a wide-form dataset to be categorical regardless of its datatype, whereas, because the long-form variable is numeric, it is assigned a quantitative color palette and legend. This may change in the future).

The absence of explicit variable assignments also means that each plot type needs to define a fixed mapping between the dimensions of the wide-form data and the roles in the plot. Because this natural mapping may vary across plot types, the results are less predictable when using wide-form data. For example, the categorical plots assign the column dimension of the table to x and then aggregate across the rows (ignoring the index):

When using pandas to represent wide-form data, you are limited to just a few variables (no more than three). This is because seaborn does not make use of multi-index information, which is how pandas represents additional variables in a tabular format. The xarray project offers labeled N-dimensional array objects, which can be considered a generalization of wide-form data to higher dimensions. At present, seaborn does not directly support objects from xarray, but they can be transformed into a long-form pandas.DataFrame using the to_pandas method and then plotted in seaborn like any other long-form data set.

In summary, we can think of long-form and wide-form datasets as looking something like this:

Many datasets cannot be clearly interpreted using either long-form or wide-form rules. If datasets that are clearly long-form or wide-form are “tidy”, we might say that these more ambiguous datasets are “messy”. In a messy dataset, the variables are neither uniquely defined by the keys nor by the dimensions of the table. This often occurs with repeated-measures data, where it is natural to organize a table such that each row corresponds to the unit of data collection. Consider this simple dataset from a psychology experiment in which twenty subjects performed a memory task where they studied anagrams while their attention was either divided or focused:

The attention variable is between-subjects, but there is also a within-subjects variable: the number of possible solutions to the anagrams, which varied from 1 to 3. The dependent measure is a score of memory performance. These two variables (number and score) are jointly encoded across several columns. As a result, the whole dataset is neither clearly long-form nor clearly wide-form.

How might we tell seaborn to plot the average score as a function of attention and number of solutions? We’d first need to coerce the data into one of our two structures. Let’s transform it to a tidy long-form table, such that each variable is a column and each row is an observation. We can use the method pandas.DataFrame.melt() to accomplish this task:

Now we can make the plot that we want:

For a longer discussion about tabular data structures, you could read the “Tidy Data” paper by Hadley Whickham. Note that seaborn uses a slightly different set of concepts than are defined in the paper. While the paper associates tidyness with long-form structure, we have drawn a distinction between “tidy wide-form” data, where there is a clear mapping between variables in the dataset and the dimensions of the table, and “messy data”, where no such mapping exists.

The long-form structure has clear advantages. It allows you to create figures by explicitly assigning variables in the dataset to roles in plot, and you can do so with more than three variables. When possible, try to represent your data with a long-form structure when embarking on serious analysis. Most of the examples in the seaborn documentation will use long-form data. But in cases where it is more natural to keep the dataset wide, remember that seaborn can remain useful.

While long-form data has a precise definition, seaborn is fairly flexible in terms of how it is actually organized across the data structures in memory. The examples in the rest of the documentation will typically use pandas.DataFrame objects and reference variables in them by assigning names of their columns to the variables in the plot. But it is also possible to store vectors in a Python dictionary or a class that implements that interface:

Many pandas operations, such as the split-apply-combine operations of a group-by, will produce a dataframe where information has moved from the columns of the input dataframe to the index of the output. So long as the name is retained, you can still reference the data as normal:

Additionally, it’s possible to pass vectors of data directly as arguments to x, y, and other plotting variables. If these vectors are pandas objects, the name attribute will be used to label the plot:

Numpy arrays and other objects that implement the Python sequence interface work too, but if they don’t have names, the plot will not be as informative without further tweaking:

The options for passing wide-form data are even more flexible. As with long-form data, pandas objects are preferable because the name (and, in some cases, index) information can be used. But in essence, any format that can be viewed as a single vector or a collection of vectors can be passed to data, and a valid plot can usually be constructed.

The example we saw above used a rectangular pandas.DataFrame, which can be thought of as a collection of its columns. A dict or list of pandas objects will also work, but we’ll lose the axis labels:

The vectors in a collection do not need to have the same length. If they have an index, it will be used to align them:

Whereas an ordinal index will be used for numpy arrays or simple Python sequences:

But a dictionary of such vectors will at least use the keys:

Rectangular numpy arrays are treated just like a dataframe without index information, so they are viewed as a collection of column vectors. Note that this is different from how numpy indexing operations work, where a single indexer will access a row. But it is consistent with how pandas would turn the array into a dataframe or how matplotlib would plot it:

**Examples:**

Example 1 (unknown):
```unknown
flights = sns.load_dataset("flights")
flights.head()
```

Example 2 (unknown):
```unknown
sns.relplot(data=flights, x="year", y="passengers", hue="month", kind="line")
```

Example 3 (unknown):
```unknown
flights_wide = flights.pivot(index="year", columns="month", values="passengers")
flights_wide.head()
```

Example 4 (unknown):
```unknown
sns.relplot(data=flights_wide, kind="line")
```

---

## Visualizing statistical relationships#

**URL:** https://seaborn.pydata.org/tutorial/relational.html

**Contents:**
- Visualizing statistical relationships#
- Relating variables with scatter plots#
- Emphasizing continuity with line plots#
  - Aggregation and representing uncertainty#
  - Plotting subsets of data with semantic mappings#
  - Controlling sorting and orientation#
- Showing multiple relationships with facets#

Statistical analysis is a process of understanding how variables in a dataset relate to each other and how those relationships depend on other variables. Visualization can be a core component of this process because, when data are visualized properly, the human visual system can see trends and patterns that indicate a relationship.

We will discuss three seaborn functions in this tutorial. The one we will use most is relplot(). This is a figure-level function for visualizing statistical relationships using two common approaches: scatter plots and line plots. relplot() combines a FacetGrid with one of two axes-level functions:

scatterplot() (with kind="scatter"; the default)

lineplot() (with kind="line")

As we will see, these functions can be quite illuminating because they use simple and easily-understood representations of data that can nevertheless represent complex dataset structures. They can do so because they plot two-dimensional graphics that can be enhanced by mapping up to three additional variables using the semantics of hue, size, and style.

The scatter plot is a mainstay of statistical visualization. It depicts the joint distribution of two variables using a cloud of points, where each point represents an observation in the dataset. This depiction allows the eye to infer a substantial amount of information about whether there is any meaningful relationship between them.

There are several ways to draw a scatter plot in seaborn. The most basic, which should be used when both variables are numeric, is the scatterplot() function. In the categorical visualization tutorial, we will see specialized tools for using scatterplots to visualize categorical data. The scatterplot() is the default kind in relplot() (it can also be forced by setting kind="scatter"):

While the points are plotted in two dimensions, another dimension can be added to the plot by coloring the points according to a third variable. In seaborn, this is referred to as using a “hue semantic”, because the color of the point gains meaning:

To emphasize the difference between the classes, and to improve accessibility, you can use a different marker style for each class:

It’s also possible to represent four variables by changing the hue and style of each point independently. But this should be done carefully, because the eye is much less sensitive to shape than to color:

In the examples above, the hue semantic was categorical, so the default qualitative palette was applied. If the hue semantic is numeric (specifically, if it can be cast to float), the default coloring switches to a sequential palette:

In both cases, you can customize the color palette. There are many options for doing so. Here, we customize a sequential palette using the string interface to cubehelix_palette():

The third kind of semantic variable changes the size of each point:

Unlike with matplotlib.pyplot.scatter(), the literal value of the variable is not used to pick the area of the point. Instead, the range of values in data units is normalized into a range in area units. This range can be customized:

More examples for customizing how the different semantics are used to show statistical relationships are shown in the scatterplot() API examples.

Scatter plots are highly effective, but there is no universally optimal type of visualisation. Instead, the visual representation should be adapted for the specifics of the dataset and to the question you are trying to answer with the plot.

With some datasets, you may want to understand changes in one variable as a function of time, or a similarly continuous variable. In this situation, a good choice is to draw a line plot. In seaborn, this can be accomplished by the lineplot() function, either directly or with relplot() by setting kind="line":

More complex datasets will have multiple measurements for the same value of the x variable. The default behavior in seaborn is to aggregate the multiple measurements at each x value by plotting the mean and the 95% confidence interval around the mean:

The confidence intervals are computed using bootstrapping, which can be time-intensive for larger datasets. It’s therefore possible to disable them:

Another good option, especially with larger data, is to represent the spread of the distribution at each timepoint by plotting the standard deviation instead of a confidence interval:

To turn off aggregation altogether, set the estimator parameter to None This might produce a strange effect when the data have multiple observations at each point.

The lineplot() function has the same flexibility as scatterplot(): it can show up to three additional variables by modifying the hue, size, and style of the plot elements. It does so using the same API as scatterplot(), meaning that we don’t need to stop and think about the parameters that control the look of lines vs. points in matplotlib.

Using semantics in lineplot() will also determine how the data get aggregated. For example, adding a hue semantic with two levels splits the plot into two lines and error bands, coloring each to indicate which subset of the data they correspond to.

Adding a style semantic to a line plot changes the pattern of dashes in the line by default:

But you can identify subsets by the markers used at each observation, either together with the dashes or instead of them:

As with scatter plots, be cautious about making line plots using multiple semantics. While sometimes informative, they can also be difficult to parse and interpret. But even when you are only examining changes across one additional variable, it can be useful to alter both the color and style of the lines. This can make the plot more accessible when printed to black-and-white or viewed by someone with color blindness:

When you are working with repeated measures data (that is, you have units that were sampled multiple times), you can also plot each sampling unit separately without distinguishing them through semantics. This avoids cluttering the legend:

The default colormap and handling of the legend in lineplot() also depends on whether the hue semantic is categorical or numeric:

It may happen that, even though the hue variable is numeric, it is poorly represented by a linear color scale. That’s the case here, where the levels of the hue variable are logarithmically scaled. You can provide specific color values for each line by passing a list or dictionary:

Or you can alter how the colormap is normalized:

The third semantic, size, changes the width of the lines:

While the size variable will typically be numeric, it’s also possible to map a categorical variable with the width of the lines. Be cautious when doing so, because it will be difficult to distinguish much more than “thick” vs “thin” lines. However, dashes can be hard to perceive when lines have high-frequency variability, so using different widths may be more effective in that case:

Because lineplot() assumes that you are most often trying to draw y as a function of x, the default behavior is to sort the data by the x values before plotting. However, this can be disabled:

It’s also possible to sort (and aggregate) along the y axis:

We’ve emphasized in this tutorial that, while these functions can show several semantic variables at once, it’s not always effective to do so. But what about when you do want to understand how a relationship between two variables depends on more than one other variable?

The best approach may be to make more than one plot. Because relplot() is based on the FacetGrid, this is easy to do. To show the influence of an additional variable, instead of assigning it to one of the semantic roles in the plot, use it to “facet” the visualization. This means that you make multiple axes and plot subsets of the data on each of them:

You can also show the influence of two variables this way: one by faceting on the columns and one by faceting on the rows. As you start adding more variables to the grid, you may want to decrease the figure size. Remember that the size FacetGrid is parameterized by the height and aspect ratio of each facet:

When you want to examine effects across many levels of a variable, it can be a good idea to facet that variable on the columns and then “wrap” the facets into the rows:

These visualizations, which are sometimes called “lattice” plots or “small-multiples”, are very effective because they present the data in a format that makes it easy for the eye to detect both overall patterns and deviations from those patterns. While you should make use of the flexibility afforded by scatterplot() and relplot(), always try to keep in mind that several simple plots are usually more effective than one complex plot.

**Examples:**

Example 1 (typescript):
```typescript
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
```

Example 2 (unknown):
```unknown
tips = sns.load_dataset("tips")
sns.relplot(data=tips, x="total_bill", y="tip")
```

Example 3 (unknown):
```unknown
sns.relplot(data=tips, x="total_bill", y="tip", hue="smoker")
```

Example 4 (unknown):
```unknown
sns.relplot(
    data=tips,
    x="total_bill", y="tip", hue="smoker", style="smoker"
)
```

---

## Visualizing distributions of data#

**URL:** https://seaborn.pydata.org/tutorial/distributions.html

**Contents:**
- Visualizing distributions of data#
- Plotting univariate histograms#
  - Choosing the bin size#
  - Conditioning on other variables#
  - Normalized histogram statistics#
- Kernel density estimation#
  - Choosing the smoothing bandwidth#
  - Conditioning on other variables#
  - Kernel density estimation pitfalls#
- Empirical cumulative distributions#

An early step in any effort to analyze or model data should be to understand how the variables are distributed. Techniques for distribution visualization can provide quick answers to many important questions. What range do the observations cover? What is their central tendency? Are they heavily skewed in one direction? Is there evidence for bimodality? Are there significant outliers? Do the answers to these questions vary across subsets defined by other variables?

The distributions module contains several functions designed to answer questions such as these. The axes-level functions are histplot(), kdeplot(), ecdfplot(), and rugplot(). They are grouped together within the figure-level displot(), jointplot(), and pairplot() functions.

There are several different approaches to visualizing a distribution, and each has its relative advantages and drawbacks. It is important to understand these factors so that you can choose the best approach for your particular aim.

Perhaps the most common approach to visualizing a distribution is the histogram. This is the default approach in displot(), which uses the same underlying code as histplot(). A histogram is a bar plot where the axis representing the data variable is divided into a set of discrete bins and the count of observations falling within each bin is shown using the height of the corresponding bar:

This plot immediately affords a few insights about the flipper_length_mm variable. For instance, we can see that the most common flipper length is about 195 mm, but the distribution appears bimodal, so this one number does not represent the data well.

The size of the bins is an important parameter, and using the wrong bin size can mislead by obscuring important features of the data or by creating apparent features out of random variability. By default, displot()/histplot() choose a default bin size based on the variance of the data and the number of observations. But you should not be over-reliant on such automatic approaches, because they depend on particular assumptions about the structure of your data. It is always advisable to check that your impressions of the distribution are consistent across different bin sizes. To choose the size directly, set the binwidth parameter:

In other circumstances, it may make more sense to specify the number of bins, rather than their size:

One example of a situation where defaults fail is when the variable takes a relatively small number of integer values. In that case, the default bin width may be too small, creating awkward gaps in the distribution:

One approach would be to specify the precise bin breaks by passing an array to bins:

This can also be accomplished by setting discrete=True, which chooses bin breaks that represent the unique values in a dataset with bars that are centered on their corresponding value.

It’s also possible to visualize the distribution of a categorical variable using the logic of a histogram. Discrete bins are automatically set for categorical variables, but it may also be helpful to “shrink” the bars slightly to emphasize the categorical nature of the axis:

Once you understand the distribution of a variable, the next step is often to ask whether features of that distribution differ across other variables in the dataset. For example, what accounts for the bimodal distribution of flipper lengths that we saw above? displot() and histplot() provide support for conditional subsetting via the hue semantic. Assigning a variable to hue will draw a separate histogram for each of its unique values and distinguish them by color:

By default, the different histograms are “layered” on top of each other and, in some cases, they may be difficult to distinguish. One option is to change the visual representation of the histogram from a bar plot to a “step” plot:

Alternatively, instead of layering each bar, they can be “stacked”, or moved vertically. In this plot, the outline of the full histogram will match the plot with only a single variable:

The stacked histogram emphasizes the part-whole relationship between the variables, but it can obscure other features (for example, it is difficult to determine the mode of the Adelie distribution. Another option is “dodge” the bars, which moves them horizontally and reduces their width. This ensures that there are no overlaps and that the bars remain comparable in terms of height. But it only works well when the categorical variable has a small number of levels:

Because displot() is a figure-level function and is drawn onto a FacetGrid, it is also possible to draw each individual distribution in a separate subplot by assigning the second variable to col or row rather than (or in addition to) hue. This represents the distribution of each subset well, but it makes it more difficult to draw direct comparisons:

None of these approaches are perfect, and we will soon see some alternatives to a histogram that are better-suited to the task of comparison.

Before we do, another point to note is that, when the subsets have unequal numbers of observations, comparing their distributions in terms of counts may not be ideal. One solution is to normalize the counts using the stat parameter:

By default, however, the normalization is applied to the entire distribution, so this simply rescales the height of the bars. By setting common_norm=False, each subset will be normalized independently:

Density normalization scales the bars so that their areas sum to 1. As a result, the density axis is not directly interpretable. Another option is to normalize the bars to that their heights sum to 1. This makes most sense when the variable is discrete, but it is an option for all histograms:

A histogram aims to approximate the underlying probability density function that generated the data by binning and counting observations. Kernel density estimation (KDE) presents a different solution to the same problem. Rather than using discrete bins, a KDE plot smooths the observations with a Gaussian kernel, producing a continuous density estimate:

Much like with the bin size in the histogram, the ability of the KDE to accurately represent the data depends on the choice of smoothing bandwidth. An over-smoothed estimate might erase meaningful features, but an under-smoothed estimate can obscure the true shape within random noise. The easiest way to check the robustness of the estimate is to adjust the default bandwidth:

Note how the narrow bandwidth makes the bimodality much more apparent, but the curve is much less smooth. In contrast, a larger bandwidth obscures the bimodality almost completely:

As with histograms, if you assign a hue variable, a separate density estimate will be computed for each level of that variable:

In many cases, the layered KDE is easier to interpret than the layered histogram, so it is often a good choice for the task of comparison. Many of the same options for resolving multiple distributions apply to the KDE as well, however:

Note how the stacked plot filled in the area between each curve by default. It is also possible to fill in the curves for single or layered densities, although the default alpha value (opacity) will be different, so that the individual densities are easier to resolve.

KDE plots have many advantages. Important features of the data are easy to discern (central tendency, bimodality, skew), and they afford easy comparisons between subsets. But there are also situations where KDE poorly represents the underlying data. This is because the logic of KDE assumes that the underlying distribution is smooth and unbounded. One way this assumption can fail is when a variable reflects a quantity that is naturally bounded. If there are observations lying close to the bound (for example, small values of a variable that cannot be negative), the KDE curve may extend to unrealistic values:

This can be partially avoided with the cut parameter, which specifies how far the curve should extend beyond the extreme datapoints. But this influences only where the curve is drawn; the density estimate will still smooth over the range where no data can exist, causing it to be artificially low at the extremes of the distribution:

The KDE approach also fails for discrete data or when data are naturally continuous but specific values are over-represented. The important thing to keep in mind is that the KDE will always show you a smooth curve, even when the data themselves are not smooth. For example, consider this distribution of diamond weights:

While the KDE suggests that there are peaks around specific values, the histogram reveals a much more jagged distribution:

As a compromise, it is possible to combine these two approaches. While in histogram mode, displot() (as with histplot()) has the option of including the smoothed KDE curve (note kde=True, not kind="kde"):

A third option for visualizing distributions computes the “empirical cumulative distribution function” (ECDF). This plot draws a monotonically-increasing curve through each datapoint such that the height of the curve reflects the proportion of observations with a smaller value:

The ECDF plot has two key advantages. Unlike the histogram or KDE, it directly represents each datapoint. That means there is no bin size or smoothing parameter to consider. Additionally, because the curve is monotonically increasing, it is well-suited for comparing multiple distributions:

The major downside to the ECDF plot is that it represents the shape of the distribution less intuitively than a histogram or density curve. Consider how the bimodality of flipper lengths is immediately apparent in the histogram, but to see it in the ECDF plot, you must look for varying slopes. Nevertheless, with practice, you can learn to answer all of the important questions about a distribution by examining the ECDF, and doing so can be a powerful approach.

All of the examples so far have considered univariate distributions: distributions of a single variable, perhaps conditional on a second variable assigned to hue. Assigning a second variable to y, however, will plot a bivariate distribution:

A bivariate histogram bins the data within rectangles that tile the plot and then shows the count of observations within each rectangle with the fill color (analogous to a heatmap()). Similarly, a bivariate KDE plot smoothes the (x, y) observations with a 2D Gaussian. The default representation then shows the contours of the 2D density:

Assigning a hue variable will plot multiple heatmaps or contour sets using different colors. For bivariate histograms, this will only work well if there is minimal overlap between the conditional distributions:

The contour approach of the bivariate KDE plot lends itself better to evaluating overlap, although a plot with too many contours can get busy:

Just as with univariate plots, the choice of bin size or smoothing bandwidth will determine how well the plot represents the underlying bivariate distribution. The same parameters apply, but they can be tuned for each variable by passing a pair of values:

To aid interpretation of the heatmap, add a colorbar to show the mapping between counts and color intensity:

The meaning of the bivariate density contours is less straightforward. Because the density is not directly interpretable, the contours are drawn at iso-proportions of the density, meaning that each curve shows a level set such that some proportion p of the density lies below it. The p values are evenly spaced, with the lowest level contolled by the thresh parameter and the number controlled by levels:

The levels parameter also accepts a list of values, for more control:

The bivariate histogram allows one or both variables to be discrete. Plotting one discrete and one continuous variable offers another way to compare conditional univariate distributions:

In contrast, plotting two discrete variables is an easy to way show the cross-tabulation of the observations:

Several other figure-level plotting functions in seaborn make use of the histplot() and kdeplot() functions.

The first is jointplot(), which augments a bivariate relational or distribution plot with the marginal distributions of the two variables. By default, jointplot() represents the bivariate distribution using scatterplot() and the marginal distributions using histplot():

Similar to displot(), setting a different kind="kde" in jointplot() will change both the joint and marginal plots the use kdeplot():

jointplot() is a convenient interface to the JointGrid class, which offeres more flexibility when used directly:

A less-obtrusive way to show marginal distributions uses a “rug” plot, which adds a small tick on the edge of the plot to represent each individual observation. This is built into displot():

And the axes-level rugplot() function can be used to add rugs on the side of any other kind of plot:

The pairplot() function offers a similar blend of joint and marginal distributions. Rather than focusing on a single relationship, however, pairplot() uses a “small-multiple” approach to visualize the univariate distribution of all variables in a dataset along with all of their pairwise relationships:

As with jointplot()/JointGrid, using the underlying PairGrid directly will afford more flexibility with only a bit more typing:

**Examples:**

Example 1 (unknown):
```unknown
penguins = sns.load_dataset("penguins")
sns.displot(penguins, x="flipper_length_mm")
```

Example 2 (unknown):
```unknown
sns.displot(penguins, x="flipper_length_mm", binwidth=3)
```

Example 3 (unknown):
```unknown
sns.displot(penguins, x="flipper_length_mm", bins=20)
```

Example 4 (unknown):
```unknown
tips = sns.load_dataset("tips")
sns.displot(tips, x="size")
```

---

## Choosing color palettes#

**URL:** https://seaborn.pydata.org/tutorial/color_palettes.html

**Contents:**
- Choosing color palettes#
- General principles for using color in plots#
  - Components of color#
  - Vary hue to distinguish categories#
  - Vary luminance to represent numbers#
- Tools for choosing color palettes#
- Qualitative color palettes#
  - Using circular color systems#
  - Using categorical Color Brewer palettes#
- Sequential color palettes#

Seaborn makes it easy to use colors that are well-suited to the characteristics of your data and your visualization goals. This chapter discusses both the general principles that should guide your choices and the tools in seaborn that help you quickly find the best solution for a given application.

Because of the way our eyes work, a particular color can be defined using three components. We usually program colors in a computer by specifying their RGB values, which set the intensity of the red, green, and blue channels in a display. But for analyzing the perceptual attributes of a color, it’s better to think in terms of hue, saturation, and luminance channels.

Hue is the component that distinguishes “different colors” in a non-technical sense. It’s property of color that leads to first-order names like “red” and “blue”:

Saturation (or chroma) is the colorfulness. Two colors with different hues will look more distinct when they have more saturation:

And lightness corresponds to how much light is emitted (or reflected, for printed colors), ranging from black to white:

When you want to represent multiple categories in a plot, you typically should vary the color of the elements. Consider this simple example: in which of these two plots is it easier to count the number of triangular points?

In the plot on the right, the orange triangles “pop out”, making it easy to distinguish them from the circles. This pop-out effect happens because our visual system prioritizes color differences.

The blue and orange colors differ mostly in terms of their hue. Hue is useful for representing categories: most people can distinguish a moderate number of hues relatively easily, and points that have different hues but similar brightness or intensity seem equally important. It also makes plots easier to talk about. Consider this example:

Most people would be able to quickly ascertain that there are five distinct categories in the plot on the left and, if asked to characterize the “blue” points, would be able to do so.

With the plot on the right, where the points are all blue but vary in their luminance and saturation, it’s harder to say how many unique categories are present. And how would we talk about a particular category? “The fairly-but-not-too-blue points?” What’s more, the gray dots seem to fade into the background, de-emphasizing them relative to the more intense blue dots. If the categories are equally important, this is a poor representation.

So as a general rule, use hue variation to represent categories. With that said, here are few notes of caution. If you have more than a handful of colors in your plot, it can become difficult to keep in mind what each one means, unless there are pre-existing associations between the categories and the colors used to represent them. This makes your plot harder to interpret: rather than focusing on the data, a viewer will have to continually refer to the legend to make sense of what is shown. So you should strive not to make plots that are too complex. And be mindful that not everyone sees colors the same way. Varying both shape (or some other attribute) and color can help people with anomalous color vision understand your plots, and it can keep them (somewhat) interpretable if they are printed to black-and-white.

On the other hand, hue variations are not well suited to representing numeric data. Consider this example, where we need colors to represent the counts in a bivariate histogram. On the left, we use a circular colormap, where gradual changes in the number of observation within each bin correspond to gradual changes in hue. On the right, we use a palette that uses brighter colors to represent bins with larger counts:

With the hue-based palette, it’s quite difficult to ascertain the shape of the bivariate distribution. In contrast, the luminance palette makes it much more clear that there are two prominent peaks.

Varying luminance helps you see structure in data, and changes in luminance are more intuitively processed as changes in importance. But the plot on the right does not use a grayscale colormap. Its colorfulness makes it more interesting, and the subtle hue variation increases the perceptual distance between two values. As a result, small differences slightly easier to resolve.

These examples show that color palette choices are about more than aesthetics: the colors you choose can reveal patterns in your data if used effectively or hide them if used poorly. There is not one optimal palette, but there are palettes that are better or worse for particular datasets and visualization approaches.

And aesthetics do matter: the more that people want to look at your figures, the greater the chance that they will learn something from them. This is true even when you are making plots for yourself. During exploratory data analysis, you may generate many similar figures. Varying the color palettes will add a sense of novelty, which keeps you engaged and prepared to notice interesting features of your data.

So how can you choose color palettes that both represent your data well and look attractive?

The most important function for working with color palettes is, aptly, color_palette(). This function provides an interface to most of the possible ways that one can generate color palettes in seaborn. And it’s used internally by any function that has a palette argument.

The primary argument to color_palette() is usually a string: either the name of a specific palette or the name of a family and additional arguments to select a specific member. In the latter case, color_palette() will delegate to more specific function, such as cubehelix_palette(). It’s also possible to pass a list of colors specified any way that matplotlib accepts (an RGB tuple, a hex code, or a name in the X11 table). The return value is an object that wraps a list of RGB tuples with a few useful methods, such as conversion to hex codes and a rich HTML representation.

Calling color_palette() with no arguments will return the current default color palette that matplotlib (and most seaborn functions) will use if colors are not otherwise specified. This default palette can be set with the corresponding set_palette() function, which calls color_palette() internally and accepts the same arguments.

To motivate the different options that color_palette() provides, it will be useful to introduce a classification scheme for color palettes. Broadly, palettes fall into one of three categories:

qualitative palettes, good for representing categorical data

sequential palettes, good for representing numeric data

diverging palettes, good for representing numeric data with a categorical boundary

Qualitative palettes are well-suited to representing categorical data because most of their variation is in the hue component. The default color palette in seaborn is a qualitative palette with ten distinct hues:

These colors have the same ordering as the default matplotlib color palette, "tab10", but they are a bit less intense. Compare:

Seaborn in fact has six variations of matplotlib’s palette, called deep, muted, pastel, bright, dark, and colorblind. These span a range of average luminance and saturation values:

Many people find the moderated hues of the default "deep" palette to be aesthetically pleasing, but they are also less distinct. As a result, they may be more difficult to discriminate in some contexts, which is something to keep in mind when making publication graphics. This comparison can be helpful for estimating how the seaborn color palettes perform when simulating different forms of colorblindess.

When you have an arbitrary number of categories, the easiest approach to finding unique hues is to draw evenly-spaced colors in a circular color space (one where the hue changes while keeping the brightness and saturation constant). This is what most seaborn functions default to when they need to use more colors than are currently set in the default color cycle.

The most common way to do this uses the hls color space, which is a simple transformation of RGB values. We saw this color palette before as a counterexample for how to plot a histogram:

Because of the way the human visual system works, colors that have the same luminance and saturation in terms of their RGB values won’t necessarily look equally intense To remedy this, seaborn provides an interface to the husl system (since renamed to HSLuv), which achieves less intensity variation as you rotate around the color wheel:

When seaborn needs a categorical palette with more colors than are available in the current default, it will use this approach.

Another source of visually pleasing categorical palettes comes from the Color Brewer tool (which also has sequential and diverging palettes, as we’ll see below).

Be aware that the qualitative Color Brewer palettes have different lengths, and the default behavior of color_palette() is to give you the full list:

The second major class of color palettes is called “sequential”. This kind of mapping is appropriate when data range from relatively low or uninteresting values to relatively high or interesting values (or vice versa). As we saw above, the primary dimension of variation in a sequential palette is luminance. Some seaborn functions will default to a sequential palette when you are mapping numeric data. (For historical reasons, both categorical and numeric mappings are specified with the hue parameter in functions like relplot() or displot(), even though numeric mappings use color palettes with relatively little hue variation).

Because they are intended to represent numeric values, the best sequential palettes will be perceptually uniform, meaning that the relative discriminability of two colors is proportional to the difference between the corresponding data values. Seaborn includes four perceptually uniform sequential colormaps: "rocket", "mako", "flare", and "crest". The first two have a very wide luminance range and are well suited for applications such as heatmaps, where colors fill the space they are plotted into:

Because the extreme values of these colormaps approach white, they are not well-suited for coloring elements such as lines or points: it will be difficult to discriminate important values against a white or gray background. The “flare” and “crest” colormaps are a better choice for such plots. They have a more restricted range of luminance variations, which they compensate for with a slightly more pronounced variation in hue. The default direction of the luminance ramp is also reversed, so that smaller values have lighter colors:

It is also possible to use the perceptually uniform colormaps provided by matplotlib, such as "magma" and "viridis":

As with the convention in matplotlib, every continuous colormap has a reversed version, which has the suffix "_r":

One thing to be aware of is that seaborn can generate discrete values from sequential colormaps and, when doing so, it will not use the most extreme values. Compare the discrete version of "rocket" against the continuous version shown above:

Internally, seaborn uses the discrete version for categorical data and the continuous version when in numeric mapping mode. Discrete sequential colormaps can be well-suited for visualizing categorical data with an intrinsic ordering, especially if there is some hue variation.

The perceptually uniform colormaps are difficult to programmatically generate, because they are not based on the RGB color space. The cubehelix system offers an RGB-based compromise: it generates sequential palettes with a linear increase or decrease in brightness and some continuous variation in hue. While not perfectly perceptually uniform, the resulting colormaps have many good properties. Importantly, many aspects of the design process are parameterizable.

Matplotlib has the default cubehelix version built into it:

The default palette returned by the seaborn cubehelix_palette() function is a bit different from the matplotlib default in that it does not rotate as far around the hue wheel or cover as wide a range of intensities. It also reverses the luminance ramp:

Other arguments to cubehelix_palette() control how the palette looks. The two main things you’ll change are the start (a value between 0 and 3) and rot, or number of rotations (an arbitrary value, but usually between -1 and 1)

The more you rotate, the more hue variation you will see:

You can control both how dark and light the endpoints are and their order:

The color_palette() accepts a string code, starting with "ch:", for generating an arbitrary cubehelix palette. You can passs the names of parameters in the string:

And for compactness, each parameter can be specified with its first letter:

For a simpler interface to custom sequential palettes, you can use light_palette() or dark_palette(), which are both seeded with a single color and produce a palette that ramps either from light or dark desaturated values to that color:

As with cubehelix palettes, you can also specify light or dark palettes through color_palette() or anywhere palette is accepted:

Reverse the colormap by adding "_r":

The Color Brewer library also has some good options for sequential palettes. They include palettes with one primary hue:

Along with multi-hue options:

The third class of color palettes is called “diverging”. These are used for data where both large low and high values are interesting and span a midpoint value (often 0) that should be de-emphasized. The rules for choosing good diverging palettes are similar to good sequential palettes, except now there should be two dominant hues in the colormap, one at (or near) each pole. It’s also important that the starting values are of similar brightness and saturation.

Seaborn includes two perceptually uniform diverging palettes: "vlag" and "icefire". They both use blue and red at their poles, which many intuitively processes as “cold” and “hot”:

You can also use the seaborn function diverging_palette() to create a custom colormap for diverging data. This function makes diverging palettes using the husl color system. You pass it two hues (in degrees) and, optionally, the lightness and saturation values for the extremes. Using husl means that the extreme values, and the resulting ramps to the midpoint, while not perfectly perceptually uniform, will be well-balanced:

This is convenient when you want to stray from the boring confines of cold-hot approaches:

It’s also possible to make a palette where the midpoint is dark rather than light:

It’s important to emphasize here that using red and green, while intuitive, should be avoided.

There are a few other good diverging palettes built into matplotlib, including Color Brewer palettes:

And the coolwarm palette, which has less contrast between the middle values and the extremes:

As you can see, there are many options for using color in your visualizations. Seaborn tries both to use good defaults and to offer a lot of flexibility.

This discussion is only the beginning, and there are a number of good resources for learning more about techniques for using color in visualizations. One great example is this series of blog posts from the NASA Earth Observatory. The matplotlib docs also have a nice tutorial that illustrates some of the perceptual properties of their colormaps.

**Examples:**

Example 1 (unknown):
```unknown
sns.color_palette()
```

Example 2 (unknown):
```unknown
sns.color_palette("tab10")
```

Example 3 (unknown):
```unknown
sns.color_palette("hls", 8)
```

Example 4 (unknown):
```unknown
sns.color_palette("husl", 8)
```

---

## The seaborn.objects interface#

**URL:** https://seaborn.pydata.org/tutorial/objects_interface.html

**Contents:**
- The seaborn.objects interface#
- Specifying a plot and mapping data#
  - Setting properties#
  - Mapping properties#
  - Defining groups#
- Transforming data before plotting#
  - Statistical transformation#
  - Resolving overplotting#
  - Creating variables through transformation#
  - Orienting marks and transforms#

The seaborn.objects namespace was introduced in version 0.12 as a completely new interface for making seaborn plots. It offers a more consistent and flexible API, comprising a collection of composable classes for transforming and plotting data. In contrast to the existing seaborn functions, the new interface aims to support end-to-end plot specification and customization without dropping down to matplotlib (although it will remain possible to do so if necessary).

The objects interface is currently experimental and incomplete. It is stable enough for serious use, but there certainly are some rough edges and missing features.

The objects interface should be imported with the following convention:

The seaborn.objects namespace will provide access to all of the relevant classes. The most important is Plot. You specify plots by instantiating a Plot object and calling its methods. Let’s see a simple example:

This code, which produces a scatter plot, should look reasonably familiar. Just as when using seaborn.scatterplot(), we passed a tidy dataframe (penguins) and assigned two of its columns to the x and y coordinates of the plot. But instead of starting with the type of chart and then adding some data assignments, here we started with the data assignments and then added a graphical element.

The Dot class is an example of a Mark: an object that graphically represents data values. Each mark will have a number of properties that can be set to change its appearance:

As with seaborn’s functions, it is also possible to map data values to various graphical properties:

While this basic functionality is not novel, an important difference from the function API is that properties are mapped using the same parameter names that would set them directly (instead of having hue vs. color, etc.). What matters is where the property is defined: passing a value when you initialize Dot will set it directly, whereas assigning a variable when you set up the Plot will map the corresponding data.

Beyond this difference, the objects interface also allows a much wider range of mark properties to be mapped:

The Dot mark represents each data point independently, so the assignment of a variable to a property only has the effect of changing each dot’s appearance. For marks that group or connect observations, such as Line, it also determines the number of distinct graphical elements:

It is also possible to define a grouping without changing any visual properties, by using group:

As with many seaborn functions, the objects interface supports statistical transformations. These are performed by Stat objects, such as Agg:

In the function interface, statistical transformations are possible with some visual representations (e.g. seaborn.barplot()) but not others (e.g. seaborn.scatterplot()). The objects interface more cleanly separates representation and transformation, allowing you to compose Mark and Stat objects:

When forming groups by mapping properties, the Stat transformation is applied to each group separately:

Some seaborn functions also have mechanisms that automatically resolve overplotting, as when seaborn.barplot() “dodges” bars once hue is assigned. The objects interface has less complex default behavior. Bars representing multiple groups will overlap by default:

Nevertheless, it is possible to compose the Bar mark with the Agg stat and a second transformation, implemented by Dodge:

The Dodge class is an example of a Move transformation, which is like a Stat but only adjusts x and y coordinates. The Move classes can be applied with any mark, and it’s not necessary to use a Stat first:

It’s also possible to apply multiple Move operations in sequence:

The Agg stat requires both x and y to already be defined, but variables can also be created through statistical transformation. For example, the Hist stat requires only one of x or y to be defined, and it will create the other by counting observations:

The Hist stat will also create new x values (by binning) when given numeric data:

Notice how we used Bars, rather than Bar for the plot with the continuous x axis. These two marks are related, but Bars has different defaults and works better for continuous histograms. It also produces a different, more efficient matplotlib artist. You will find the pattern of singular/plural marks elsewhere. The plural version is typically optimized for cases with larger numbers of marks.

Some transforms accept both x and y, but add interval data for each coordinate. This is particularly relevant for plotting error bars after aggregating:

When aggregating, dodging, and drawing a bar, the x and y variables are treated differently. Each operation has the concept of an orientation. The Plot tries to determine the orientation automatically based on the data types of the variables. For instance, if we flip the assignment of species and body_mass_g, we’ll get the same plot, but oriented horizontally:

Sometimes, the correct orientation is ambiguous, as when both the x and y variables are numeric. In these cases, you can be explicit by passing the orient parameter to Plot.add():

Most examples this far have produced a single subplot with just one kind of mark on it. But Plot does not limit you to this.

More complex single-subplot graphics can be created by calling Plot.add() repeatedly. Each time it is called, it defines a layer in the plot. For example, we may want to add a scatterplot (now using Dots) and then a regression fit:

Variable mappings that are defined in the Plot constructor will be used for all layers:

You can also define a mapping such that it is used only in a specific layer. This is accomplished by defining the mapping within the call to Plot.add for the relevant layer:

Alternatively, define the layer for the entire plot, but remove it from a specific layer by setting the variable to None:

To recap, there are three ways to specify the value of a mark property: (1) by mapping a variable in all layers, (2) by mapping a variable in a specific layer, and (3) by setting the property directly:

As with seaborn’s figure-level functions (seaborn.displot(), seaborn.catplot(), etc.), the Plot interface can also produce figures with multiple “facets”, or subplots containing subsets of data. This is accomplished with the Plot.facet() method:

Call Plot.facet() with the variables that should be used to define the columns and/or rows of the plot:

You can facet using a variable with a larger number of levels by “wrapping” across the other dimension:

All layers will be faceted unless you explicitly exclude them, which can be useful for providing additional context on each subplot:

An alternate way to produce subplots is Plot.pair(). Like seaborn.PairGrid, this draws all of the data on each subplot, using different variables for the x and/or y coordinates:

You can combine faceting and pairing so long as the operations add subplots on opposite dimensions:

There may be cases where you want multiple subplots to appear in a figure with a more complex structure than what Plot.facet() or Plot.pair() can provide. The current solution is to delegate figure setup to matplotlib and to supply the matplotlib object that Plot should use with the Plot.on() method. This object can be either a matplotlib.axes.Axes, matplotlib.figure.Figure, or matplotlib.figure.SubFigure; the latter is most useful for constructing bespoke subplot layouts:

An important thing to know is that Plot methods clone the object they are called on and return that clone instead of updating the object in place. This means that you can define a common plot spec and then produce several variations on it.

So, take this basic specification:

We could use it to draw a line plot:

Or perhaps a stacked area plot:

The Plot methods are fully declarative. Calling them updates the plot spec, but it doesn’t actually do any plotting. One consequence of this is that methods can be called in any order, and many of them can be called multiple times.

When does the plot actually get rendered? Plot is optimized for use in notebook environments. The rendering is automatically triggered when the Plot gets displayed in the Jupyter REPL. That’s why we didn’t see anything in the example above, where we defined a Plot but assigned it to p rather than letting it return out to the REPL.

To see a plot in a notebook, either return it from the final line of a cell or call Jupyter’s built-in display function on the object. The notebook integration bypasses matplotlib.pyplot entirely, but you can use its figure-display machinery in other contexts by calling Plot.show().

You can also save the plot to a file (or buffer) by calling Plot.save().

The new interface aims to support a deep amount of customization through Plot, reducing the need to switch gears and use matplotlib functionality directly. (But please be patient; not all of the features needed to achieve this goal have been implemented!)

All of the data-dependent properties are controlled by the concept of a Scale and the Plot.scale() method. This method accepts several different types of arguments. One possibility, which is closest to the use of scales in matplotlib, is to pass the name of a function that transforms the coordinates:

Plot.scale() can also control the mappings for semantic properties like color. You can directly pass it any argument that you would pass to the palette parameter in seaborn’s function interface:

Another option is to provide a tuple of (min, max) values, controlling the range that the scale should map into. This works both for numeric properties and for colors:

For additional control, you can pass a Scale object. There are several different types of Scale, each with appropriate parameters. For example, Continuous lets you define the input domain (norm), the output range (values), and the function that maps between them (trans), while Nominal allows you to specify an ordering:

The Scale objects are also how you specify which values should appear as tick labels / in the legend, along with how they appear. For example, the Continuous.tick() method lets you control the density or locations of the ticks, and the Continuous.label() method lets you modify the format:

Plot has a number of methods for simple customization, including Plot.label(), Plot.limit(), and Plot.share():

Finally, Plot supports data-independent theming through the Plot.theme method. Currently, this method accepts a dictionary of matplotlib rc parameters. You can set them directly and/or pass a package of parameters from seaborn’s theming functions:

To change the theme for all Plot instances, update the settings in Plot.config:

**Examples:**

Example 1 (typescript):
```typescript
import seaborn.objects as so
```

Example 2 (unknown):
```unknown
(
    so.Plot(penguins, x="bill_length_mm", y="bill_depth_mm")
    .add(so.Dot())
)
```

Example 3 (unknown):
```unknown
(
    so.Plot(penguins, x="bill_length_mm", y="bill_depth_mm")
    .add(so.Dot(color="g", pointsize=4))
)
```

Example 4 (unknown):
```unknown
(
    so.Plot(
        penguins, x="bill_length_mm", y="bill_depth_mm",
        color="species", pointsize="body_mass_g",
    )
    .add(so.Dot())
)
```

---

## Building structured multi-plot grids#

**URL:** https://seaborn.pydata.org/tutorial/axis_grids.html

**Contents:**
- Building structured multi-plot grids#
- Conditional small multiples#
- Using custom functions#
- Plotting pairwise data relationships#

When exploring multi-dimensional data, a useful approach is to draw multiple instances of the same plot on different subsets of your dataset. This technique is sometimes called either “lattice” or “trellis” plotting, and it is related to the idea of “small multiples”. It allows a viewer to quickly extract a large amount of information about a complex dataset. Matplotlib offers good support for making figures with multiple axes; seaborn builds on top of this to directly link the structure of the plot to the structure of your dataset.

The figure-level functions are built on top of the objects discussed in this chapter of the tutorial. In most cases, you will want to work with those functions. They take care of some important bookkeeping that synchronizes the multiple plots in each grid. This chapter explains how the underlying objects work, which may be useful for advanced applications.

The FacetGrid class is useful when you want to visualize the distribution of a variable or the relationship between multiple variables separately within subsets of your dataset. A FacetGrid can be drawn with up to three dimensions: row, col, and hue. The first two have obvious correspondence with the resulting array of axes; think of the hue variable as a third dimension along a depth axis, where different levels are plotted with different colors.

Each of relplot(), displot(), catplot(), and lmplot() use this object internally, and they return the object when they are finished so that it can be used for further tweaking.

The class is used by initializing a FacetGrid object with a dataframe and the names of the variables that will form the row, column, or hue dimensions of the grid. These variables should be categorical or discrete, and then the data at each level of the variable will be used for a facet along that axis. For example, say we wanted to examine differences between lunch and dinner in the tips dataset:

Initializing the grid like this sets up the matplotlib figure and axes, but doesn’t draw anything on them.

The main approach for visualizing data on this grid is with the FacetGrid.map() method. Provide it with a plotting function and the name(s) of variable(s) in the dataframe to plot. Let’s look at the distribution of tips in each of these subsets, using a histogram:

This function will draw the figure and annotate the axes, hopefully producing a finished plot in one step. To make a relational plot, just pass multiple variable names. You can also provide keyword arguments, which will be passed to the plotting function:

There are several options for controlling the look of the grid that can be passed to the class constructor.

Note that margin_titles isn’t formally supported by the matplotlib API, and may not work well in all cases. In particular, it currently can’t be used with a legend that lies outside of the plot.

The size of the figure is set by providing the height of each facet, along with the aspect ratio:

The default ordering of the facets is derived from the information in the DataFrame. If the variable used to define facets has a categorical type, then the order of the categories is used. Otherwise, the facets will be in the order of appearance of the category levels. It is possible, however, to specify an ordering of any facet dimension with the appropriate *_order parameter:

Any seaborn color palette (i.e., something that can be passed to color_palette()) can be provided. You can also use a dictionary that maps the names of values in the hue variable to valid matplotlib colors:

If you have many levels of one variable, you can plot it along the columns but “wrap” them so that they span multiple rows. When doing this, you cannot use a row variable.

Once you’ve drawn a plot using FacetGrid.map() (which can be called multiple times), you may want to adjust some aspects of the plot. There are also a number of methods on the FacetGrid object for manipulating the figure at a higher level of abstraction. The most general is FacetGrid.set(), and there are other more specialized methods like FacetGrid.set_axis_labels(), which respects the fact that interior facets do not have axis labels. For example:

For even more customization, you can work directly with the underling matplotlib Figure and Axes objects, which are stored as member attributes at figure and axes_dict, respectively. When making a figure without row or column faceting, you can also use the ax attribute to directly access the single axes.

You’re not limited to existing matplotlib and seaborn functions when using FacetGrid. However, to work properly, any function you use must follow a few rules:

It must plot onto the “currently active” matplotlib Axes. This will be true of functions in the matplotlib.pyplot namespace, and you can call matplotlib.pyplot.gca() to get a reference to the current Axes if you want to work directly with its methods.

It must accept the data that it plots in positional arguments. Internally, FacetGrid will pass a Series of data for each of the named positional arguments passed to FacetGrid.map().

It must be able to accept color and label keyword arguments, and, ideally, it will do something useful with them. In most cases, it’s easiest to catch a generic dictionary of **kwargs and pass it along to the underlying plotting function.

Let’s look at minimal example of a function you can plot with. This function will just take a single vector of data for each facet:

If we want to make a bivariate plot, you should write the function so that it accepts the x-axis variable first and the y-axis variable second:

Because matplotlib.pyplot.scatter() accepts color and label keyword arguments and does the right thing with them, we can add a hue facet without any difficulty:

Sometimes, though, you’ll want to map a function that doesn’t work the way you expect with the color and label keyword arguments. In this case, you’ll want to explicitly catch them and handle them in the logic of your custom function. For example, this approach will allow use to map matplotlib.pyplot.hexbin(), which otherwise does not play well with the FacetGrid API:

PairGrid also allows you to quickly draw a grid of small subplots using the same plot type to visualize data in each. In a PairGrid, each row and column is assigned to a different variable, so the resulting plot shows each pairwise relationship in the dataset. This style of plot is sometimes called a “scatterplot matrix”, as this is the most common way to show each relationship, but PairGrid is not limited to scatterplots.

It’s important to understand the differences between a FacetGrid and a PairGrid. In the former, each facet shows the same relationship conditioned on different levels of other variables. In the latter, each plot shows a different relationship (although the upper and lower triangles will have mirrored plots). Using PairGrid can give you a very quick, very high-level summary of interesting relationships in your dataset.

The basic usage of the class is very similar to FacetGrid. First you initialize the grid, then you pass plotting function to a map method and it will be called on each subplot. There is also a companion function, pairplot() that trades off some flexibility for faster plotting.

It’s possible to plot a different function on the diagonal to show the univariate distribution of the variable in each column. Note that the axis ticks won’t correspond to the count or density axis of this plot, though.

A very common way to use this plot colors the observations by a separate categorical variable. For example, the iris dataset has four measurements for each of three different species of iris flowers so you can see how they differ.

By default every numeric column in the dataset is used, but you can focus on particular relationships if you want.

It’s also possible to use a different function in the upper and lower triangles to emphasize different aspects of the relationship.

The square grid with identity relationships on the diagonal is actually just a special case, and you can plot with different variables in the rows and columns.

Of course, the aesthetic attributes are configurable. For instance, you can use a different palette (say, to show an ordering of the hue variable) and pass keyword arguments into the plotting functions.

PairGrid is flexible, but to take a quick look at a dataset, it can be easier to use pairplot(). This function uses scatterplots and histograms by default, although a few other kinds will be added (currently, you can also plot regression plots on the off-diagonals and KDEs on the diagonal).

You can also control the aesthetics of the plot with keyword arguments, and it returns the PairGrid instance for further tweaking.

**Examples:**

Example 1 (unknown):
```unknown
tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, col="time")
```

Example 2 (unknown):
```unknown
g = sns.FacetGrid(tips, col="time")
g.map(sns.histplot, "tip")
```

Example 3 (unknown):
```unknown
g = sns.FacetGrid(tips, col="sex", hue="smoker")
g.map(sns.scatterplot, "total_bill", "tip", alpha=.7)
g.add_legend()
```

Example 4 (unknown):
```unknown
g = sns.FacetGrid(tips, row="smoker", col="time", margin_titles=True)
g.map(sns.regplot, "size", "total_bill", color=".3", fit_reg=False, x_jitter=.1)
```

---

## Statistical estimation and error bars#

**URL:** https://seaborn.pydata.org/tutorial/error_bars.html

**Contents:**
- Statistical estimation and error bars#
- Measures of data spread#
  - Standard deviation error bars#
  - Percentile interval error bars#
- Measures of estimate uncertainty#
  - Standard error bars#
  - Confidence interval error bars#
  - Custom error bars#
- Error bars on regression fits#
- Are error bars enough?#

Data visualization sometimes involves a step of aggregation or estimation, where multiple data points are reduced to a summary statistic such as the mean or median. When showing a summary statistic, it is usually appropriate to add error bars, which provide a visual cue about how well the summary represents the underlying data points.

Several seaborn functions will automatically calculate both summary statistics and the error bars when given a full dataset. This chapter explains how you can control what the error bars show and why you might choose each of the options that seaborn affords.

The error bars around an estimate of central tendency can show one of two general things: either the range of uncertainty about the estimate or the spread of the underlying data around it. These measures are related: given the same sample size, estimates will be more uncertain when data has a broader spread. But uncertainty will decrease as sample sizes grow, whereas spread will not.

In seaborn, there are two approaches for constructing each kind of error bar. One approach is parametric, using a formula that relies on assumptions about the shape of the distribution. The other approach is nonparametric, using only the data that you provide.

Your choice is made with the errorbar parameter, which exists for each function that does estimation as part of plotting. This parameter accepts the name of the method to use and, optionally, a parameter that controls the size of the interval. The choices can be defined in a 2D taxonomy that depends on what is shown and how it is constructed:

You will note that the size parameter is defined differently for the parametric and nonparametric approaches. For parametric error bars, it is a scalar factor that is multiplied by the statistic defining the error (standard error or standard deviation). For nonparametric error bars, it is a percentile width. This is explained further for each specific approach below.

The errorbar API described here was introduced in seaborn v0.12. In prior versions, the only options were to show a bootstrap confidence interval or a standard deviation, with the choice controlled by the ci parameter (i.e., ci=<size> or ci="sd").

To compare the different parameterizations, we’ll use the following helper function:

Error bars that represent data spread present a compact display of the distribution, using three numbers where boxplot() would use 5 or more and violinplot() would use a complicated algorithm.

Standard deviation error bars are the simplest to explain, because the standard deviation is a familiar statistic. It is the average distance from each data point to the sample mean. By default, errorbar="sd" will draw error bars at +/- 1 sd around the estimate, but the range can be increased by passing a scaling size parameter. Note that, assuming normally-distributed data, ~68% of the data will lie within one standard deviation, ~95% will lie within two, and ~99.7% will lie within three:

Percentile intervals also represent the range where some amount of the data fall, but they do so by computing those percentiles directly from your sample. By default, errorbar="pi" will show a 95% interval, ranging from the 2.5 to the 97.5 percentiles. You can choose a different range by passing a size parameter, e.g., to show the inter-quartile range:

The standard deviation error bars will always be symmetrical around the estimate. This can be a problem when the data are skewed, especially if there are natural bounds (e.g., if the data represent a quantity that can only be positive). In some cases, standard deviation error bars may extend to “impossible” values. The nonparametric approach does not have this problem, because it can account for asymmetrical spread and will never extend beyond the range of the data.

If your data are a random sample from a larger population, then the mean (or other estimate) will be an imperfect measure of the true population average. Error bars that show estimate uncertainty try to represent the range of likely values for the true parameter.

The standard error statistic is related to the standard deviation: in fact it is just the standard deviation divided by the square root of the sample size. The default, with errorbar="se", draws an interval +/-1 standard error from the mean:

The nonparametric approach to representing uncertainty uses bootstrapping: a procedure where the dataset is randomly resampled with replacement a number of times, and the estimate is recalculated from each resample. This procedure creates a distribution of statistics approximating the distribution of values that you could have gotten for your estimate if you had a different sample.

The confidence interval is constructed by taking a percentile interval of the bootstrap distribution. By default errorbar="ci" draws a 95% confidence interval:

The seaborn terminology is somewhat specific, because a confidence interval in statistics can be parametric or nonparametric. To draw a parametric confidence interval, you scale the standard error, using a formula similar to the one mentioned above. For example, an approximate 95% confidence interval can be constructed by taking the mean +/- two standard errors:

The nonparametric bootstrap has advantages similar to those of the percentile interval: it will naturally adapt to skewed and bounded data in a way that a standard error interval cannot. It is also more general. While the standard error formula is specific to the mean, error bars can be computed using the bootstrap for any estimator:

Bootstrapping involves randomness, and the error bars will appear slightly different each time you run the code that creates them. A few parameters control this. One sets the number of iterations (n_boot): with more iterations, the resulting intervals will be more stable. The other sets the seed for the random number generator, which will ensure identical results:

Because of its iterative process, bootstrap intervals can be expensive to compute, especially for large datasets. But because uncertainty decreases with sample size, it may be more informative in that case to use an error bar that represents data spread.

If these recipes are not sufficient, it is also possible to pass a generic function to the errorbar parameter. This function should take a vector and produce a pair of values representing the minimum and maximum points of the interval:

(In practice, you could show the full range of the data with errorbar=("pi", 100) rather than the custom function shown above).

Note that seaborn functions cannot currently draw error bars from values that have been calculated externally, although matplotlib functions can be used to add such error bars to seaborn plots.

The preceding discussion has focused on error bars shown around parameter estimates for aggregate data. Error bars also arise in seaborn when estimating regression models to visualize relationships. Here, the error bars will be represented by a “band” around the regression line:

Currently, the error bars on a regression estimate are less flexible, only showing a confidence interval with a size set through ci=. This may change in the future.

You should always ask yourself whether it’s best to use a plot that displays only a summary statistic and error bar. In many cases, it isn’t.

If you are interested in questions about summaries (such as whether the mean value differs between groups or increases over time), aggregation reduces the complexity of the plot and makes those inferences easier. But in doing so, it obscures valuable information about the underlying data points, such as the shape of the distributions and the presence of outliers.

When analyzing your own data, don’t be satisfied with summary statistics. Always look at the underlying distributions too. Sometimes, it can be helpful to combine both perspectives into the same figure. Many seaborn functions can help with this task, especially those discussed in the categorical tutorial.

**Examples:**

Example 1 (python):
```python
def plot_errorbars(arg, **kws):
    np.random.seed(sum(map(ord, "error_bars")))
    x = np.random.normal(0, 1, 100)
    f, axs = plt.subplots(2, figsize=(7, 2), sharex=True, layout="tight")
    sns.pointplot(x=x, errorbar=arg, **kws, capsize=.3, ax=axs[0])
    sns.stripplot(x=x, jitter=.3, ax=axs[1])
```

Example 2 (unknown):
```unknown
plot_errorbars("sd")
```

Example 3 (unknown):
```unknown
plot_errorbars(("pi", 50))
```

Example 4 (unknown):
```unknown
plot_errorbars("se")
```

---

## Properties of Mark objects#

**URL:** https://seaborn.pydata.org/tutorial/properties.html

**Contents:**
- Properties of Mark objects#
- Coordinate properties#
  - x, y, xmin, xmax, ymin, ymax#
- Color properties#
  - color, fillcolor, edgecolor#
  - alpha, fillalpha, edgealpha#
- Style properties#
  - fill#
  - marker#
  - linestyle, edgestyle#

Coordinate properties determine where a mark is drawn on a plot. Canonically, the x coordinate is the horizontal position and the y coordinate is the vertical position. Some marks accept a span (i.e., min, max) parameterization for one or both variables. Others may accept x and y but also use a baseline parameter to show a span. The layer’s orient parameter determines how this works.

If a variable does not contain numeric data, its scale will apply a conversion so that data can be drawn on a screen. For instance, Nominal scales assign an integer index to each distinct category, and Temporal scales represent dates as the number of days from a reference “epoch”:

A Continuous scale can also apply a nonlinear transform between data values and spatial positions:

All marks can be given a color, and many distinguish between the color of the mark’s “edge” and “fill”. Often, simply using color will set both, while the more-specific properties allow further control:

When the color property is mapped, the default palette depends on the type of scale. Nominal scales use discrete, unordered hues, while continuous scales (including temporal ones) use a sequential gradient:

The default continuous scale is subject to change in future releases to improve discriminability.

Color scales are parameterized by the name of a palette, such as 'viridis', 'rocket', or 'deep'. Some palette names can include parameters, including simple gradients (e.g. 'dark:blue') or the cubehelix system (e.g. 'ch:start=.2,rot=-.4`). See the color palette tutorial for guidance on making an appropriate selection.

Continuous scales can also be parameterized by a tuple of colors that the scale should interpolate between. When using a nominal scale, it is possible to provide either the name of the palette (which will be discretely-sampled, if necessary), a list of individual color values, or a dictionary directly mapping data values to colors.

Individual colors may be specified in a wide range of formats. These include indexed references to the current color cycle ('C0'), single-letter shorthands ('b'), grayscale values ('.4'), RGB hex codes ('#4c72b0'), X11 color names ('seagreen'), and XKCD color survey names ('purpleish'):

The alpha property determines the mark’s opacity. Lowering the alpha can be helpful for representing density in the case of overplotting:

Mapping the alpha property can also be useful even when marks do not overlap because it conveys a sense of importance and can be combined with a color scale to represent two variables. Moreover, colors with lower alpha appear less saturated, which can improve the appearance of larger filled marks (such as bars).

As with color, some marks define separate edgealpha and fillalpha properties for additional control.

The fill property is relevant to marks with a distinction between the edge and interior and determines whether the interior is visible. It is a boolean state: fill can be set only to True or False:

The marker property is relevant for dot marks and some line marks. The API for specifying markers is very flexible, as detailed in the matplotlib API docs: matplotlib.markers.

Markers can be specified using a number of simple string codes:

They can also be programatically generated using a (num_sides, fill_style, angle) tuple:

See the matplotlib docs for additional formats, including mathtex character codes ('$...$') and arrays of vertices.

A marker property is always mapped with a nominal scale; there is no inherent ordering to the different shapes. If no scale is provided, the plot will programmatically generate a suitably large set of unique markers:

While this ensures that the shapes are technically distinct, bear in mind that — in most cases — it will be difficult to tell the markers apart if more than a handful are used in a single plot.

The default marker scale is subject to change in future releases to improve discriminability.

The linestyle property is relevant to line marks, and the edgestyle property is relevant to a number of marks with “edges. Both properties determine the “dashing” of a line in terms of on-off segments.

Dashes can be specified with a small number of shorthand codes ('-', '--', '-.', and ':') or programatically using (on, off, ...) tuples. In the tuple specification, the unit is equal to the linewidth:

The pointsize property is relevant to dot marks and to line marks that can show markers at individual data points. The units correspond to the diameter of the mark in points.

Note that, while the parameterization corresponds to diameter, scales will be applied with a square root transform so that data values are linearly proportional to area:

The linewidth property is relevant to line marks and determines their thickness. The value should be non-negative and has point units:

The edgewidth property is akin to linewidth but applies to marks with an edge/fill rather than to lines. It also has a different default range when used in a scale. The units are the same:

The stroke property is akin to edgewidth but applies when a dot mark is defined by its stroke rather than its fill. It also has a slightly different default scale range, but otherwise behaves similarly:

The halign and valign properties control the horizontal and vertical alignment of text marks. The options for horizontal alignment are 'left', 'right', and 'center', while the options for vertical alignment are 'top', 'bottom', 'center', 'baseline', and 'center_baseline'.

The fontsize property controls the size of textual marks. The value has point units:

The offset property controls the spacing between a text mark and its anchor position. It applies when not using center alignment (i.e., when using left/right or top/bottom). The value has point units.

The text property is used to set the content of a textual mark. It is always used literally (not mapped), and cast to string when necessary.

The group property is special in that it does not change anything about the mark’s appearance but defines additional data subsets that transforms should operate on independently.

---

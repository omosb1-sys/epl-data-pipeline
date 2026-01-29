# Pandas - User Guide

**Pages:** 39

---

## Time deltas#

**URL:** https://pandas.pydata.org/docs/user_guide/timedeltas.html

**Contents:**
- Time deltas#
- Parsing#
  - to_timedelta#
  - Timedelta limitations#
- Operations#
- Reductions#
- Frequency conversion#
- Attributes#
- TimedeltaIndex#
  - Generating ranges of time deltas#

Timedeltas are differences in times, expressed in difference units, e.g. days, hours, minutes, seconds. They can be both positive and negative.

Timedelta is a subclass of datetime.timedelta, and behaves in a similar manner, but allows compatibility with np.timedelta64 types as well as a host of custom representation, parsing, and attributes.

You can construct a Timedelta scalar through various arguments, including ISO 8601 Duration strings.

DateOffsets (Day, Hour, Minute, Second, Milli, Micro, Nano) can also be used in construction.

Further, operations among the scalars yield another scalar Timedelta.

Using the top-level pd.to_timedelta, you can convert a scalar, array, list, or Series from a recognized timedelta format / value into a Timedelta type. It will construct Series if the input is a Series, a scalar if the input is scalar-like, otherwise it will output a TimedeltaIndex.

You can parse a single string to a Timedelta:

or a list/array of strings:

The unit keyword argument specifies the unit of the Timedelta if the input is numeric:

If a string or array of strings is passed as an input then the unit keyword argument will be ignored. If a string without units is passed then the default unit of nanoseconds is assumed.

pandas represents Timedeltas in nanosecond resolution using 64 bit integers. As such, the 64 bit integer limits determine the Timedelta limits.

You can operate on Series/DataFrames and construct timedelta64[ns] Series through subtraction operations on datetime64[ns] Series, or Timestamps.

Operations with scalars from a timedelta64[ns] series:

Series of timedeltas with NaT values are supported:

Elements can be set to NaT using np.nan analogously to datetimes:

Operands can also appear in a reversed order (a singular object operated with a Series):

min, max and the corresponding idxmin, idxmax operations are supported on frames:

min, max, idxmin, idxmax operations are supported on Series as well. A scalar result will be a Timedelta.

You can fillna on timedeltas, passing a timedelta to get a particular value.

You can also negate, multiply and use abs on Timedeltas:

Numeric reduction operation for timedelta64[ns] will return Timedelta objects. As usual NaT are skipped during evaluation.

Timedelta Series and TimedeltaIndex, and Timedelta can be converted to other frequencies by astyping to a specific timedelta dtype.

For timedelta64 resolutions other than the supported “s”, “ms”, “us”, “ns”, an alternative is to divide by another timedelta object. Note that division by the NumPy scalar is true division, while astyping is equivalent of floor division.

Dividing or multiplying a timedelta64[ns] Series by an integer or integer Series yields another timedelta64[ns] dtypes Series.

Rounded division (floor-division) of a timedelta64[ns] Series by a scalar Timedelta gives a series of integers.

The mod (%) and divmod operations are defined for Timedelta when operating with another timedelta-like or with a numeric argument.

You can access various components of the Timedelta or TimedeltaIndex directly using the attributes days,seconds,microseconds,nanoseconds. These are identical to the values returned by datetime.timedelta, in that, for example, the .seconds attribute represents the number of seconds >= 0 and < 1 day. These are signed according to whether the Timedelta is signed.

These operations can also be directly accessed via the .dt property of the Series as well.

Note that the attributes are NOT the displayed values of the Timedelta. Use .components to retrieve the displayed values.

You can access the value of the fields for a scalar Timedelta directly.

You can use the .components property to access a reduced form of the timedelta. This returns a DataFrame indexed similarly to the Series. These are the displayed values of the Timedelta.

You can convert a Timedelta to an ISO 8601 Duration string with the .isoformat method

To generate an index with time delta, you can use either the TimedeltaIndex or the timedelta_range() constructor.

Using TimedeltaIndex you can pass string-like, Timedelta, timedelta, or np.timedelta64 objects. Passing np.nan/pd.NaT/nat will represent missing values.

The string ‘infer’ can be passed in order to set the frequency of the index as the inferred frequency upon creation:

Similar to date_range(), you can construct regular ranges of a TimedeltaIndex using timedelta_range(). The default frequency for timedelta_range is calendar day:

Various combinations of start, end, and periods can be used with timedelta_range:

The freq parameter can passed a variety of frequency aliases:

Specifying start, end, and periods will generate a range of evenly spaced timedeltas from start to end inclusively, with periods number of elements in the resulting TimedeltaIndex:

Similarly to other of the datetime-like indices, DatetimeIndex and PeriodIndex, you can use TimedeltaIndex as the index of pandas objects.

Selections work similarly, with coercion on string-likes and slices:

Furthermore you can use partial string selection and the range will be inferred:

Finally, the combination of TimedeltaIndex with DatetimeIndex allow certain combination operations that are NaT preserving:

Similarly to frequency conversion on a Series above, you can convert these indices to yield another Index.

Scalars type ops work as well. These can potentially return a different type of index.

Similar to timeseries resampling, we can resample with a TimedeltaIndex.

**Examples:**

Example 1 (sql):
```sql
In [1]: import datetime

# strings
In [2]: pd.Timedelta("1 days")
Out[2]: Timedelta('1 days 00:00:00')

In [3]: pd.Timedelta("1 days 00:00:00")
Out[3]: Timedelta('1 days 00:00:00')

In [4]: pd.Timedelta("1 days 2 hours")
Out[4]: Timedelta('1 days 02:00:00')

In [5]: pd.Timedelta("-1 days 2 min 3us")
Out[5]: Timedelta('-2 days +23:57:59.999997')

# like datetime.timedelta
# note: these MUST be specified as keyword arguments
In [6]: pd.Timedelta(days=1, seconds=1)
Out[6]: Timedelta('1 days 00:00:01')

# integers with a unit
In [7]: pd.Timedelta(1, unit="d")
Out[7]: Timedelta('1 days 00:00:00')

# from a datetime.timedelta/np.timedelta64
In [8]: pd.Timedelta(datetime.timedelta(days=1, seconds=1))
Out[8]: Timedelta('1 days 00:00:01')

In [9]: pd.Timedelta(np.timedelta64(1, "ms"))
Out[9]: Timedelta('0 days 00:00:00.001000')

# negative Timedeltas have this string repr
# to be more consistent with datetime.timedelta conventions
In [10]: pd.Timedelta("-1us")
Out[10]: Timedelta('-1 days +23:59:59.999999')

# a NaT
In [11]: pd.Timedelta("nan")
Out[11]: NaT

In [12]: pd.Timedelta("nat")
Out[12]: NaT

# ISO 8601 Duration strings
In [13]: pd.Timedelta("P0DT0H1M0S")
Out[13]: Timedelta('0 days 00:01:00')

In [14]: pd.Timedelta("P0DT0H0M0.000000123S")
Out[14]: Timedelta('0 days 00:00:00.000000123')
```

Example 2 (json):
```json
In [15]: pd.Timedelta(pd.offsets.Second(2))
Out[15]: Timedelta('0 days 00:00:02')
```

Example 3 (json):
```json
In [16]: pd.Timedelta(pd.offsets.Day(2)) + pd.Timedelta(pd.offsets.Second(2)) + pd.Timedelta(
   ....:     "00:00:00.000123"
   ....: )
   ....: 
Out[16]: Timedelta('2 days 00:00:02.000123')
```

Example 4 (json):
```json
In [17]: pd.to_timedelta("1 days 06:05:01.00003")
Out[17]: Timedelta('1 days 06:05:01.000030')

In [18]: pd.to_timedelta("15.5us")
Out[18]: Timedelta('0 days 00:00:00.000015500')
```

---

## Enhancing performance#

**URL:** https://pandas.pydata.org/docs/user_guide/enhancingperf.html

**Contents:**
- Enhancing performance#
- Cython (writing C extensions for pandas)#
  - Pure Python#
  - Plain Cython#
  - Declaring C types#
  - Using ndarray#
  - Disabling compiler directives#
- Numba (JIT compilation)#
  - pandas Numba Engine#
  - Custom Function Examples#

In this part of the tutorial, we will investigate how to speed up certain functions operating on pandas DataFrame using Cython, Numba and pandas.eval(). Generally, using Cython and Numba can offer a larger speedup than using pandas.eval() but will require a lot more code.

In addition to following the steps in this tutorial, users interested in enhancing performance are highly encouraged to install the recommended dependencies for pandas. These dependencies are often not installed by default, but will offer speed improvements if present.

For many use cases writing pandas in pure Python and NumPy is sufficient. In some computationally heavy applications however, it can be possible to achieve sizable speed-ups by offloading work to cython.

This tutorial assumes you have refactored as much as possible in Python, for example by trying to remove for-loops and making use of NumPy vectorization. It’s always worth optimising in Python first.

This tutorial walks through a “typical” process of cythonizing a slow computation. We use an example from the Cython documentation but in the context of pandas. Our final cythonized solution is around 100 times faster than the pure Python solution.

We have a DataFrame to which we want to apply a function row-wise.

Here’s the function in pure Python:

We achieve our result by using DataFrame.apply() (row-wise):

Let’s take a look and see where the time is spent during this operation using the prun ipython magic function:

By far the majority of time is spend inside either integrate_f or f, hence we’ll concentrate our efforts cythonizing these two functions.

First we’re going to need to import the Cython magic function to IPython:

Now, let’s simply copy our functions over to Cython:

This has improved the performance compared to the pure Python approach by one-third.

We can annotate the function variables and return types as well as use cdef and cpdef to improve performance:

Annotating the functions with C types yields an over ten times performance improvement compared to the original Python implementation.

When re-profiling, time is spent creating a Series from each row, and calling __getitem__ from both the index and the series (three times for each row). These Python function calls are expensive and can be improved by passing an np.ndarray.

This implementation creates an array of zeros and inserts the result of integrate_f_typed applied over each row. Looping over an ndarray is faster in Cython than looping over a Series object.

Since apply_integrate_f is typed to accept an np.ndarray, Series.to_numpy() calls are needed to utilize this function.

Performance has improved from the prior implementation by almost ten times.

The majority of the time is now spent in apply_integrate_f. Disabling Cython’s boundscheck and wraparound checks can yield more performance.

However, a loop indexer i accessing an invalid location in an array would cause a segfault because memory access isn’t checked. For more about boundscheck and wraparound, see the Cython docs on compiler directives.

An alternative to statically compiling Cython code is to use a dynamic just-in-time (JIT) compiler with Numba.

Numba allows you to write a pure Python function which can be JIT compiled to native machine instructions, similar in performance to C, C++ and Fortran, by decorating your function with @jit.

Numba works by generating optimized machine code using the LLVM compiler infrastructure at import time, runtime, or statically (using the included pycc tool). Numba supports compilation of Python to run on either CPU or GPU hardware and is designed to integrate with the Python scientific software stack.

The @jit compilation will add overhead to the runtime of the function, so performance benefits may not be realized especially when using small data sets. Consider caching your function to avoid compilation overhead each time your function is run.

Numba can be used in 2 ways with pandas:

Specify the engine="numba" keyword in select pandas methods

Define your own Python function decorated with @jit and pass the underlying NumPy array of Series or DataFrame (using Series.to_numpy()) into the function

If Numba is installed, one can specify engine="numba" in select pandas methods to execute the method using Numba. Methods that support engine="numba" will also have an engine_kwargs keyword that accepts a dictionary that allows one to specify "nogil", "nopython" and "parallel" keys with boolean values to pass into the @jit decorator. If engine_kwargs is not specified, it defaults to {"nogil": False, "nopython": True, "parallel": False} unless otherwise specified.

In terms of performance, the first time a function is run using the Numba engine will be slow as Numba will have some function compilation overhead. However, the JIT compiled functions are cached, and subsequent calls will be fast. In general, the Numba engine is performant with a larger amount of data points (e.g. 1+ million).

If your compute hardware contains multiple CPUs, the largest performance gain can be realized by setting parallel to True to leverage more than 1 CPU. Internally, pandas leverages numba to parallelize computations over the columns of a DataFrame; therefore, this performance benefit is only beneficial for a DataFrame with a large number of columns.

A custom Python function decorated with @jit can be used with pandas objects by passing their NumPy array representations with Series.to_numpy().

In this example, using Numba was faster than Cython.

Numba can also be used to write vectorized functions that do not require the user to explicitly loop over the observations of a vector; a vectorized function will be applied to each row automatically. Consider the following example of doubling each observation:

Numba is best at accelerating functions that apply numerical functions to NumPy arrays. If you try to @jit a function that contains unsupported Python or NumPy code, compilation will revert object mode which will mostly likely not speed up your function. If you would prefer that Numba throw an error if it cannot compile a function in a way that speeds up your code, pass Numba the argument nopython=True (e.g. @jit(nopython=True)). For more on troubleshooting Numba modes, see the Numba troubleshooting page.

Using parallel=True (e.g. @jit(parallel=True)) may result in a SIGABRT if the threading layer leads to unsafe behavior. You can first specify a safe threading layer before running a JIT function with parallel=True.

Generally if the you encounter a segfault (SIGSEGV) while using Numba, please report the issue to the Numba issue tracker.

The top-level function pandas.eval() implements performant expression evaluation of Series and DataFrame. Expression evaluation allows operations to be expressed as strings and can potentially provide a performance improvement by evaluate arithmetic and boolean expression all at once for large DataFrame.

You should not use eval() for simple expressions or for expressions involving small DataFrames. In fact, eval() is many orders of magnitude slower for smaller expressions or objects than plain Python. A good rule of thumb is to only use eval() when you have a DataFrame with more than 10,000 rows.

These operations are supported by pandas.eval():

Arithmetic operations except for the left shift (<<) and right shift (>>) operators, e.g., df + 2 * pi / s ** 4 % 42 - the_golden_ratio

Comparison operations, including chained comparisons, e.g., 2 < df < df2

Boolean operations, e.g., df < df2 and df3 < df4 or not df_bool

list and tuple literals, e.g., [1, 2] or (1, 2)

Attribute access, e.g., df.a

Subscript expressions, e.g., df[0]

Simple variable evaluation, e.g., pd.eval("df") (this is not very useful)

Math functions: sin, cos, exp, log, expm1, log1p, sqrt, sinh, cosh, tanh, arcsin, arccos, arctan, arccosh, arcsinh, arctanh, abs, arctan2 and log10.

The following Python syntax is not allowed:

Function calls other than math functions.

list/set/dict comprehensions

Literal dict and set expressions

Generator expressions

Boolean expressions consisting of only scalar values

Neither simple or compound statements are allowed. This includes for, while, and if.

You must explicitly reference any local variable that you want to use in an expression by placing the @ character in front of the name. This mechanism is the same for both DataFrame.query() and DataFrame.eval(). For example,

If you don’t prefix the local variable with @, pandas will raise an exception telling you the variable is undefined.

When using DataFrame.eval() and DataFrame.query(), this allows you to have a local variable and a DataFrame column with the same name in an expression.

pandas.eval() will raise an exception if you cannot use the @ prefix because it isn’t defined in that context.

In this case, you should simply refer to the variables like you would in standard Python.

There are two different expression syntax parsers.

The default 'pandas' parser allows a more intuitive syntax for expressing query-like operations (comparisons, conjunctions and disjunctions). In particular, the precedence of the & and | operators is made equal to the precedence of the corresponding boolean operations and and or.

For example, the above conjunction can be written without parentheses. Alternatively, you can use the 'python' parser to enforce strict Python semantics.

The same expression can be “anded” together with the word and as well:

The and and or operators here have the same precedence that they would in Python.

There are two different expression engines.

The 'numexpr' engine is the more performant engine that can yield performance improvements compared to standard Python syntax for large DataFrame. This engine requires the optional dependency numexpr to be installed.

The 'python' engine is generally not useful except for testing other evaluation engines against it. You will achieve no performance benefits using eval() with engine='python' and may incur a performance hit.

In addition to the top level pandas.eval() function you can also evaluate an expression in the “context” of a DataFrame.

Any expression that is a valid pandas.eval() expression is also a valid DataFrame.eval() expression, with the added benefit that you don’t have to prefix the name of the DataFrame to the column(s) you’re interested in evaluating.

In addition, you can perform assignment of columns within an expression. This allows for formulaic evaluation. The assignment target can be a new column name or an existing column name, and it must be a valid Python identifier.

A copy of the DataFrame with the new or modified columns is returned, and the original frame is unchanged.

Multiple column assignments can be performed by using a multi-line string.

The equivalent in standard Python would be

pandas.eval() works well with expressions containing large arrays.

DataFrame arithmetic:

DataFrame comparison:

DataFrame arithmetic with unaligned axes.

should be performed in Python. An exception will be raised if you try to perform any boolean/bitwise operations with scalar operands that are not of type bool or np.bool_.

Here is a plot showing the running time of pandas.eval() as function of the size of the frame involved in the computation. The two lines are two different engines.

You will only see the performance benefits of using the numexpr engine with pandas.eval() if your DataFrame has more than approximately 100,000 rows.

This plot was created using a DataFrame with 3 columns each containing floating point values generated using numpy.random.randn().

Expressions that would result in an object dtype or involve datetime operations because of NaT must be evaluated in Python space, but part of an expression can still be evaluated with numexpr. For example:

The numeric part of the comparison (nums == 1) will be evaluated by numexpr and the object part of the comparison ("strings == 'a') will be evaluated by Python.

**Examples:**

Example 1 (json):
```json
In [1]: df = pd.DataFrame(
   ...:     {
   ...:         "a": np.random.randn(1000),
   ...:         "b": np.random.randn(1000),
   ...:         "N": np.random.randint(100, 1000, (1000)),
   ...:         "x": "x",
   ...:     }
   ...: )
   ...: 

In [2]: df
Out[2]: 
            a         b    N  x
0    0.469112 -0.218470  585  x
1   -0.282863 -0.061645  841  x
2   -1.509059 -0.723780  251  x
3   -1.135632  0.551225  972  x
4    1.212112 -0.497767  181  x
..        ...       ...  ... ..
995 -1.512743  0.874737  374  x
996  0.933753  1.120790  246  x
997 -0.308013  0.198768  157  x
998 -0.079915  1.757555  977  x
999 -1.010589 -1.115680  770  x

[1000 rows x 4 columns]
```

Example 2 (python):
```python
In [3]: def f(x):
   ...:     return x * (x - 1)
   ...: 

In [4]: def integrate_f(a, b, N):
   ...:     s = 0
   ...:     dx = (b - a) / N
   ...:     for i in range(N):
   ...:         s += f(a + i * dx)
   ...:     return s * dx
   ...:
```

Example 3 (unknown):
```unknown
In [5]: %timeit df.apply(lambda x: integrate_f(x["a"], x["b"], x["N"]), axis=1)
82.4 ms +- 2.27 ms per loop (mean +- std. dev. of 7 runs, 10 loops each)
```

Example 4 (javascript):
```javascript
# most time consuming 4 calls
In [6]: %prun -l 4 df.apply(lambda x: integrate_f(x["a"], x["b"], x["N"]), axis=1)  # noqa E999
         605956 function calls (605938 primitive calls) in 0.173 seconds

   Ordered by: internal time
   List reduced from 163 to 4 due to restriction <4>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     1000    0.102    0.000    0.153    0.000 <ipython-input-4-c2a74e076cf0>:1(integrate_f)
   552423    0.051    0.000    0.051    0.000 <ipython-input-3-c138bdd570e3>:1(f)
     3000    0.003    0.000    0.013    0.000 series.py:1107(__getitem__)
     3000    0.002    0.000    0.006    0.000 series.py:1232(_get_value)
```

---

## Categorical data#

**URL:** https://pandas.pydata.org/docs/user_guide/categorical.html

**Contents:**
- Categorical data#
- Object creation#
  - Series creation#
  - DataFrame creation#
  - Controlling behavior#
  - Regaining original data#
- CategoricalDtype#
  - Equality semantics#
- Description#
- Working with categories#

This is an introduction to pandas categorical data type, including a short comparison with R’s factor.

Categoricals are a pandas data type corresponding to categorical variables in statistics. A categorical variable takes on a limited, and usually fixed, number of possible values (categories; levels in R). Examples are gender, social class, blood type, country affiliation, observation time or rating via Likert scales.

In contrast to statistical categorical variables, categorical data might have an order (e.g. ‘strongly agree’ vs ‘agree’ or ‘first observation’ vs. ‘second observation’), but numerical operations (additions, divisions, …) are not possible.

All values of categorical data are either in categories or np.nan. Order is defined by the order of categories, not lexical order of the values. Internally, the data structure consists of a categories array and an integer array of codes which point to the real value in the categories array.

The categorical data type is useful in the following cases:

A string variable consisting of only a few different values. Converting such a string variable to a categorical variable will save some memory, see here.

The lexical order of a variable is not the same as the logical order (“one”, “two”, “three”). By converting to a categorical and specifying an order on the categories, sorting and min/max will use the logical order instead of the lexical order, see here.

As a signal to other Python libraries that this column should be treated as a categorical variable (e.g. to use suitable statistical methods or plot types).

See also the API docs on categoricals.

Categorical Series or columns in a DataFrame can be created in several ways:

By specifying dtype="category" when constructing a Series:

By converting an existing Series or column to a category dtype:

By using special functions, such as cut(), which groups data into discrete bins. See the example on tiling in the docs.

By passing a pandas.Categorical object to a Series or assigning it to a DataFrame.

Categorical data has a specific category dtype:

Similar to the previous section where a single column was converted to categorical, all columns in a DataFrame can be batch converted to categorical either during or after construction.

This can be done during construction by specifying dtype="category" in the DataFrame constructor:

Note that the categories present in each column differ; the conversion is done column by column, so only labels present in a given column are categories:

Analogously, all columns in an existing DataFrame can be batch converted using DataFrame.astype():

This conversion is likewise done column by column:

In the examples above where we passed dtype='category', we used the default behavior:

Categories are inferred from the data.

Categories are unordered.

To control those behaviors, instead of passing 'category', use an instance of CategoricalDtype.

Similarly, a CategoricalDtype can be used with a DataFrame to ensure that categories are consistent among all columns.

To perform table-wise conversion, where all labels in the entire DataFrame are used as categories for each column, the categories parameter can be determined programmatically by categories = pd.unique(df.to_numpy().ravel()).

If you already have codes and categories, you can use the from_codes() constructor to save the factorize step during normal constructor mode:

To get back to the original Series or NumPy array, use Series.astype(original_dtype) or np.asarray(categorical):

In contrast to R’s factor function, categorical data is not converting input values to strings; categories will end up the same data type as the original values.

In contrast to R’s factor function, there is currently no way to assign/change labels at creation time. Use categories to change the categories after creation time.

A categorical’s type is fully described by

categories: a sequence of unique values and no missing values

This information can be stored in a CategoricalDtype. The categories argument is optional, which implies that the actual categories should be inferred from whatever is present in the data when the pandas.Categorical is created. The categories are assumed to be unordered by default.

A CategoricalDtype can be used in any place pandas expects a dtype. For example pandas.read_csv(), pandas.DataFrame.astype(), or in the Series constructor.

As a convenience, you can use the string 'category' in place of a CategoricalDtype when you want the default behavior of the categories being unordered, and equal to the set values present in the array. In other words, dtype='category' is equivalent to dtype=CategoricalDtype().

Two instances of CategoricalDtype compare equal whenever they have the same categories and order. When comparing two unordered categoricals, the order of the categories is not considered.

All instances of CategoricalDtype compare equal to the string 'category'.

Using describe() on categorical data will produce similar output to a Series or DataFrame of type string.

Categorical data has a categories and a ordered property, which list their possible values and whether the ordering matters or not. These properties are exposed as s.cat.categories and s.cat.ordered. If you don’t manually specify categories and ordering, they are inferred from the passed arguments.

It’s also possible to pass in the categories in a specific order:

New categorical data are not automatically ordered. You must explicitly pass ordered=True to indicate an ordered Categorical.

The result of unique() is not always the same as Series.cat.categories, because Series.unique() has a couple of guarantees, namely that it returns categories in the order of appearance, and it only includes values that are actually present.

Renaming categories is done by using the rename_categories() method:

In contrast to R’s factor, categorical data can have categories of other types than string.

Categories must be unique or a ValueError is raised:

Categories must also not be NaN or a ValueError is raised:

Appending categories can be done by using the add_categories() method:

Removing categories can be done by using the remove_categories() method. Values which are removed are replaced by np.nan.:

Removing unused categories can also be done:

If you want to do remove and add new categories in one step (which has some speed advantage), or simply set the categories to a predefined scale, use set_categories().

Be aware that Categorical.set_categories() cannot know whether some category is omitted intentionally or because it is misspelled or (under Python3) due to a type difference (e.g., NumPy S1 dtype and Python strings). This can result in surprising behaviour!

If categorical data is ordered (s.cat.ordered == True), then the order of the categories has a meaning and certain operations are possible. If the categorical is unordered, .min()/.max() will raise a TypeError.

You can set categorical data to be ordered by using as_ordered() or unordered by using as_unordered(). These will by default return a new object.

Sorting will use the order defined by categories, not any lexical order present on the data type. This is even true for strings and numeric data:

Reordering the categories is possible via the Categorical.reorder_categories() and the Categorical.set_categories() methods. For Categorical.reorder_categories(), all old categories must be included in the new categories and no new categories are allowed. This will necessarily make the sort order the same as the categories order.

Note the difference between assigning new categories and reordering the categories: the first renames categories and therefore the individual values in the Series, but if the first position was sorted last, the renamed value will still be sorted last. Reordering means that the way values are sorted is different afterwards, but not that individual values in the Series are changed.

If the Categorical is not ordered, Series.min() and Series.max() will raise TypeError. Numeric operations like +, -, *, / and operations based on them (e.g. Series.median(), which would need to compute the mean between two values if the length of an array is even) do not work and raise a TypeError.

A categorical dtyped column will participate in a multi-column sort in a similar manner to other columns. The ordering of the categorical is determined by the categories of that column.

Reordering the categories changes a future sort.

Comparing categorical data with other objects is possible in three cases:

Comparing equality (== and !=) to a list-like object (list, Series, array, …) of the same length as the categorical data.

All comparisons (==, !=, >, >=, <, and <=) of categorical data to another categorical Series, when ordered==True and the categories are the same.

All comparisons of a categorical data to a scalar.

All other comparisons, especially “non-equality” comparisons of two categoricals with different categories or a categorical with any list-like object, will raise a TypeError.

Any “non-equality” comparisons of categorical data with a Series, np.array, list or categorical data with different categories or ordering will raise a TypeError because custom categories ordering could be interpreted in two ways: one with taking into account the ordering and one without.

Comparing to a categorical with the same categories and ordering or to a scalar works:

Equality comparisons work with any list-like object of same length and scalars:

This doesn’t work because the categories are not the same:

If you want to do a “non-equality” comparison of a categorical series with a list-like object which is not categorical data, you need to be explicit and convert the categorical data back to the original values:

When you compare two unordered categoricals with the same categories, the order is not considered:

Apart from Series.min(), Series.max() and Series.mode(), the following operations are possible with categorical data:

Series methods like Series.value_counts() will use all categories, even if some categories are not present in the data:

DataFrame methods like DataFrame.sum() also show “unused” categories when observed=False.

Groupby will also show “unused” categories when observed=False:

The optimized pandas data access methods .loc, .iloc, .at, and .iat, work as normal. The only difference is the return type (for getting) and that only values already in categories can be assigned.

If the slicing operation returns either a DataFrame or a column of type Series, the category dtype is preserved.

An example where the category type is not preserved is if you take one single row: the resulting Series is of dtype object:

Returning a single item from categorical data will also return the value, not a categorical of length “1”.

The is in contrast to R’s factor function, where factor(c(1,2,3))[1] returns a single value factor.

To get a single value Series of type category, you pass in a list with a single value:

The accessors .dt and .str will work if the s.cat.categories are of an appropriate type:

The returned Series (or DataFrame) is of the same type as if you used the .str.<method> / .dt.<method> on a Series of that type (and not of type category!).

That means, that the returned values from methods and properties on the accessors of a Series and the returned values from methods and properties on the accessors of this Series transformed to one of type category will be equal:

The work is done on the categories and then a new Series is constructed. This has some performance implication if you have a Series of type string, where lots of elements are repeated (i.e. the number of unique elements in the Series is a lot smaller than the length of the Series). In this case it can be faster to convert the original Series to one of type category and use .str.<method> or .dt.<property> on that.

Setting values in a categorical column (or Series) works as long as the value is included in the categories:

Setting values by assigning categorical data will also check that the categories match:

Assigning a Categorical to parts of a column of other types will use the values:

By default, combining Series or DataFrames which contain the same categories results in category dtype, otherwise results will depend on the dtype of the underlying categories. Merges that result in non-categorical dtypes will likely have higher memory usage. Use .astype or union_categoricals to ensure category results.

The following table summarizes the results of merging Categoricals:

object (dtype is inferred)

float (dtype is inferred)

If you want to combine categoricals that do not necessarily have the same categories, the union_categoricals() function will combine a list-like of categoricals. The new categories will be the union of the categories being combined.

By default, the resulting categories will be ordered as they appear in the data. If you want the categories to be lexsorted, use sort_categories=True argument.

union_categoricals also works with the “easy” case of combining two categoricals of the same categories and order information (e.g. what you could also append for).

The below raises TypeError because the categories are ordered and not identical.

Ordered categoricals with different categories or orderings can be combined by using the ignore_ordered=True argument.

union_categoricals() also works with a CategoricalIndex, or Series containing categorical data, but note that the resulting array will always be a plain Categorical:

union_categoricals may recode the integer codes for categories when combining categoricals. This is likely what you want, but if you are relying on the exact numbering of the categories, be aware.

You can write data that contains category dtypes to a HDFStore. See here for an example and caveats.

It is also possible to write data to and reading data from Stata format files. See here for an example and caveats.

Writing to a CSV file will convert the data, effectively removing any information about the categorical (categories and ordering). So if you read back the CSV file you have to convert the relevant columns back to category and assign the right categories and categories ordering.

The same holds for writing to a SQL database with to_sql.

pandas primarily uses the value np.nan to represent missing data. It is by default not included in computations. See the Missing Data section.

Missing values should not be included in the Categorical’s categories, only in the values. Instead, it is understood that NaN is different, and is always a possibility. When working with the Categorical’s codes, missing values will always have a code of -1.

Methods for working with missing data, e.g. isna(), fillna(), dropna(), all work normally:

The following differences to R’s factor functions can be observed:

R’s levels are named categories.

R’s levels are always of type string, while categories in pandas can be of any dtype.

It’s not possible to specify labels at creation time. Use s.cat.rename_categories(new_labels) afterwards.

In contrast to R’s factor function, using categorical data as the sole input to create a new categorical series will not remove unused categories but create a new categorical series which is equal to the passed in one!

R allows for missing values to be included in its levels (pandas’ categories). pandas does not allow NaN categories, but missing values can still be in the values.

The memory usage of a Categorical is proportional to the number of categories plus the length of the data. In contrast, an object dtype is a constant times the length of the data.

If the number of categories approaches the length of the data, the Categorical will use nearly the same or more memory than an equivalent object dtype representation.

Currently, categorical data and the underlying Categorical is implemented as a Python object and not as a low-level NumPy array dtype. This leads to some problems.

NumPy itself doesn’t know about the new dtype:

Dtype comparisons work:

To check if a Series contains Categorical data, use hasattr(s, 'cat'):

Using NumPy functions on a Series of type category should not work as Categoricals are not numeric data (even in the case that .categories is numeric).

If such a function works, please file a bug at pandas-dev/pandas!

pandas currently does not preserve the dtype in apply functions: If you apply along rows you get a Series of object dtype (same as getting a row -> getting one element will return a basic type) and applying along columns will also convert to object. NaN values are unaffected. You can use fillna to handle missing values before applying a function.

CategoricalIndex is a type of index that is useful for supporting indexing with duplicates. This is a container around a Categorical and allows efficient indexing and storage of an index with a large number of duplicated elements. See the advanced indexing docs for a more detailed explanation.

Setting the index will create a CategoricalIndex:

Constructing a Series from a Categorical will not copy the input Categorical. This means that changes to the Series will in most cases change the original Categorical:

Use copy=True to prevent such a behaviour or simply don’t reuse Categoricals:

This also happens in some cases when you supply a NumPy array instead of a Categorical: using an int array (e.g. np.array([1,2,3,4])) will exhibit the same behavior, while using a string array (e.g. np.array(["a","b","c","a"])) will not.

**Examples:**

Example 1 (typescript):
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

Example 2 (json):
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

Example 3 (typescript):
```typescript
In [6]: df = pd.DataFrame({"value": np.random.randint(0, 100, 20)})

In [7]: labels = ["{0} - {1}".format(i, i + 9) for i in range(0, 100, 10)]

In [8]: df["group"] = pd.cut(df.value, range(0, 105, 10), right=False, labels=labels)

In [9]: df.head(10)
Out[9]: 
   value    group
0     65  60 - 69
1     49  40 - 49
2     56  50 - 59
3     43  40 - 49
4     43  40 - 49
5     91  90 - 99
6     32  30 - 39
7     87  80 - 89
8     36  30 - 39
9      8    0 - 9
```

Example 4 (json):
```json
In [10]: raw_cat = pd.Categorical(
   ....:     ["a", "b", "c", "a"], categories=["b", "c", "d"], ordered=False
   ....: )
   ....: 

In [11]: s = pd.Series(raw_cat)

In [12]: s
Out[12]: 
0    NaN
1      b
2      c
3    NaN
dtype: category
Categories (3, object): ['b', 'c', 'd']

In [13]: df = pd.DataFrame({"A": ["a", "b", "c", "a"]})

In [14]: df["B"] = raw_cat

In [15]: df
Out[15]: 
   A    B
0  a  NaN
1  b    b
2  c    c
3  a  NaN
```

---

## Options and settings#

**URL:** https://pandas.pydata.org/docs/user_guide/options.html

**Contents:**
- Options and settings#
- Overview#
- Available options#
- Getting and setting options#
- Setting startup options in Python/IPython environment#
- Frequently used options#
- Number formatting#
- Unicode formatting#
- Table schema display#

pandas has an options API configure and customize global behavior related to DataFrame display, data behavior and more.

Options have a full “dotted-style”, case-insensitive name (e.g. display.max_rows). You can get/set options directly as attributes of the top-level options attribute:

The API is composed of 5 relevant functions, available directly from the pandas namespace:

get_option() / set_option() - get/set the value of a single option.

reset_option() - reset one or more options to their default value.

describe_option() - print the descriptions of one or more options.

option_context() - execute a codeblock with a set of options that revert to prior settings after execution.

Developers can check out pandas/core/config_init.py for more information.

All of the functions above accept a regexp pattern (re.search style) as an argument, to match an unambiguous substring:

The following will not work because it matches multiple option names, e.g. display.max_colwidth, display.max_rows, display.max_columns:

Using this form of shorthand may cause your code to break if new options with similar names are added in future versions.

You can get a list of available options and their descriptions with describe_option(). When called with no argument describe_option() will print out the descriptions for all available options.

As described above, get_option() and set_option() are available from the pandas namespace. To change an option, call set_option('option regex', new_value).

The option 'mode.sim_interactive' is mostly used for debugging purposes.

You can use reset_option() to revert to a setting’s default value

It’s also possible to reset multiple options at once (using a regex):

option_context() context manager has been exposed through the top-level API, allowing you to execute code with given option values. Option values are restored automatically when you exit the with block:

Using startup scripts for the Python/IPython environment to import pandas and set options makes working with pandas more efficient. To do this, create a .py or .ipy script in the startup directory of the desired profile. An example where the startup folder is in a default IPython profile can be found at:

More information can be found in the IPython documentation. An example startup script for pandas is displayed below:

The following is a demonstrates the more frequently used display options.

display.max_rows and display.max_columns sets the maximum number of rows and columns displayed when a frame is pretty-printed. Truncated lines are replaced by an ellipsis.

Once the display.max_rows is exceeded, the display.min_rows options determines how many rows are shown in the truncated repr.

display.expand_frame_repr allows for the representation of a DataFrame to stretch across pages, wrapped over the all the columns.

display.large_repr displays a DataFrame that exceed max_columns or max_rows as a truncated frame or summary.

display.max_colwidth sets the maximum width of columns. Cells of this length or longer will be truncated with an ellipsis.

display.max_info_columns sets a threshold for the number of columns displayed when calling info().

display.max_info_rows: info() will usually show null-counts for each column. For a large DataFrame, this can be quite slow. max_info_rows and max_info_cols limit this null check to the specified rows and columns respectively. The info() keyword argument show_counts=True will override this.

display.precision sets the output display precision in terms of decimal places.

display.chop_threshold sets the rounding threshold to zero when displaying a Series or DataFrame. This setting does not change the precision at which the number is stored.

display.colheader_justify controls the justification of the headers. The options are 'right', and 'left'.

pandas also allows you to set how numbers are displayed in the console. This option is not set through the set_options API.

Use the set_eng_float_format function to alter the floating-point formatting of pandas objects to produce a particular format.

Use round() to specifically control rounding of an individual DataFrame

Enabling this option will affect the performance for printing of DataFrame and Series (about 2 times slower). Use only when it is actually required.

Some East Asian countries use Unicode characters whose width corresponds to two Latin characters. If a DataFrame or Series contains these characters, the default output mode may not align them properly.

Enabling display.unicode.east_asian_width allows pandas to check each character’s “East Asian Width” property. These characters can be aligned properly by setting this option to True. However, this will result in longer render times than the standard len function.

In addition, Unicode characters whose width is “ambiguous” can either be 1 or 2 characters wide depending on the terminal setting or encoding. The option display.unicode.ambiguous_as_wide can be used to handle the ambiguity.

By default, an “ambiguous” character’s width, such as “¡” (inverted exclamation) in the example below, is taken to be 1.

Enabling display.unicode.ambiguous_as_wide makes pandas interpret these characters’ widths to be 2. (Note that this option will only be effective when display.unicode.east_asian_width is enabled.)

However, setting this option incorrectly for your terminal will cause these characters to be aligned incorrectly:

DataFrame and Series will publish a Table Schema representation by default. This can be enabled globally with the display.html.table_schema option:

Only 'display.max_rows' are serialized and published.

**Examples:**

Example 1 (typescript):
```typescript
In [1]: import pandas as pd

In [2]: pd.options.display.max_rows
Out[2]: 15

In [3]: pd.options.display.max_rows = 999

In [4]: pd.options.display.max_rows
Out[4]: 999
```

Example 2 (json):
```json
In [5]: pd.get_option("display.chop_threshold")

In [6]: pd.set_option("display.chop_threshold", 2)

In [7]: pd.get_option("display.chop_threshold")
Out[7]: 2

In [8]: pd.set_option("chop", 4)

In [9]: pd.get_option("display.chop_threshold")
Out[9]: 4
```

Example 3 (yaml):
```yaml
In [10]: pd.get_option("max")
---------------------------------------------------------------------------
OptionError                               Traceback (most recent call last)
Cell In[10], line 1
----> 1 pd.get_option("max")

File ~/work/pandas/pandas/pandas/_config/config.py:274, in CallableDynamicDoc.__call__(self, *args, **kwds)
    273 def __call__(self, *args, **kwds) -> T:
--> 274     return self.__func__(*args, **kwds)

File ~/work/pandas/pandas/pandas/_config/config.py:146, in _get_option(pat, silent)
    145 def _get_option(pat: str, silent: bool = False) -> Any:
--> 146     key = _get_single_key(pat, silent)
    148     # walk the nested dict
    149     root, k = _get_root(key)

File ~/work/pandas/pandas/pandas/_config/config.py:134, in _get_single_key(pat, silent)
    132     raise OptionError(f"No such keys(s): {repr(pat)}")
    133 if len(keys) > 1:
--> 134     raise OptionError("Pattern matched multiple keys")
    135 key = keys[0]
    137 if not silent:

OptionError: Pattern matched multiple keys
```

Example 4 (julia):
```julia
In [11]: pd.describe_option()
compute.use_bottleneck : bool
    Use the bottleneck library to accelerate if it is installed,
    the default is True
    Valid values: False,True
    [default: True] [currently: True]
compute.use_numba : bool
    Use the numba engine option for select operations if it is installed,
    the default is False
    Valid values: False,True
    [default: False] [currently: False]
compute.use_numexpr : bool
    Use the numexpr library to accelerate computation if it is installed,
    the default is True
    Valid values: False,True
    [default: True] [currently: True]
display.chop_threshold : float or None
    if set to a float value, all float values smaller than the given threshold
    will be displayed as exactly 0 by repr and friends.
    [default: None] [currently: None]
display.colheader_justify : 'left'/'right'
    Controls the justification of column headers. used by DataFrameFormatter.
    [default: right] [currently: right]
display.date_dayfirst : boolean
    When True, prints and parses dates with the day first, eg 20/01/2005
    [default: False] [currently: False]
display.date_yearfirst : boolean
    When True, prints and parses dates with the year first, eg 2005/01/20
    [default: False] [currently: False]
display.encoding : str/unicode
    Defaults to the detected encoding of the console.
    Specifies the encoding to be used for strings returned by to_string,
    these are generally strings meant to be displayed on the console.
    [default: utf-8] [currently: utf8]
display.expand_frame_repr : boolean
    Whether to print out the full DataFrame repr for wide DataFrames across
    multiple lines, `max_columns` is still respected, but the output will
    wrap-around across multiple "pages" if its width exceeds `display.width`.
    [default: True] [currently: True]
display.float_format : callable
    The callable should accept a floating point number and return
    a string with the desired format of the number. This is used
    in some places like SeriesFormatter.
    See formats.format.EngFormatter for an example.
    [default: None] [currently: None]
display.html.border : int
    A ``border=value`` attribute is inserted in the ``<table>`` tag
    for the DataFrame HTML repr.
    [default: 1] [currently: 1]
display.html.table_schema : boolean
    Whether to publish a Table Schema representation for frontends
    that support it.
    (default: False)
    [default: False] [currently: False]
display.html.use_mathjax : boolean
    When True, Jupyter notebook will process table contents using MathJax,
    rendering mathematical expressions enclosed by the dollar symbol.
    (default: True)
    [default: True] [currently: True]
display.large_repr : 'truncate'/'info'
    For DataFrames exceeding max_rows/max_cols, the repr (and HTML repr) can
    show a truncated table, or switch to the view from
    df.info() (the behaviour in earlier versions of pandas).
    [default: truncate] [currently: truncate]
display.max_categories : int
    This sets the maximum number of categories pandas should output when
    printing out a `Categorical` or a Series of dtype "category".
    [default: 8] [currently: 8]
display.max_columns : int
    If max_cols is exceeded, switch to truncate view. Depending on
    `large_repr`, objects are either centrally truncated or printed as
    a summary view. 'None' value means unlimited.

    In case python/IPython is running in a terminal and `large_repr`
    equals 'truncate' this can be set to 0 or None and pandas will auto-detect
    the width of the terminal and print a truncated object which fits
    the screen width. The IPython notebook, IPython qtconsole, or IDLE
    do not run in a terminal and hence it is not possible to do
    correct auto-detection and defaults to 20.
    [default: 0] [currently: 0]
display.max_colwidth : int or None
    The maximum width in characters of a column in the repr of
    a pandas data structure. When the column overflows, a "..."
    placeholder is embedded in the output. A 'None' value means unlimited.
    [default: 50] [currently: 50]
display.max_dir_items : int
    The number of items that will be added to `dir(...)`. 'None' value means
    unlimited. Because dir is cached, changing this option will not immediately
    affect already existing dataframes until a column is deleted or added.

    This is for instance used to suggest columns from a dataframe to tab
    completion.
    [default: 100] [currently: 100]
display.max_info_columns : int
    max_info_columns is used in DataFrame.info method to decide if
    per column information will be printed.
    [default: 100] [currently: 100]
display.max_info_rows : int
    df.info() will usually show null-counts for each column.
    For large frames this can be quite slow. max_info_rows and max_info_cols
    limit this null check only to frames with smaller dimensions than
    specified.
    [default: 1690785] [currently: 1690785]
display.max_rows : int
    If max_rows is exceeded, switch to truncate view. Depending on
    `large_repr`, objects are either centrally truncated or printed as
    a summary view. 'None' value means unlimited.

    In case python/IPython is running in a terminal and `large_repr`
    equals 'truncate' this can be set to 0 and pandas will auto-detect
    the height of the terminal and print a truncated object which fits
    the screen height. The IPython notebook, IPython qtconsole, or
    IDLE do not run in a terminal and hence it is not possible to do
    correct auto-detection.
    [default: 60] [currently: 60]
display.max_seq_items : int or None
    When pretty-printing a long sequence, no more then `max_seq_items`
    will be printed. If items are omitted, they will be denoted by the
    addition of "..." to the resulting string.

    If set to None, the number of items to be printed is unlimited.
    [default: 100] [currently: 100]
display.memory_usage : bool, string or None
    This specifies if the memory usage of a DataFrame should be displayed when
    df.info() is called. Valid values True,False,'deep'
    [default: True] [currently: True]
display.min_rows : int
    The numbers of rows to show in a truncated view (when `max_rows` is
    exceeded). Ignored when `max_rows` is set to None or 0. When set to
    None, follows the value of `max_rows`.
    [default: 10] [currently: 10]
display.multi_sparse : boolean
    "sparsify" MultiIndex display (don't display repeated
    elements in outer levels within groups)
    [default: True] [currently: True]
display.notebook_repr_html : boolean
    When True, IPython notebook will use html representation for
    pandas objects (if it is available).
    [default: True] [currently: True]
display.pprint_nest_depth : int
    Controls the number of nested levels to process when pretty-printing
    [default: 3] [currently: 3]
display.precision : int
    Floating point output precision in terms of number of places after the
    decimal, for regular formatting as well as scientific notation. Similar
    to ``precision`` in :meth:`numpy.set_printoptions`.
    [default: 6] [currently: 6]
display.show_dimensions : boolean or 'truncate'
    Whether to print out dimensions at the end of DataFrame repr.
    If 'truncate' is specified, only print out the dimensions if the
    frame is truncated (e.g. not display all rows and/or columns)
    [default: truncate] [currently: truncate]
display.unicode.ambiguous_as_wide : boolean
    Whether to use the Unicode East Asian Width to calculate the display text
    width.
    Enabling this may affect to the performance (default: False)
    [default: False] [currently: False]
display.unicode.east_asian_width : boolean
    Whether to use the Unicode East Asian Width to calculate the display text
    width.
    Enabling this may affect to the performance (default: False)
    [default: False] [currently: False]
display.width : int
    Width of the display in characters. In case python/IPython is running in
    a terminal this can be set to None and pandas will correctly auto-detect
    the width.
    Note that the IPython notebook, IPython qtconsole, or IDLE do not run in a
    terminal and hence it is not possible to correctly detect the width.
    [default: 80] [currently: 80]
future.infer_string Whether to infer sequence of str objects as pyarrow string dtype, which will be the default in pandas 3.0 (at which point this option will be deprecated).
    [default: False] [currently: False]
future.no_silent_downcasting Whether to opt-in to the future behavior which will *not* silently downcast results from Series and DataFrame `where`, `mask`, and `clip` methods. Silent downcasting will be removed in pandas 3.0 (at which point this option will be deprecated).
    [default: False] [currently: False]
io.excel.ods.reader : string
    The default Excel reader engine for 'ods' files. Available options:
    auto, odf, calamine.
    [default: auto] [currently: auto]
io.excel.ods.writer : string
    The default Excel writer engine for 'ods' files. Available options:
    auto, odf.
    [default: auto] [currently: auto]
io.excel.xls.reader : string
    The default Excel reader engine for 'xls' files. Available options:
    auto, xlrd, calamine.
    [default: auto] [currently: auto]
io.excel.xlsb.reader : string
    The default Excel reader engine for 'xlsb' files. Available options:
    auto, pyxlsb, calamine.
    [default: auto] [currently: auto]
io.excel.xlsm.reader : string
    The default Excel reader engine for 'xlsm' files. Available options:
    auto, xlrd, openpyxl, calamine.
    [default: auto] [currently: auto]
io.excel.xlsm.writer : string
    The default Excel writer engine for 'xlsm' files. Available options:
    auto, openpyxl.
    [default: auto] [currently: auto]
io.excel.xlsx.reader : string
    The default Excel reader engine for 'xlsx' files. Available options:
    auto, xlrd, openpyxl, calamine.
    [default: auto] [currently: auto]
io.excel.xlsx.writer : string
    The default Excel writer engine for 'xlsx' files. Available options:
    auto, openpyxl, xlsxwriter.
    [default: auto] [currently: auto]
io.hdf.default_format : format
    default format writing format, if None, then
    put will default to 'fixed' and append will default to 'table'
    [default: None] [currently: None]
io.hdf.dropna_table : boolean
    drop ALL nan rows when appending to a table
    [default: False] [currently: False]
io.parquet.engine : string
    The default parquet reader/writer engine. Available options:
    'auto', 'pyarrow', 'fastparquet', the default is 'auto'
    [default: auto] [currently: auto]
io.sql.engine : string
    The default sql reader/writer engine. Available options:
    'auto', 'sqlalchemy', the default is 'auto'
    [default: auto] [currently: auto]
mode.chained_assignment : string
    Raise an exception, warn, or no action if trying to use chained assignment,
    The default is warn
    [default: warn] [currently: warn]
mode.copy_on_write : bool
    Use new copy-view behaviour using Copy-on-Write. Defaults to False,
    unless overridden by the 'PANDAS_COPY_ON_WRITE' environment variable
    (if set to "1" for True, needs to be set before pandas is imported).
    [default: False] [currently: False]
mode.data_manager : string
    Internal data manager type; can be "block" or "array". Defaults to "block",
    unless overridden by the 'PANDAS_DATA_MANAGER' environment variable (needs
    to be set before pandas is imported).
    [default: block] [currently: block]
    (Deprecated, use `` instead.)
mode.sim_interactive : boolean
    Whether to simulate interactive mode for purposes of testing
    [default: False] [currently: False]
mode.string_storage : string
    The default storage for StringDtype.
    [default: auto] [currently: auto]
mode.use_inf_as_na : boolean
    True means treat None, NaN, INF, -INF as NA (old way),
    False means None and NaN are null, but INF, -INF are not NA
    (new way).

    This option is deprecated in pandas 2.1.0 and will be removed in 3.0.
    [default: False] [currently: False]
    (Deprecated, use `` instead.)
plotting.backend : str
    The plotting backend to use. The default value is "matplotlib", the
    backend provided with pandas. Other backends can be specified by
    providing the name of the module that implements the backend.
    [default: matplotlib] [currently: matplotlib]
plotting.matplotlib.register_converters : bool or 'auto'.
    Whether to register converters with matplotlib's units registry for
    dates, times, datetimes, and Periods. Toggling to False will remove
    the converters, restoring any converters that pandas overwrote.
    [default: auto] [currently: auto]
styler.format.decimal : str
    The character representation for the decimal separator for floats and complex.
    [default: .] [currently: .]
styler.format.escape : str, optional
    Whether to escape certain characters according to the given context; html or latex.
    [default: None] [currently: None]
styler.format.formatter : str, callable, dict, optional
    A formatter object to be used as default within ``Styler.format``.
    [default: None] [currently: None]
styler.format.na_rep : str, optional
    The string representation for values identified as missing.
    [default: None] [currently: None]
styler.format.precision : int
    The precision for floats and complex numbers.
    [default: 6] [currently: 6]
styler.format.thousands : str, optional
    The character representation for thousands separator for floats, int and complex.
    [default: None] [currently: None]
styler.html.mathjax : bool
    If False will render special CSS classes to table attributes that indicate Mathjax
    will not be used in Jupyter Notebook.
    [default: True] [currently: True]
styler.latex.environment : str
    The environment to replace ``\begin{table}``. If "longtable" is used results
    in a specific longtable environment format.
    [default: None] [currently: None]
styler.latex.hrules : bool
    Whether to add horizontal rules on top and bottom and below the headers.
    [default: False] [currently: False]
styler.latex.multicol_align : {"r", "c", "l", "naive-l", "naive-r"}
    The specifier for horizontal alignment of sparsified LaTeX multicolumns. Pipe
    decorators can also be added to non-naive values to draw vertical
    rules, e.g. "\|r" will draw a rule on the left side of right aligned merged cells.
    [default: r] [currently: r]
styler.latex.multirow_align : {"c", "t", "b"}
    The specifier for vertical alignment of sparsified LaTeX multirows.
    [default: c] [currently: c]
styler.render.encoding : str
    The encoding used for output HTML and LaTeX files.
    [default: utf-8] [currently: utf-8]
styler.render.max_columns : int, optional
    The maximum number of columns that will be rendered. May still be reduced to
    satisfy ``max_elements``, which takes precedence.
    [default: None] [currently: None]
styler.render.max_elements : int
    The maximum number of data-cell (<td>) elements that will be rendered before
    trimming will occur over columns, rows or both if needed.
    [default: 262144] [currently: 262144]
styler.render.max_rows : int, optional
    The maximum number of rows that will be rendered. May still be reduced to
    satisfy ``max_elements``, which takes precedence.
    [default: None] [currently: None]
styler.render.repr : str
    Determine which output to use in Jupyter Notebook in {"html", "latex"}.
    [default: html] [currently: html]
styler.sparse.columns : bool
    Whether to sparsify the display of hierarchical columns. Setting to False will
    display each explicit level element in a hierarchical key for each column.
    [default: True] [currently: True]
styler.sparse.index : bool
    Whether to sparsify the display of a hierarchical index. Setting to False will
    display each explicit level element in a hierarchical key for each row.
    [default: True] [currently: True]
```

---

## 

**URL:** https://pandas.pydata.org/docs/_sources/user_guide/merging.rst.txt

---

## Table Visualization#

**URL:** https://pandas.pydata.org/docs/user_guide/style.html

**Contents:**
- Table Visualization#
- Styler Object and Customising the Display#
- Formatting the Display#
  - Formatting Values#
  - Hiding Data#
  - Concatenating DataFrame Outputs#
- Styler Object and HTML#
- Methods to Add Styles#
- Table Styles#
- Setting Classes and Linking to External CSS#

This section demonstrates visualization of tabular data using the Styler class. For information on visualization with charting please see Chart Visualization. This document is written as a Jupyter Notebook, and can be viewed or downloaded here.

Styling and output display customisation should be performed after the data in a DataFrame has been processed. The Styler is not dynamically updated if further changes to the DataFrame are made. The DataFrame.style attribute is a property that returns a Styler object. It has a _repr_html_ method defined on it so it is rendered automatically in Jupyter Notebook.

The Styler, which can be used for large data but is primarily designed for small data, currently has the ability to output to these formats:

String (and CSV by extension)

(JSON is not currently available)

The first three of these have display customisation methods designed to format and customise the output. These include:

Formatting values, the index and columns headers, using .format() and .format_index(),

Renaming the index or column header labels, using .relabel_index()

Hiding certain columns, the index and/or column headers, or index names, using .hide()

Concatenating similar DataFrames, using .concat()

The Styler distinguishes the display value from the actual value, in both data values and index or columns headers. To control the display value, the text is printed in each cell as a string, and we can use the .format() and .format_index() methods to manipulate this according to a format spec string or a callable that takes a single value and returns a string. It is possible to define this for the whole table, or index, or for individual columns, or MultiIndex levels. We can also overwrite index names.

Additionally, the format function has a precision argument to specifically help format floats, as well as decimal and thousands separators to support other locales, an na_rep argument to display missing data, and an escape and hyperlinks arguments to help displaying safe-HTML or safe-LaTeX. The default formatter is configured to adopt pandas’ global options such as styler.format.precision option, controllable using with pd.option_context('format.precision', 2):

Using Styler to manipulate the display is a useful feature because maintaining the indexing and data values for other purposes gives greater control. You do not have to overwrite your DataFrame to display it how you like. Here is a more comprehensive example of using the formatting functions whilst still relying on the underlying data for indexing and calculations.

The index and column headers can be completely hidden, as well subselecting rows or columns that one wishes to exclude. Both these options are performed using the same methods.

The index can be hidden from rendering by calling .hide() without any arguments, which might be useful if your index is integer based. Similarly column headers can be hidden by calling .hide(axis=”columns”) without any further arguments.

Specific rows or columns can be hidden from rendering by calling the same .hide() method and passing in a row/column label, a list-like or a slice of row/column labels to for the subset argument.

Hiding does not change the integer arrangement of CSS classes, e.g. hiding the first two columns of a DataFrame means the column class indexing will still start at col2, since col0 and col1 are simply ignored.

To invert the function to a show functionality it is best practice to compose a list of hidden items.

Two or more Stylers can be concatenated together provided they share the same columns. This is very useful for showing summary statistics for a DataFrame, and is often used in combination with DataFrame.agg.

Since the objects concatenated are Stylers they can independently be styled as will be shown below and their concatenation preserves those styles.

The Styler was originally constructed to support the wide array of HTML formatting options. Its HTML output creates an HTML <table> and leverages CSS styling language to manipulate many parameters including colors, fonts, borders, background, etc. See here for more information on styling HTML tables. This allows a lot of flexibility out of the box, and even enables web developers to integrate DataFrames into their exiting user interface designs.

Below we demonstrate the default output, which looks very similar to the standard DataFrame HTML representation. But the HTML here has already attached some CSS classes to each cell, even if we haven’t yet created any styles. We can view these by calling the .to_html() method, which returns the raw HTML as string, which is useful for further processing or adding to a file - read on in More about CSS and HTML. This section will also provide a walkthrough for how to convert this default output to represent a DataFrame output that is more communicative. For example how we can build s:

The first step we have taken is the create the Styler object from the DataFrame and then select the range of interest by hiding unwanted columns with .hide().

There are 3 primary methods of adding custom CSS styles to Styler:

Using .set_table_styles() to control broader areas of the table with specified internal CSS. Although table styles allow the flexibility to add CSS selectors and properties controlling all individual parts of the table, they are unwieldy for individual cell specifications. Also, note that table styles cannot be exported to Excel.

Using .set_td_classes() to directly link either external CSS classes to your data cells or link the internal CSS classes created by .set_table_styles(). See here. These cannot be used on column header rows or indexes, and also won’t export to Excel.

Using the .apply() and .map() functions to add direct internal CSS to specific data cells. See here. As of v1.4.0 there are also methods that work directly on column header rows or indexes; .apply_index() and .map_index(). Note that only these methods add styles that will export to Excel. These methods work in a similar way to DataFrame.apply() and DataFrame.map().

Table styles are flexible enough to control all individual parts of the table, including column headers and indexes. However, they can be unwieldy to type for individual data cells or for any kind of conditional formatting, so we recommend that table styles are used for broad styling, such as entire rows or columns at a time.

Table styles are also used to control features which can apply to the whole table at once such as creating a generic hover functionality. The :hover pseudo-selector, as well as other pseudo-selectors, can only be used this way.

To replicate the normal format of CSS selectors and properties (attribute value pairs), e.g.

the necessary format to pass styles to .set_table_styles() is as a list of dicts, each with a CSS-selector tag and CSS-properties. Properties can either be a list of 2-tuples, or a regular CSS-string, for example:

Next we just add a couple more styling artifacts targeting specific parts of the table. Be careful here, since we are chaining methods we need to explicitly instruct the method not to overwrite the existing styles.

As a convenience method (since version 1.2.0) we can also pass a dict to .set_table_styles() which contains row or column keys. Behind the scenes Styler just indexes the keys and adds relevant .col<m> or .row<n> classes as necessary to the given CSS selectors.

If you have designed a website then it is likely you will already have an external CSS file that controls the styling of table and cell objects within it. You may want to use these native files rather than duplicate all the CSS in python (and duplicate any maintenance work).

It is very easy to add a class to the main <table> using .set_table_attributes(). This method can also attach inline styles - read more in CSS Hierarchies.

The .set_td_classes() method accepts a DataFrame with matching indices and columns to the underlying Styler’s DataFrame. That DataFrame will contain strings as css-classes to add to individual data cells: the <td> elements of the <table>. Rather than use external CSS we will create our classes internally and add them to table style. We will save adding the borders until the section on tooltips.

We use the following methods to pass your style functions. Both of those methods take a function (and some other keyword arguments) and apply it to the DataFrame in a certain way, rendering CSS styles.

.map() (elementwise): accepts a function that takes a single value and returns a string with the CSS attribute-value pair.

.apply() (column-/row-/table-wise): accepts a function that takes a Series or DataFrame and returns a Series, DataFrame, or numpy array with an identical shape where each element is a string with a CSS attribute-value pair. This method passes each column or row of your DataFrame one-at-a-time or the entire table at once, depending on the axis keyword argument. For columnwise use axis=0, rowwise use axis=1, and for the entire table at once use axis=None.

This method is powerful for applying multiple, complex logic to data cells. We create a new DataFrame to demonstrate this.

For example we can build a function that colors text if it is negative, and chain this with a function that partially fades cells of negligible value. Since this looks at each element in turn we use map.

We can also build a function that highlights the maximum value across rows, cols, and the DataFrame all at once. In this case we use apply. Below we highlight the maximum in a column.

We can use the same function across the different axes, highlighting here the DataFrame maximum in purple, and row maximums in pink.

This last example shows how some styles have been overwritten by others. In general the most recent style applied is active but you can read more in the section on CSS hierarchies. You can also apply these styles to more granular parts of the DataFrame - read more in section on subset slicing.

It is possible to replicate some of this functionality using just classes but it can be more cumbersome. See item 3) of Optimization

Debugging Tip: If you’re having trouble writing your style function, try just passing it into DataFrame.apply. Internally, Styler.apply uses DataFrame.apply so the result should be the same, and with DataFrame.apply you will be able to inspect the CSS string output of your intended function in each cell.

Similar application is achieved for headers by using:

.map_index() (elementwise): accepts a function that takes a single value and returns a string with the CSS attribute-value pair.

.apply_index() (level-wise): accepts a function that takes a Series and returns a Series, or numpy array with an identical shape where each element is a string with a CSS attribute-value pair. This method passes each level of your Index one-at-a-time. To style the index use axis=0 and to style the column headers use axis=1.

You can select a level of a MultiIndex but currently no similar subset application is available for these methods.

Table captions can be added with the .set_caption() method. You can use table styles to control the CSS relevant to the caption.

Adding tooltips (since version 1.3.0) can be done using the .set_tooltips() method in the same way you can add CSS classes to data cells by providing a string based DataFrame with intersecting indices and columns. You don’t have to specify a css_class name or any css props for the tooltips, since there are standard defaults, but the option is there if you want more visual control.

The only thing left to do for our table is to add the highlighting borders to draw the audience attention to the tooltips. We will create internal CSS classes as before using table styles. Setting classes always overwrites so we need to make sure we add the previous classes.

The examples we have shown so far for the Styler.apply and Styler.map functions have not demonstrated the use of the subset argument. This is a useful argument which permits a lot of flexibility: it allows you to apply styles to specific rows or columns, without having to code that logic into your style function.

The value passed to subset behaves similar to slicing a DataFrame;

A scalar is treated as a column label

A list (or Series or NumPy array) is treated as multiple column labels

A tuple is treated as (row_indexer, column_indexer)

Consider using pd.IndexSlice to construct the tuple for the last one. We will create a MultiIndexed DataFrame to demonstrate the functionality.

We will use subset to highlight the maximum in the third and fourth columns with red text. We will highlight the subset sliced region in yellow.

If combined with the IndexSlice as suggested then it can index across both dimensions with greater flexibility.

This also provides the flexibility to sub select rows when used with the axis=1.

There is also scope to provide conditional filtering.

Suppose we want to highlight the maximum across columns 2 and 4 only in the case that the sum of columns 1 and 3 is less than -2.0 (essentially excluding rows (:,'r2')).

Only label-based slicing is supported right now, not positional, and not callables.

If your style function uses a subset or axis keyword argument, consider wrapping your function in a functools.partial, partialing out that keyword.

Generally, for smaller tables and most cases, the rendered HTML does not need to be optimized, and we don’t really recommend it. There are two cases where it is worth considering:

If you are rendering and styling a very large HTML table, certain browsers have performance issues.

If you are using Styler to dynamically create part of online user interfaces and want to improve network performance.

Here we recommend the following steps to implement:

Ignore the uuid and set cell_ids to False. This will prevent unnecessary HTML.

Use table styles where possible (e.g. for all cells or rows or columns at a time) since the CSS is nearly always more efficient than other formats.

For large DataFrames where the same style is applied to many cells it can be more efficient to declare the styles as classes and then apply those classes to data cells, rather than directly applying styles to cells. It is, however, probably still easier to use the Styler function api when you are not concerned about optimization.

Tooltips require cell_ids to work and they generate extra HTML elements for every data cell.

You can remove unnecessary HTML, or shorten the default class names by replacing the default css dict. You can read a little more about CSS below.

Some styling functions are common enough that we’ve “built them in” to the Styler, so you don’t have to write them and apply them yourself. The current list of such functions is:

.highlight_null: for use with identifying missing data.

.highlight_min and .highlight_max: for use with identifying extremeties in data.

.highlight_between and .highlight_quantile: for use with identifying classes within data.

.background_gradient: a flexible method for highlighting cells based on their, or other, values on a numeric scale.

.text_gradient: similar method for highlighting text based on their, or other, values on a numeric scale.

.bar: to display mini-charts within cell backgrounds.

The individual documentation on each function often gives more examples of their arguments.

This method accepts ranges as float, or NumPy arrays or Series provided the indexes match.

Useful for detecting the highest or lowest percentile values

You can create “heatmaps” with the background_gradient and text_gradient methods. These require matplotlib, and we’ll use Seaborn to get a nice colormap.

.background_gradient and .text_gradient have a number of keyword arguments to customise the gradients and colors. See the documentation.

Use Styler.set_properties when the style doesn’t actually depend on the values. This is just a simple wrapper for .map where the function returns the same properties for all cells.

You can include “bar charts” in your DataFrame.

Additional keyword arguments give more control on centering and positioning, and you can pass a list of [color_negative, color_positive] to highlight lower and higher values or a matplotlib colormap.

To showcase an example here’s how you can change the above with the new align option, combined with setting vmin and vmax limits, the width of the figure, and underlying css props of cells, leaving space to display the text and the bars. We also use text_gradient to color the text the same as the bars using a matplotlib colormap (although in this case the visualization is probably better without this additional effect).

The following example aims to give a highlight of the behavior of the new align options:

Say you have a lovely style built up for a DataFrame, and now you want to apply the same style to a second DataFrame. Export the style with df1.style.export, and import it on the second DataFrame with df1.style.set

Notice that you’re able to share the styles even though they’re data aware. The styles are re-evaluated on the new DataFrame they’ve been used upon.

DataFrame only (use Series.to_frame().style)

The index and columns do not need to be unique, but certain styling functions can only work with unique indexes.

No large repr, and construction performance isn’t great; although we have some HTML optimizations

You can only apply styles, you can’t insert new HTML entities, except via subclassing.

Here are a few interesting examples.

Styler interacts pretty well with widgets. If you’re viewing this online instead of running the notebook yourself, you’re missing out on interactively adjusting the color palette.

If you display a large matrix or DataFrame in a notebook, but you want to always see the column and row headers you can use the .set_sticky method which manipulates the table styles CSS.

It is also possible to stick MultiIndexes and even only specific levels.

Suppose you have to display HTML within HTML, that can be a bit of pain when the renderer can’t distinguish. You can use the escape formatting option to handle this, and even use it within a formatter that contains HTML itself.

Some support (since version 0.20.0) is available for exporting styled DataFrames to Excel worksheets using the OpenPyXL or XlsxWriter engines. CSS2.2 properties handled include:

border-style properties

border-width properties

border-color properties

Shorthand and side-specific border properties are supported (e.g. border-style and border-left-style) as well as the border shorthands for all sides (border: 1px solid green) or specified sides (border-left: 1px solid green). Using a border shorthand will override any border properties set before it (See CSS Working Group for more details)

Only CSS2 named colors and hex colors of the form #rgb or #rrggbb are currently supported.

The following pseudo CSS properties are also available to set Excel specific style properties:

border-style (for Excel-specific styles: “hair”, “mediumDashDot”, “dashDotDot”, “mediumDashDotDot”, “dashDot”, “slantDashDot”, or “mediumDashed”)

Table level styles, and data cell CSS-classes are not included in the export to Excel: individual cells must have their properties mapped by the Styler.apply and/or Styler.map methods.

A screenshot of the output:

There is support (since version 1.3.0) to export Styler to LaTeX. The documentation for the .to_latex method gives further detail and numerous examples.

Cascading Style Sheet (CSS) language, which is designed to influence how a browser renders HTML elements, has its own peculiarities. It never reports errors: it just silently ignores them and doesn’t render your objects how you intend so can sometimes be frustrating. Here is a very brief primer on how Styler creates HTML and interacts with CSS, with advice on common pitfalls to avoid.

The precise structure of the CSS class attached to each cell is as follows.

Cells with Index and Column names include index_name and level<k> where k is its level in a MultiIndex

Index label cells include

level<k> where k is the level in a MultiIndex

row<m> where m is the numeric position of the row

Column label cells include

level<k> where k is the level in a MultiIndex

col<n> where n is the numeric position of the column

row<m>, where m is the numeric position of the cell.

col<n>, where n is the numeric position of the cell.

Blank cells include blank

Trimmed cells include col_trim or row_trim

The structure of the id is T_uuid_level<k>_row<m>_col<n> where level<k> is used only on headings, and headings will only have either row<m> or col<n> whichever is needed. By default we’ve also prepended each row/column identifier with a UUID unique to each DataFrame so that the style from one doesn’t collide with the styling from another within the same notebook or page. You can read more about the use of UUIDs in Optimization.

We can see example of the HTML by calling the .to_html() method.

The examples have shown that when CSS styles overlap, the one that comes last in the HTML render, takes precedence. So the following yield different results:

This is only true for CSS rules that are equivalent in hierarchy, or importance. You can read more about CSS specificity here but for our purposes it suffices to summarize the key points:

A CSS importance score for each HTML element is derived by starting at zero and adding:

1000 for an inline style attribute

10 for each attribute, class or pseudo-class

1 for each element name or pseudo-element

Let’s use this to describe the action of the following configurations

This text is red because the generated selector #T_a_ td is worth 101 (ID plus element), whereas #T_a_row0_col0 is only worth 100 (ID), so is considered inferior even though in the HTML it comes after the previous.

In the above case the text is blue because the selector #T_b_ .cls-1 is worth 110 (ID plus class), which takes precedence.

Now we have created another table style this time the selector T_c_ td.data (ID plus element plus class) gets bumped up to 111.

If your style fails to be applied, and its really frustrating, try the !important trump card.

Finally got that green text after all!

The core of pandas is, and will remain, its “high-performance, easy-to-use data structures”. With that in mind, we hope that DataFrame.style accomplishes two goals

Provide an API that is pleasing to use interactively and is “good enough” for many tasks

Provide the foundations for dedicated libraries to build on

If you build a great library on top of this, let us know and we’ll link to it.

If the default template doesn’t quite suit your needs, you can subclass Styler and extend or override the template. We’ll show an example of extending the default template to insert a custom header before each table.

We’ll use the following template:

Now that we’ve created a template, we need to set up a subclass of Styler that knows about it.

Notice that we include the original loader in our environment’s loader. That’s because we extend the original template, so the Jinja environment needs to be able to find it.

Now we can use that custom styler. It’s __init__ takes a DataFrame.

Our custom template accepts a table_title keyword. We can provide the value in the .to_html method.

For convenience, we provide the Styler.from_custom_template method that does the same as the custom subclass.

Here’s the template structure for the both the style generation template and the table generation template:

See the template in the GitHub repo for more details.

**Examples:**

Example 1 (json):
```json
import pandas as pd
import numpy as np
import matplotlib as mpl

df = pd.DataFrame({
    "strings": ["Adam", "Mike"],
    "ints": [1, 3],
    "floats": [1.123, 1000.23]
})
df.style \
  .format(precision=3, thousands=".", decimal=",") \
  .format_index(str.upper, axis=1) \
  .relabel_index(["row 1", "row 2"], axis=0)
```

Example 2 (python):
```python
weather_df = pd.DataFrame(np.random.rand(10,2)*5,
                          index=pd.date_range(start="2021-01-01", periods=10),
                          columns=["Tokyo", "Beijing"])

def rain_condition(v):
    if v < 1.75:
        return "Dry"
    elif v < 2.75:
        return "Rain"
    return "Heavy Rain"

def make_pretty(styler):
    styler.set_caption("Weather Conditions")
    styler.format(rain_condition)
    styler.format_index(lambda v: v.strftime("%A"))
    styler.background_gradient(axis=None, vmin=1, vmax=5, cmap="YlGnBu")
    return styler

weather_df
```

Example 3 (json):
```json
weather_df.loc["2021-01-04":"2021-01-08"].style.pipe(make_pretty)
```

Example 4 (unknown):
```unknown
df = pd.DataFrame(np.random.randn(5, 5))
df.style \
  .hide(subset=[0, 2, 4], axis=0) \
  .hide(subset=[0, 2, 4], axis=1)
```

---

## Time series / date functionality#

**URL:** https://pandas.pydata.org/docs/user_guide/timeseries.html

**Contents:**
- Time series / date functionality#
- Overview#
- Timestamps vs. time spans#
- Converting to timestamps#
  - Providing a format argument#
  - Assembling datetime from multiple DataFrame columns#
  - Invalid data#
  - Epoch timestamps#
  - From timestamps to epoch#
  - Using the origin parameter#

pandas contains extensive capabilities and features for working with time series data for all domains. Using the NumPy datetime64 and timedelta64 dtypes, pandas has consolidated a large number of features from other Python libraries like scikits.timeseries as well as created a tremendous amount of new functionality for manipulating time series data.

For example, pandas supports:

Parsing time series information from various sources and formats

Generate sequences of fixed-frequency dates and time spans

Manipulating and converting date times with timezone information

Resampling or converting a time series to a particular frequency

Performing date and time arithmetic with absolute or relative time increments

pandas provides a relatively compact and self-contained set of tools for performing the above tasks and more.

pandas captures 4 general time related concepts:

Date times: A specific date and time with timezone support. Similar to datetime.datetime from the standard library.

Time deltas: An absolute time duration. Similar to datetime.timedelta from the standard library.

Time spans: A span of time defined by a point in time and its associated frequency.

Date offsets: A relative time duration that respects calendar arithmetic. Similar to dateutil.relativedelta.relativedelta from the dateutil package.

Primary Creation Method

datetime64[ns] or datetime64[ns, tz]

to_datetime or date_range

to_timedelta or timedelta_range

Period or period_range

For time series data, it’s conventional to represent the time component in the index of a Series or DataFrame so manipulations can be performed with respect to the time element.

However, Series and DataFrame can directly also support the time component as data itself.

Series and DataFrame have extended data type support and functionality for datetime, timedelta and Period data when passed into those constructors. DateOffset data however will be stored as object data.

Lastly, pandas represents null date times, time deltas, and time spans as NaT which is useful for representing missing or null date like values and behaves similar as np.nan does for float data.

Timestamped data is the most basic type of time series data that associates values with points in time. For pandas objects it means using the points in time.

However, in many cases it is more natural to associate things like change variables with a time span instead. The span represented by Period can be specified explicitly, or inferred from datetime string format.

Timestamp and Period can serve as an index. Lists of Timestamp and Period are automatically coerced to DatetimeIndex and PeriodIndex respectively.

pandas allows you to capture both representations and convert between them. Under the hood, pandas represents timestamps using instances of Timestamp and sequences of timestamps using instances of DatetimeIndex. For regular time spans, pandas uses Period objects for scalar values and PeriodIndex for sequences of spans. Better support for irregular intervals with arbitrary start and end points are forth-coming in future releases.

To convert a Series or list-like object of date-like objects e.g. strings, epochs, or a mixture, you can use the to_datetime function. When passed a Series, this returns a Series (with the same index), while a list-like is converted to a DatetimeIndex:

If you use dates which start with the day first (i.e. European style), you can pass the dayfirst flag:

You see in the above example that dayfirst isn’t strict. If a date can’t be parsed with the day being first it will be parsed as if dayfirst were False and a warning will also be raised.

If you pass a single string to to_datetime, it returns a single Timestamp. Timestamp can also accept string input, but it doesn’t accept string parsing options like dayfirst or format, so use to_datetime if these are required.

You can also use the DatetimeIndex constructor directly:

The string ‘infer’ can be passed in order to set the frequency of the index as the inferred frequency upon creation:

In addition to the required datetime string, a format argument can be passed to ensure specific parsing. This could also potentially speed up the conversion considerably.

For more information on the choices available when specifying the format option, see the Python datetime documentation.

You can also pass a DataFrame of integer or string columns to assemble into a Series of Timestamps.

You can pass only the columns that you need to assemble.

pd.to_datetime looks for standard designations of the datetime component in the column names, including:

required: year, month, day

optional: hour, minute, second, millisecond, microsecond, nanosecond

The default behavior, errors='raise', is to raise when unparsable:

Pass errors='coerce' to convert unparsable data to NaT (not a time):

pandas supports converting integer or float epoch times to Timestamp and DatetimeIndex. The default unit is nanoseconds, since that is how Timestamp objects are stored internally. However, epochs are often stored in another unit which can be specified. These are computed from the starting point specified by the origin parameter.

The unit parameter does not use the same strings as the format parameter that was discussed above). The available units are listed on the documentation for pandas.to_datetime().

Constructing a Timestamp or DatetimeIndex with an epoch timestamp with the tz argument specified will raise a ValueError. If you have epochs in wall time in another timezone, you can read the epochs as timezone-naive timestamps and then localize to the appropriate timezone:

Epoch times will be rounded to the nearest nanosecond.

Conversion of float epoch times can lead to inaccurate and unexpected results. Python floats have about 15 digits precision in decimal. Rounding during conversion from float to high precision Timestamp is unavoidable. The only way to achieve exact precision is to use a fixed-width types (e.g. an int64).

Using the origin parameter

To invert the operation from above, namely, to convert from a Timestamp to a ‘unix’ epoch:

We subtract the epoch (midnight at January 1, 1970 UTC) and then floor divide by the “unit” (1 second).

Using the origin parameter, one can specify an alternative starting point for creation of a DatetimeIndex. For example, to use 1960-01-01 as the starting date:

The default is set at origin='unix', which defaults to 1970-01-01 00:00:00. Commonly called ‘unix epoch’ or POSIX time.

To generate an index with timestamps, you can use either the DatetimeIndex or Index constructor and pass in a list of datetime objects:

In practice this becomes very cumbersome because we often need a very long index with a large number of timestamps. If we need timestamps on a regular frequency, we can use the date_range() and bdate_range() functions to create a DatetimeIndex. The default frequency for date_range is a calendar day while the default for bdate_range is a business day:

Convenience functions like date_range and bdate_range can utilize a variety of frequency aliases:

date_range and bdate_range make it easy to generate a range of dates using various combinations of parameters like start, end, periods, and freq. The start and end dates are strictly inclusive, so dates outside of those specified will not be generated:

Specifying start, end, and periods will generate a range of evenly spaced dates from start to end inclusively, with periods number of elements in the resulting DatetimeIndex:

bdate_range can also generate a range of custom frequency dates by using the weekmask and holidays parameters. These parameters will only be used if a custom frequency string is passed.

The limits of timestamp representation depend on the chosen resolution. For nanosecond resolution, the time span that can be represented using a 64-bit integer is limited to approximately 584 years:

When choosing second-resolution, the available range grows to +/- 2.9e11 years. Different resolutions can be converted to each other through as_unit.

Representing out-of-bounds spans

One of the main uses for DatetimeIndex is as an index for pandas objects. The DatetimeIndex class contains many time series related optimizations:

A large range of dates for various offsets are pre-computed and cached under the hood in order to make generating subsequent date ranges very fast (just have to grab a slice).

Fast shifting using the shift method on pandas objects.

Unioning of overlapping DatetimeIndex objects with the same frequency is very fast (important for fast data alignment).

Quick access to date fields via properties such as year, month, etc.

Regularization functions like snap and very fast asof logic.

DatetimeIndex objects have all the basic functionality of regular Index objects, and a smorgasbord of advanced time series specific methods for easy frequency processing.

While pandas does not force you to have a sorted date index, some of these methods may have unexpected or incorrect behavior if the dates are unsorted.

DatetimeIndex can be used like a regular index and offers all of its intelligent functionality like selection, slicing, etc.

Dates and strings that parse to timestamps can be passed as indexing parameters:

To provide convenience for accessing longer time series, you can also pass in the year or year and month as strings:

This type of slicing will work on a DataFrame with a DatetimeIndex as well. Since the partial string selection is a form of label slicing, the endpoints will be included. This would include matching times on an included date:

Indexing DataFrame rows with a single string with getitem (e.g. frame[dtstring]) is deprecated starting with pandas 1.2.0 (given the ambiguity whether it is indexing the rows or selecting a column) and will be removed in a future version. The equivalent with .loc (e.g. frame.loc[dtstring]) is still supported.

This starts on the very first time in the month, and includes the last date and time for the month:

This specifies a stop time that includes all of the times on the last day:

This specifies an exact stop time (and is not the same as the above):

We are stopping on the included end-point as it is part of the index:

DatetimeIndex partial string indexing also works on a DataFrame with a MultiIndex:

Slicing with string indexing also honors UTC offset.

The same string used as an indexing parameter can be treated either as a slice or as an exact match depending on the resolution of the index. If the string is less accurate than the index, it will be treated as a slice, otherwise as an exact match.

Consider a Series object with a minute resolution index:

A timestamp string less accurate than a minute gives a Series object.

A timestamp string with minute resolution (or more accurate), gives a scalar instead, i.e. it is not casted to a slice.

If index resolution is second, then the minute-accurate timestamp gives a Series.

If the timestamp string is treated as a slice, it can be used to index DataFrame with .loc[] as well.

However, if the string is treated as an exact match, the selection in DataFrame’s [] will be column-wise and not row-wise, see Indexing Basics. For example dft_minute['2011-12-31 23:59'] will raise KeyError as '2012-12-31 23:59' has the same resolution as the index and there is no column with such name:

To always have unambiguous selection, whether the row is treated as a slice or a single selection, use .loc.

Note also that DatetimeIndex resolution cannot be less precise than day.

As discussed in previous section, indexing a DatetimeIndex with a partial string depends on the “accuracy” of the period, in other words how specific the interval is in relation to the resolution of the index. In contrast, indexing with Timestamp or datetime objects is exact, because the objects have exact meaning. These also follow the semantics of including both endpoints.

These Timestamp and datetime objects have exact hours, minutes, and seconds, even though they were not explicitly specified (they are 0).

A truncate() convenience function is provided that is similar to slicing. Note that truncate assumes a 0 value for any unspecified date component in a DatetimeIndex in contrast to slicing which returns any partially matching dates:

Even complicated fancy indexing that breaks the DatetimeIndex frequency regularity will result in a DatetimeIndex, although frequency is lost:

There are several time/date properties that one can access from Timestamp or a collection of timestamps like a DatetimeIndex.

The year of the datetime

The month of the datetime

The days of the datetime

The hour of the datetime

The minutes of the datetime

The seconds of the datetime

The microseconds of the datetime

The nanoseconds of the datetime

Returns datetime.date (does not contain timezone information)

Returns datetime.time (does not contain timezone information)

Returns datetime.time as local time with timezone information

The ordinal day of year

The ordinal day of year

The week ordinal of the year

The week ordinal of the year

The number of the day of the week with Monday=0, Sunday=6

The number of the day of the week with Monday=0, Sunday=6

The number of the day of the week with Monday=0, Sunday=6

Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, etc.

The number of days in the month of the datetime

Logical indicating if first day of month (defined by frequency)

Logical indicating if last day of month (defined by frequency)

Logical indicating if first day of quarter (defined by frequency)

Logical indicating if last day of quarter (defined by frequency)

Logical indicating if first day of year (defined by frequency)

Logical indicating if last day of year (defined by frequency)

Logical indicating if the date belongs to a leap year

Furthermore, if you have a Series with datetimelike values, then you can access these properties via the .dt accessor, as detailed in the section on .dt accessors.

You may obtain the year, week and day components of the ISO year from the ISO 8601 standard:

In the preceding examples, frequency strings (e.g. 'D') were used to specify a frequency that defined:

how the date times in DatetimeIndex were spaced when using date_range()

the frequency of a Period or PeriodIndex

These frequency strings map to a DateOffset object and its subclasses. A DateOffset is similar to a Timedelta that represents a duration of time but follows specific calendar duration rules. For example, a Timedelta day will always increment datetimes by 24 hours, while a DateOffset day will increment datetimes to the same time the next day whether a day represents 23, 24 or 25 hours due to daylight savings time. However, all DateOffset subclasses that are an hour or smaller (Hour, Minute, Second, Milli, Micro, Nano) behave like Timedelta and respect absolute time.

The basic DateOffset acts similar to dateutil.relativedelta (relativedelta documentation) that shifts a date time by the corresponding calendar duration specified. The arithmetic operator (+) can be used to perform the shift.

Most DateOffsets have associated frequencies strings, or offset aliases, that can be passed into freq keyword arguments. The available date offsets and associated frequency strings can be found below:

Generic offset class, defaults to absolute 24 hours

business day (weekday)

CDay or CustomBusinessDay

one week, optionally anchored on a day of the week

the x-th day of the y-th week of each month

the x-th day of the last week of each month

BMonthEnd or BusinessMonthEnd

BMonthBegin or BusinessMonthBegin

CBMonthEnd or CustomBusinessMonthEnd

custom business month end

CBMonthBegin or CustomBusinessMonthBegin

custom business month begin

15th (or other day_of_month) and calendar month end

15th (or other day_of_month) and calendar month begin

calendar quarter begin

business quarter begin

retail (aka 52-53 week) quarter

retail (aka 52-53 week) year

DateOffsets additionally have rollforward() and rollback() methods for moving a date forward or backward respectively to a valid offset date relative to the offset. For example, business offsets will roll dates that land on the weekends (Saturday and Sunday) forward to Monday since business offsets operate on the weekdays.

These operations preserve time (hour, minute, etc) information by default. To reset time to midnight, use normalize() before or after applying the operation (depending on whether you want the time information included in the operation).

Some of the offsets can be “parameterized” when created to result in different behaviors. For example, the Week offset for generating weekly data accepts a weekday parameter which results in the generated dates always lying on a particular day of the week:

The normalize option will be effective for addition and subtraction.

Another example is parameterizing YearEnd with the specific ending month:

Offsets can be used with either a Series or DatetimeIndex to apply the offset to each element.

If the offset class maps directly to a Timedelta (Day, Hour, Minute, Second, Micro, Milli, Nano) it can be used exactly like a Timedelta - see the Timedelta section for more examples.

Note that some offsets (such as BQuarterEnd) do not have a vectorized implementation. They can still be used but may calculate significantly slower and will show a PerformanceWarning

The CDay or CustomBusinessDay class provides a parametric BusinessDay class which can be used to create customized business day calendars which account for local holidays and local weekend conventions.

As an interesting example, let’s look at Egypt where a Friday-Saturday weekend is observed.

Let’s map to the weekday names:

Holiday calendars can be used to provide the list of holidays. See the holiday calendar section for more information.

Monthly offsets that respect a certain holiday calendar can be defined in the usual way.

The frequency string ‘C’ is used to indicate that a CustomBusinessDay DateOffset is used, it is important to note that since CustomBusinessDay is a parameterised type, instances of CustomBusinessDay may differ and this is not detectable from the ‘C’ frequency string. The user therefore needs to ensure that the ‘C’ frequency string is used consistently within the user’s application.

The BusinessHour class provides a business hour representation on BusinessDay, allowing to use specific start and end times.

By default, BusinessHour uses 9:00 - 17:00 as business hours. Adding BusinessHour will increment Timestamp by hourly frequency. If target Timestamp is out of business hours, move to the next business hour then increment it. If the result exceeds the business hours end, the remaining hours are added to the next business day.

You can also specify start and end time by keywords. The argument must be a str with an hour:minute representation or a datetime.time instance. Specifying seconds, microseconds and nanoseconds as business hour results in ValueError.

Passing start time later than end represents midnight business hour. In this case, business hour exceeds midnight and overlap to the next day. Valid business hours are distinguished by whether it started from valid BusinessDay.

Applying BusinessHour.rollforward and rollback to out of business hours results in the next business hour start or previous day’s end. Different from other offsets, BusinessHour.rollforward may output different results from apply by definition.

This is because one day’s business hour end is equal to next day’s business hour start. For example, under the default business hours (9:00 - 17:00), there is no gap (0 minutes) between 2014-08-01 17:00 and 2014-08-04 09:00.

BusinessHour regards Saturday and Sunday as holidays. To use arbitrary holidays, you can use CustomBusinessHour offset, as explained in the following subsection.

The CustomBusinessHour is a mixture of BusinessHour and CustomBusinessDay which allows you to specify arbitrary holidays. CustomBusinessHour works as the same as BusinessHour except that it skips specified custom holidays.

You can use keyword arguments supported by either BusinessHour and CustomBusinessDay.

A number of string aliases are given to useful common time series frequencies. We will refer to these aliases as offset aliases.

business day frequency

custom business day frequency

calendar day frequency

semi-month end frequency (15th and end of month)

business month end frequency

custom business month end frequency

month start frequency

semi-month start frequency (1st and 15th)

business month start frequency

custom business month start frequency

quarter end frequency

business quarter end frequency

quarter start frequency

business quarter start frequency

business year end frequency

business year start frequency

business hour frequency

custom business hour frequency

Deprecated since version 2.2.0: Aliases H, BH, CBH, T, S, L, U, and N are deprecated in favour of the aliases h, bh, cbh, min, s, ms, us, and ns.

When using the offset aliases above, it should be noted that functions such as date_range(), bdate_range(), will only return timestamps that are in the interval defined by start_date and end_date. If the start_date does not correspond to the frequency, the returned timestamps will start at the next valid timestamp, same for end_date, the returned timestamps will stop at the previous valid timestamp.

For example, for the offset MS, if the start_date is not the first of the month, the returned timestamps will start with the first day of the next month. If end_date is not the first day of a month, the last returned timestamp will be the first day of the corresponding month.

We can see in the above example date_range() and bdate_range() will only return the valid timestamps between the start_date and end_date. If these are not valid timestamps for the given frequency it will roll to the next value for start_date (respectively previous for the end_date)

A number of string aliases are given to useful common time series frequencies. We will refer to these aliases as period aliases.

business day frequency

calendar day frequency

Deprecated since version 2.2.0: Aliases A, H, T, S, L, U, and N are deprecated in favour of the aliases Y, h, min, s, ms, us, and ns.

As we have seen previously, the alias and the offset instance are fungible in most functions:

You can combine together day and intraday offsets:

For some frequencies you can specify an anchoring suffix:

weekly frequency (Sundays). Same as ‘W’

weekly frequency (Mondays)

weekly frequency (Tuesdays)

weekly frequency (Wednesdays)

weekly frequency (Thursdays)

weekly frequency (Fridays)

weekly frequency (Saturdays)

quarterly frequency, year ends in December. Same as ‘QE’

quarterly frequency, year ends in January

quarterly frequency, year ends in February

quarterly frequency, year ends in March

quarterly frequency, year ends in April

quarterly frequency, year ends in May

quarterly frequency, year ends in June

quarterly frequency, year ends in July

quarterly frequency, year ends in August

quarterly frequency, year ends in September

quarterly frequency, year ends in October

quarterly frequency, year ends in November

annual frequency, anchored end of December. Same as ‘YE’

annual frequency, anchored end of January

annual frequency, anchored end of February

annual frequency, anchored end of March

annual frequency, anchored end of April

annual frequency, anchored end of May

annual frequency, anchored end of June

annual frequency, anchored end of July

annual frequency, anchored end of August

annual frequency, anchored end of September

annual frequency, anchored end of October

annual frequency, anchored end of November

These can be used as arguments to date_range, bdate_range, constructors for DatetimeIndex, as well as various other timeseries-related functions in pandas.

For those offsets that are anchored to the start or end of specific frequency (MonthEnd, MonthBegin, WeekEnd, etc), the following rules apply to rolling forward and backwards.

When n is not 0, if the given date is not on an anchor point, it snapped to the next(previous) anchor point, and moved |n|-1 additional steps forwards or backwards.

If the given date is on an anchor point, it is moved |n| points forwards or backwards.

For the case when n=0, the date is not moved if on an anchor point, otherwise it is rolled forward to the next anchor point.

Holidays and calendars provide a simple way to define holiday rules to be used with CustomBusinessDay or in other analysis that requires a predefined set of holidays. The AbstractHolidayCalendar class provides all the necessary methods to return a list of holidays and only rules need to be defined in a specific holiday calendar class. Furthermore, the start_date and end_date class attributes determine over what date range holidays are generated. These should be overwritten on the AbstractHolidayCalendar class to have the range apply to all calendar subclasses. USFederalHolidayCalendar is the only calendar that exists and primarily serves as an example for developing other calendars.

For holidays that occur on fixed dates (e.g., US Memorial Day or July 4th) an observance rule determines when that holiday is observed if it falls on a weekend or some other non-observed day. Defined observance rules are:

move Saturday to Friday and Sunday to Monday

move Sunday to following Monday

next_monday_or_tuesday

move Saturday to Monday and Sunday/Monday to Tuesday

move Saturday and Sunday to previous Friday”

move Saturday and Sunday to following Monday

An example of how holidays and holiday calendars are defined:

weekday=MO(2) is same as 2 * Week(weekday=2)

Using this calendar, creating an index or doing offset arithmetic skips weekends and holidays (i.e., Memorial Day/July 4th). For example, the below defines a custom business day offset using the ExampleCalendar. Like any other offset, it can be used to create a DatetimeIndex or added to datetime or Timestamp objects.

Ranges are defined by the start_date and end_date class attributes of AbstractHolidayCalendar. The defaults are shown below.

These dates can be overwritten by setting the attributes as datetime/Timestamp/string.

Every calendar class is accessible by name using the get_calendar function which returns a holiday class instance. Any imported calendar class will automatically be available by this function. Also, HolidayCalendarFactory provides an easy interface to create calendars that are combinations of calendars or calendars with additional rules.

One may want to shift or lag the values in a time series back and forward in time. The method for this is shift(), which is available on all of the pandas objects.

The shift method accepts an freq argument which can accept a DateOffset class or other timedelta-like object or also an offset alias.

When freq is specified, shift method changes all the dates in the index rather than changing the alignment of the data and the index:

Note that with when freq is specified, the leading entry is no longer NaN because the data is not being realigned.

The primary function for changing frequencies is the asfreq() method. For a DatetimeIndex, this is basically just a thin, but convenient wrapper around reindex() which generates a date_range and calls reindex.

asfreq provides a further convenience so you can specify an interpolation method for any gaps that may appear after the frequency conversion.

Related to asfreq and reindex is fillna(), which is documented in the missing data section.

DatetimeIndex can be converted to an array of Python native datetime.datetime objects using the to_pydatetime method.

pandas has a simple, powerful, and efficient functionality for performing resampling operations during frequency conversion (e.g., converting secondly data into 5-minutely data). This is extremely common in, but not limited to, financial applications.

resample() is a time-based groupby, followed by a reduction method on each of its groups. See some cookbook examples for some advanced strategies.

The resample() method can be used directly from DataFrameGroupBy objects, see the groupby docs.

The resample function is very flexible and allows you to specify many different parameters to control the frequency conversion and resampling operation.

Any built-in method available via GroupBy is available as a method of the returned object, including sum, mean, std, sem, max, min, median, first, last, ohlc:

For downsampling, closed can be set to ‘left’ or ‘right’ to specify which end of the interval is closed:

Parameters like label are used to manipulate the resulting labels. label specifies whether the result is labeled with the beginning or the end of the interval.

The default values for label and closed is ‘left’ for all frequency offsets except for ‘ME’, ‘YE’, ‘QE’, ‘BME’, ‘BYE’, ‘BQE’, and ‘W’ which all have a default of ‘right’.

This might unintendedly lead to looking ahead, where the value for a later time is pulled back to a previous time as in the following example with the BusinessDay frequency:

Notice how the value for Sunday got pulled back to the previous Friday. To get the behavior where the value for Sunday is pushed to Monday, use instead

The axis parameter can be set to 0 or 1 and allows you to resample the specified axis for a DataFrame.

kind can be set to ‘timestamp’ or ‘period’ to convert the resulting index to/from timestamp and time span representations. By default resample retains the input representation.

convention can be set to ‘start’ or ‘end’ when resampling period data (detail below). It specifies how low frequency periods are converted to higher frequency periods.

For upsampling, you can specify a way to upsample and the limit parameter to interpolate over the gaps that are created:

Sparse timeseries are the ones where you have a lot fewer points relative to the amount of time you are looking to resample. Naively upsampling a sparse series can potentially generate lots of intermediate values. When you don’t want to use a method to fill these values, e.g. fill_method is None, then intermediate values will be filled with NaN.

Since resample is a time-based groupby, the following is a method to efficiently resample only the groups that are not all NaN.

If we want to resample to the full range of the series:

We can instead only resample those groups where we have points as follows:

The resample() method returns a pandas.api.typing.Resampler instance. Similar to the aggregating API, groupby API, and the window API, a Resampler can be selectively resampled.

Resampling a DataFrame, the default will be to act on all columns with the same function.

We can select a specific column or columns using standard getitem.

You can pass a list or dict of functions to do aggregation with, outputting a DataFrame:

On a resampled DataFrame, you can pass a list of functions to apply to each column, which produces an aggregated result with a hierarchical index:

By passing a dict to aggregate you can apply a different aggregation to the columns of a DataFrame:

The function names can also be strings. In order for a string to be valid it must be implemented on the resampled object:

Furthermore, you can also specify multiple aggregation functions for each column separately.

If a DataFrame does not have a datetimelike index, but instead you want to resample based on datetimelike column in the frame, it can passed to the on keyword.

Similarly, if you instead want to resample by a datetimelike level of MultiIndex, its name or location can be passed to the level keyword.

With the Resampler object in hand, iterating through the grouped data is very natural and functions similarly to itertools.groupby():

See Iterating through groups or Resampler.__iter__ for more.

The bins of the grouping are adjusted based on the beginning of the day of the time series starting point. This works well with frequencies that are multiples of a day (like 30D) or that divide a day evenly (like 90s or 1min). This can create inconsistencies with some frequencies that do not meet this criteria. To change this behavior you can specify a fixed Timestamp with the argument origin.

Here we can see that, when using origin with its default value ('start_day'), the result after '2000-10-02 00:00:00' are not identical depending on the start of time series:

Here we can see that, when setting origin to 'epoch', the result after '2000-10-02 00:00:00' are identical depending on the start of time series:

If needed you can use a custom timestamp for origin:

If needed you can just adjust the bins with an offset Timedelta that would be added to the default origin. Those two examples are equivalent for this time series:

Note the use of 'start' for origin on the last example. In that case, origin will be set to the first value of the timeseries.

Added in version 1.3.0.

Instead of adjusting the beginning of bins, sometimes we need to fix the end of the bins to make a backward resample with a given freq. The backward resample sets closed to 'right' by default since the last value should be considered as the edge point for the last bin.

We can set origin to 'end'. The value for a specific Timestamp index stands for the resample result from the current Timestamp minus freq to the current Timestamp with a right close.

Besides, in contrast with the 'start_day' option, end_day is supported. This will set the origin as the ceiling midnight of the largest Timestamp.

The above result uses 2000-10-02 00:29:00 as the last bin’s right edge since the following computation.

Regular intervals of time are represented by Period objects in pandas while sequences of Period objects are collected in a PeriodIndex, which can be created with the convenience function period_range.

A Period represents a span of time (e.g., a day, a month, a quarter, etc). You can specify the span via freq keyword using a frequency alias like below. Because freq represents a span of Period, it cannot be negative like “-3D”.

Adding and subtracting integers from periods shifts the period by its own frequency. Arithmetic is not allowed between Period with different freq (span).

If Period freq is daily or higher (D, h, min, s, ms, us, and ns), offsets and timedelta-like can be added if the result can have the same freq. Otherwise, ValueError will be raised.

If Period has other frequencies, only the same offsets can be added. Otherwise, ValueError will be raised.

Taking the difference of Period instances with the same frequency will return the number of frequency units between them:

Regular sequences of Period objects can be collected in a PeriodIndex, which can be constructed using the period_range convenience function:

The PeriodIndex constructor can also be used directly:

Passing multiplied frequency outputs a sequence of Period which has multiplied span.

If start or end are Period objects, they will be used as anchor endpoints for a PeriodIndex with frequency matching that of the PeriodIndex constructor.

Just like DatetimeIndex, a PeriodIndex can also be used to index pandas objects:

PeriodIndex supports addition and subtraction with the same rule as Period.

PeriodIndex has its own dtype named period, refer to Period Dtypes.

PeriodIndex has a custom period dtype. This is a pandas extension dtype similar to the timezone aware dtype (datetime64[ns, tz]).

The period dtype holds the freq attribute and is represented with period[freq] like period[D] or period[M], using frequency strings.

The period dtype can be used in .astype(...). It allows one to change the freq of a PeriodIndex like .asfreq() and convert a DatetimeIndex to PeriodIndex like to_period():

PeriodIndex now supports partial string slicing with non-monotonic indexes.

You can pass in dates and strings to Series and DataFrame with PeriodIndex, in the same manner as DatetimeIndex. For details, refer to DatetimeIndex Partial String Indexing.

Passing a string representing a lower frequency than PeriodIndex returns partial sliced data.

As with DatetimeIndex, the endpoints will be included in the result. The example below slices data starting from 10:00 to 11:59.

The frequency of Period and PeriodIndex can be converted via the asfreq method. Let’s start with the fiscal year 2011, ending in December:

We can convert it to a monthly frequency. Using the how parameter, we can specify whether to return the starting or ending month:

The shorthands ‘s’ and ‘e’ are provided for convenience:

Converting to a “super-period” (e.g., annual frequency is a super-period of quarterly frequency) automatically returns the super-period that includes the input period:

Note that since we converted to an annual frequency that ends the year in November, the monthly period of December 2011 is actually in the 2012 Y-NOV period.

Period conversions with anchored frequencies are particularly useful for working with various quarterly data common to economics, business, and other fields. Many organizations define quarters relative to the month in which their fiscal year starts and ends. Thus, first quarter of 2011 could start in 2010 or a few months into 2011. Via anchored frequencies, pandas works for all quarterly frequencies Q-JAN through Q-DEC.

Q-DEC define regular calendar quarters:

Q-MAR defines fiscal year end in March:

Timestamped data can be converted to PeriodIndex-ed data using to_period and vice-versa using to_timestamp:

Remember that ‘s’ and ‘e’ can be used to return the timestamps at the start or end of the period:

Converting between period and timestamp enables some convenient arithmetic functions to be used. In the following example, we convert a quarterly frequency with year ending in November to 9am of the end of the month following the quarter end:

If you have data that is outside of the Timestamp bounds, see Timestamp limitations, then you can use a PeriodIndex and/or Series of Periods to do computations.

To convert from an int64 based YYYYMMDD representation.

These can easily be converted to a PeriodIndex:

pandas provides rich support for working with timestamps in different time zones using the pytz and dateutil libraries or datetime.timezone objects from the standard library.

By default, pandas objects are time zone unaware:

To localize these dates to a time zone (assign a particular time zone to a naive date), you can use the tz_localize method or the tz keyword argument in date_range(), Timestamp, or DatetimeIndex. You can either pass pytz or dateutil time zone objects or Olson time zone database strings. Olson time zone strings will return pytz time zone objects by default. To return dateutil time zone objects, append dateutil/ before the string.

In pytz you can find a list of common (and less common) time zones using from pytz import common_timezones, all_timezones.

dateutil uses the OS time zones so there isn’t a fixed list available. For common zones, the names are the same as pytz.

Note that the UTC time zone is a special case in dateutil and should be constructed explicitly as an instance of dateutil.tz.tzutc. You can also construct other time zones objects explicitly first.

To convert a time zone aware pandas object from one time zone to another, you can use the tz_convert method.

When using pytz time zones, DatetimeIndex will construct a different time zone object than a Timestamp for the same time zone input. A DatetimeIndex can hold a collection of Timestamp objects that may have different UTC offsets and cannot be succinctly represented by one pytz time zone instance while one Timestamp represents one point in time with a specific UTC offset.

Be wary of conversions between libraries. For some time zones, pytz and dateutil have different definitions of the zone. This is more of a problem for unusual time zones than for ‘standard’ zones like US/Eastern.

Be aware that a time zone definition across versions of time zone libraries may not be considered equal. This may cause problems when working with stored data that is localized using one version and operated on with a different version. See here for how to handle such a situation.

For pytz time zones, it is incorrect to pass a time zone object directly into the datetime.datetime constructor (e.g., datetime.datetime(2011, 1, 1, tzinfo=pytz.timezone('US/Eastern')). Instead, the datetime needs to be localized using the localize method on the pytz time zone object.

Be aware that for times in the future, correct conversion between time zones (and UTC) cannot be guaranteed by any time zone library because a timezone’s offset from UTC may be changed by the respective government.

If you are using dates beyond 2038-01-18, due to current deficiencies in the underlying libraries caused by the year 2038 problem, daylight saving time (DST) adjustments to timezone aware dates will not be applied. If and when the underlying libraries are fixed, the DST transitions will be applied.

For example, for two dates that are in British Summer Time (and so would normally be GMT+1), both the following asserts evaluate as true:

Under the hood, all timestamps are stored in UTC. Values from a time zone aware DatetimeIndex or Timestamp will have their fields (day, hour, minute, etc.) localized to the time zone. However, timestamps with the same UTC value are still considered to be equal even if they are in different time zones:

Operations between Series in different time zones will yield UTC Series, aligning the data on the UTC timestamps:

To remove time zone information, use tz_localize(None) or tz_convert(None). tz_localize(None) will remove the time zone yielding the local time representation. tz_convert(None) will remove the time zone after converting to UTC time.

For ambiguous times, pandas supports explicitly specifying the keyword-only fold argument. Due to daylight saving time, one wall clock time can occur twice when shifting from summer to winter time; fold describes whether the datetime-like corresponds to the first (0) or the second time (1) the wall clock hits the ambiguous time. Fold is supported only for constructing from naive datetime.datetime (see datetime documentation for details) or from Timestamp or for constructing from components (see below). Only dateutil timezones are supported (see dateutil documentation for dateutil methods that deal with ambiguous datetimes) as pytz timezones do not support fold (see pytz documentation for details on how pytz deals with ambiguous datetimes). To localize an ambiguous datetime with pytz, please use Timestamp.tz_localize(). In general, we recommend to rely on Timestamp.tz_localize() when localizing ambiguous datetimes if you need direct control over how they are handled.

tz_localize may not be able to determine the UTC offset of a timestamp because daylight savings time (DST) in a local time zone causes some times to occur twice within one day (“clocks fall back”). The following options are available:

'raise': Raises a pytz.AmbiguousTimeError (the default behavior)

'infer': Attempt to determine the correct offset base on the monotonicity of the timestamps

'NaT': Replaces ambiguous times with NaT

bool: True represents a DST time, False represents non-DST time. An array-like of bool values is supported for a sequence of times.

This will fail as there are ambiguous times ('11/06/2011 01:00')

Handle these ambiguous times by specifying the following.

A DST transition may also shift the local time ahead by 1 hour creating nonexistent local times (“clocks spring forward”). The behavior of localizing a timeseries with nonexistent times can be controlled by the nonexistent argument. The following options are available:

'raise': Raises a pytz.NonExistentTimeError (the default behavior)

'NaT': Replaces nonexistent times with NaT

'shift_forward': Shifts nonexistent times forward to the closest real time

'shift_backward': Shifts nonexistent times backward to the closest real time

timedelta object: Shifts nonexistent times by the timedelta duration

Localization of nonexistent times will raise an error by default.

Transform nonexistent times to NaT or shift the times.

A Series with time zone naive values is represented with a dtype of datetime64[ns].

A Series with a time zone aware values is represented with a dtype of datetime64[ns, tz] where tz is the time zone

Both of these Series time zone information can be manipulated via the .dt accessor, see the dt accessor section.

For example, to localize and convert a naive stamp to time zone aware.

Time zone information can also be manipulated using the astype method. This method can convert between different timezone-aware dtypes.

Using Series.to_numpy() on a Series, returns a NumPy array of the data. NumPy does not currently support time zones (even though it is printing in the local time zone!), therefore an object array of Timestamps is returned for time zone aware data:

By converting to an object array of Timestamps, it preserves the time zone information. For example, when converting back to a Series:

However, if you want an actual NumPy datetime64[ns] array (with the values converted to UTC) instead of an array of objects, you can specify the dtype argument:

**Examples:**

Example 1 (typescript):
```typescript
In [1]: import datetime

In [2]: dti = pd.to_datetime(
   ...:     ["1/1/2018", np.datetime64("2018-01-01"), datetime.datetime(2018, 1, 1)]
   ...: )
   ...: 

In [3]: dti
Out[3]: DatetimeIndex(['2018-01-01', '2018-01-01', '2018-01-01'], dtype='datetime64[ns]', freq=None)
```

Example 2 (typescript):
```typescript
In [4]: dti = pd.date_range("2018-01-01", periods=3, freq="h")

In [5]: dti
Out[5]: 
DatetimeIndex(['2018-01-01 00:00:00', '2018-01-01 01:00:00',
               '2018-01-01 02:00:00'],
              dtype='datetime64[ns]', freq='h')
```

Example 3 (typescript):
```typescript
In [6]: dti = dti.tz_localize("UTC")

In [7]: dti
Out[7]: 
DatetimeIndex(['2018-01-01 00:00:00+00:00', '2018-01-01 01:00:00+00:00',
               '2018-01-01 02:00:00+00:00'],
              dtype='datetime64[ns, UTC]', freq='h')

In [8]: dti.tz_convert("US/Pacific")
Out[8]: 
DatetimeIndex(['2017-12-31 16:00:00-08:00', '2017-12-31 17:00:00-08:00',
               '2017-12-31 18:00:00-08:00'],
              dtype='datetime64[ns, US/Pacific]', freq='h')
```

Example 4 (typescript):
```typescript
In [9]: idx = pd.date_range("2018-01-01", periods=5, freq="h")

In [10]: ts = pd.Series(range(len(idx)), index=idx)

In [11]: ts
Out[11]: 
2018-01-01 00:00:00    0
2018-01-01 01:00:00    1
2018-01-01 02:00:00    2
2018-01-01 03:00:00    3
2018-01-01 04:00:00    4
Freq: h, dtype: int64

In [12]: ts.resample("2h").mean()
Out[12]: 
2018-01-01 00:00:00    0.5
2018-01-01 02:00:00    2.5
2018-01-01 04:00:00    4.0
Freq: 2h, dtype: float64
```

---

## 

**URL:** https://pandas.pydata.org/docs/_sources/user_guide/copy_on_write.rst.txt

---

## Cookbook#

**URL:** https://pandas.pydata.org/docs/user_guide/cookbook.html

**Contents:**
- Cookbook#
- Idioms#
  - if-then…#
  - Splitting#
  - Building criteria#
- Selection#
  - Dataframes#
  - New columns#
- Multiindexing#
  - Arithmetic#

This is a repository for short and sweet examples and links for useful pandas recipes. We encourage users to add to this documentation.

Adding interesting links and/or inline examples to this section is a great First Pull Request.

Simplified, condensed, new-user friendly, in-line examples have been inserted where possible to augment the Stack-Overflow and GitHub links. Many of the links contain expanded information, above what the in-line examples offer.

pandas (pd) and NumPy (np) are the only two abbreviated imported modules. The rest are kept explicitly imported for newer users.

These are some neat pandas idioms

if-then/if-then-else on one column, and assignment to another one or more columns:

An if-then on one column

An if-then with assignment to 2 columns:

Add another line with different logic, to do the -else

Or use pandas where after you’ve set up a mask

if-then-else using NumPy’s where()

Split a frame with a boolean criterion

Select with multi-column criteria

…and (without assignment returns a Series)

…or (without assignment returns a Series)

…or (with assignment modifies the DataFrame.)

Select rows with data closest to certain value using argsort

Dynamically reduce a list of criteria using a binary operators

…Or it can be done with a list of dynamically built criteria

Using both row labels and value conditionals

Use loc for label-oriented slicing and iloc positional slicing GH 2904

There are 2 explicit slicing methods, with a third general case

Positional-oriented (Python slicing style : exclusive of end)

Label-oriented (Non-Python slicing style : inclusive of end)

General (Either slicing style : depends on if the slice contains labels or positions)

Ambiguity arises when an index consists of integers with a non-zero start or non-unit increment.

Using inverse operator (~) to take the complement of a mask

Efficiently and dynamically creating new columns using DataFrame.map (previously named applymap)

Keep other columns when using min() with groupby

Method 1 : idxmin() to get the index of the minimums

Method 2 : sort then take first of each

Notice the same results, with the exception of the index.

The multindexing docs.

Creating a MultiIndex from a labeled frame

Performing arithmetic with a MultiIndex that needs broadcasting

Slicing a MultiIndex with xs

To take the cross section of the 1st level and 1st axis the index:

…and now the 2nd level of the 1st axis.

Slicing a MultiIndex with xs, method #2

Setting portions of a MultiIndex with xs

Sort by specific column or an ordered list of columns, with a MultiIndex

Partial selection, the need for sortedness GH 2995

Prepending a level to a multiindex

Flatten Hierarchical columns

The missing data docs.

Fill forward a reversed timeseries

cumsum reset at NaN values

Using replace with backrefs

Basic grouping with apply

Unlike agg, apply’s callable is passed a sub-DataFrame which gives you access to all the columns

Apply to different items in a group

Replacing some values with mean of the rest of a group

Sort groups by aggregated data

Create multiple aggregated columns

Create a value counts column and reassign back to the DataFrame

Shift groups of the values in a column based on the index

Select row with maximum value from each group

Grouping like Python’s itertools.groupby

Alignment and to-date

Rolling Computation window based on values instead of counts

Rolling Mean by Time Interval

Create a list of dataframes, split using a delineation based on logic included in rows.

Partial sums and subtotals

Frequency table like plyr in R

Plot pandas DataFrame with year over year data

To create year and month cross tabulation:

Rolling apply to organize - Turning embedded lists into a MultiIndex frame

Rolling apply with a DataFrame returning a Series

Rolling Apply to multiple columns where function calculates a Series before a Scalar from the Series is returned

Rolling apply with a DataFrame returning a Scalar

Rolling Apply to multiple columns where function returns a Scalar (Volume Weighted Average Price)

Using indexer between time

Constructing a datetime range that excludes weekends and includes only certain times

Aggregation and plotting time series

Turn a matrix with hours in columns and days in rows into a continuous row sequence in the form of a time series. How to rearrange a Python pandas DataFrame?

Dealing with duplicates when reindexing a timeseries to a specified frequency

Calculate the first day of the month for each entry in a DatetimeIndex

Using Grouper instead of TimeGrouper for time grouping of values

Time grouping with some missing values

Valid frequency arguments to Grouper Timeseries

Grouping using a MultiIndex

Using TimeGrouper and another grouping to create subgroups, then apply a custom function GH 3791

Resampling with custom periods

Resample intraday frame without adding new days

Resample with groupby

Concatenate two dataframes with overlapping index (emulate R rbind)

Depending on df construction, ignore_index may be needed

Self Join of a DataFrame GH 2996

How to set the index and join

Join with a criteria based on the values

Using searchsorted to merge based on values inside a range

Make Matplotlib look like R

Setting x-axis major and minor labels

Plotting multiple charts in an IPython Jupyter notebook

Creating a multi-line plot

Annotate a time-series plot

Annotate a time-series plot #2

Generate Embedded plots in excel files using Pandas, Vincent and xlsxwriter

Boxplot for each quartile of a stratifying variable

Performance comparison of SQL vs HDF5

Reading a csv chunk-by-chunk

Reading only certain rows of a csv chunk-by-chunk

Reading the first few lines of a frame

Reading a file that is compressed but not by gzip/bz2 (the native compressed formats which read_csv understands). This example shows a WinZipped file, but is a general application of opening the file within a context manager and using that handle to read. See here

Inferring dtypes from a file

Dealing with bad lines GH 2886

Write a multi-row index CSV without writing duplicates

The best way to combine multiple files into a single DataFrame is to read the individual frames one by one, put all of the individual frames into a list, and then combine the frames in the list using pd.concat():

You can use the same approach to read all files matching a pattern. Here is an example using glob:

Finally, this strategy will work with the other pd.read_*(...) functions described in the io docs.

Parsing date components in multi-columns is faster with a format

Reading from databases with SQL

Reading from a filelike handle

Modifying formatting in XlsxWriter output

Loading only visible sheets GH 19842#issuecomment-892150745

Reading HTML tables from a server that cannot handle the default request header

Simple queries with a Timestamp Index

Managing heterogeneous data using a linked multiple table hierarchy GH 3032

Merging on-disk tables with millions of rows

Avoiding inconsistencies when writing to a store from multiple processes/threads

De-duplicating a large store by chunks, essentially a recursive reduction operation. Shows a function for taking in data from csv file and creating a store by chunks, with date parsing as well. See here

Creating a store chunk-by-chunk from a csv file

Appending to a store, while creating a unique index

Large Data work flows

Reading in a sequence of files, then providing a global unique index to a store while appending

Groupby on a HDFStore with low group density

Groupby on a HDFStore with high group density

Hierarchical queries on a HDFStore

Counting with a HDFStore

Troubleshoot HDFStore exceptions

Setting min_itemsize with strings

Using ptrepack to create a completely-sorted-index on a store

Storing Attributes to a group node

You can create or load a HDFStore in-memory by passing the driver parameter to PyTables. Changes are only written to disk when the HDFStore is closed.

pandas readily accepts NumPy record arrays, if you need to read in a binary file consisting of an array of C structs. For example, given this C program in a file called main.c compiled with gcc main.c -std=gnu99 on a 64-bit machine,

the following Python code will read the binary file 'binary.dat' into a pandas DataFrame, where each element of the struct corresponds to a column in the frame:

The offsets of the structure elements may be different depending on the architecture of the machine on which the file was created. Using a raw binary file format like this for general data storage is not recommended, as it is not cross platform. We recommended either HDF5 or parquet, both of which are supported by pandas’ IO facilities.

Numerical integration (sample-based) of a time series

Often it’s useful to obtain the lower (or upper) triangular form of a correlation matrix calculated from DataFrame.corr(). This can be achieved by passing a boolean mask to where as follows:

The method argument within DataFrame.corr can accept a callable in addition to the named correlation types. Here we compute the distance correlation matrix for a DataFrame object.

Adding and subtracting deltas and dates

Values can be set to NaT using np.nan, similar to datetime

To create a dataframe from every combination of some given values, like R’s expand.grid() function, we can create a dict where the keys are column names and the values are lists of the data values:

To assess if a series has a constant value, we can check if series.nunique() <= 1. However, a more performant approach, that does not count all unique values first, is:

This approach assumes that the series does not contain missing values. For the case that we would drop NA values, we can simply remove those values first:

If missing values are considered distinct from any other value, then one could use:

(Note that this example does not disambiguate between np.nan, pd.NA and None)

**Examples:**

Example 1 (json):
```json
In [1]: df = pd.DataFrame(
   ...:     {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
   ...: )
   ...: 

In [2]: df
Out[2]: 
   AAA  BBB  CCC
0    4   10  100
1    5   20   50
2    6   30  -30
3    7   40  -50
```

Example 2 (unknown):
```unknown
In [3]: df.loc[df.AAA >= 5, "BBB"] = -1

In [4]: df
Out[4]: 
   AAA  BBB  CCC
0    4   10  100
1    5   -1   50
2    6   -1  -30
3    7   -1  -50
```

Example 3 (unknown):
```unknown
In [5]: df.loc[df.AAA >= 5, ["BBB", "CCC"]] = 555

In [6]: df
Out[6]: 
   AAA  BBB  CCC
0    4   10  100
1    5  555  555
2    6  555  555
3    7  555  555
```

Example 4 (unknown):
```unknown
In [7]: df.loc[df.AAA < 5, ["BBB", "CCC"]] = 2000

In [8]: df
Out[8]: 
   AAA   BBB   CCC
0    4  2000  2000
1    5   555   555
2    6   555   555
3    7   555   555
```

---

## Scaling to large datasets#

**URL:** https://pandas.pydata.org/docs/user_guide/scale.html

**Contents:**
- Scaling to large datasets#
- Load less data#
- Use efficient datatypes#
- Use chunking#
- Use Other Libraries#

pandas provides data structures for in-memory analytics, which makes using pandas to analyze datasets that are larger than memory datasets somewhat tricky. Even datasets that are a sizable fraction of memory become unwieldy, as some pandas operations need to make intermediate copies.

This document provides a few recommendations for scaling your analysis to larger datasets. It’s a complement to Enhancing performance, which focuses on speeding up analysis for datasets that fit in memory.

Suppose our raw dataset on disk has many columns.

To load the columns we want, we have two options. Option 1 loads in all the data and then filters to what we need.

Option 2 only loads the columns we request.

If we were to measure the memory usage of the two calls, we’d see that specifying columns uses about 1/10th the memory in this case.

With pandas.read_csv(), you can specify usecols to limit the columns read into memory. Not all file formats that can be read by pandas provide an option to read a subset of columns.

The default pandas data types are not the most memory efficient. This is especially true for text data columns with relatively few unique values (commonly referred to as “low-cardinality” data). By using more efficient data types, you can store larger datasets in memory.

Now, let’s inspect the data types and memory usage to see where we should focus our attention.

The name column is taking up much more memory than any other. It has just a few unique values, so it’s a good candidate for converting to a pandas.Categorical. With a pandas.Categorical, we store each unique name once and use space-efficient integers to know which specific name is used in each row.

We can go a bit further and downcast the numeric columns to their smallest types using pandas.to_numeric().

In all, we’ve reduced the in-memory footprint of this dataset to 1/5 of its original size.

See Categorical data for more on pandas.Categorical and dtypes for an overview of all of pandas’ dtypes.

Some workloads can be achieved with chunking by splitting a large problem into a bunch of small problems. For example, converting an individual CSV file into a Parquet file and repeating that for each file in a directory. As long as each chunk fits in memory, you can work with datasets that are much larger than memory.

Chunking works well when the operation you’re performing requires zero or minimal coordination between chunks. For more complicated workflows, you’re better off using other libraries.

Suppose we have an even larger “logical dataset” on disk that’s a directory of parquet files. Each file in the directory represents a different year of the entire dataset.

Now we’ll implement an out-of-core pandas.Series.value_counts(). The peak memory usage of this workflow is the single largest chunk, plus a small series storing the unique value counts up to this point. As long as each individual file fits in memory, this will work for arbitrary-sized datasets.

Some readers, like pandas.read_csv(), offer parameters to control the chunksize when reading a single file.

Manually chunking is an OK option for workflows that don’t require too sophisticated of operations. Some operations, like pandas.DataFrame.groupby(), are much harder to do chunkwise. In these cases, you may be better switching to a different library that implements these out-of-core algorithms for you.

There are other libraries which provide similar APIs to pandas and work nicely with pandas DataFrame, and can give you the ability to scale your large dataset processing and analytics by parallel runtime, distributed memory, clustering, etc. You can find more information in the ecosystem page.

**Examples:**

Example 1 (json):
```json
In [1]: import pandas as pd

In [2]: import numpy as np

In [3]: def make_timeseries(start="2000-01-01", end="2000-12-31", freq="1D", seed=None):
   ...:     index = pd.date_range(start=start, end=end, freq=freq, name="timestamp")
   ...:     n = len(index)
   ...:     state = np.random.RandomState(seed)
   ...:     columns = {
   ...:         "name": state.choice(["Alice", "Bob", "Charlie"], size=n),
   ...:         "id": state.poisson(1000, size=n),
   ...:         "x": state.rand(n) * 2 - 1,
   ...:         "y": state.rand(n) * 2 - 1,
   ...:     }
   ...:     df = pd.DataFrame(columns, index=index, columns=sorted(columns))
   ...:     if df.index[-1] == end:
   ...:         df = df.iloc[:-1]
   ...:     return df
   ...: 

In [4]: timeseries = [
   ...:     make_timeseries(freq="1min", seed=i).rename(columns=lambda x: f"{x}_{i}")
   ...:     for i in range(10)
   ...: ]
   ...: 

In [5]: ts_wide = pd.concat(timeseries, axis=1)

In [6]: ts_wide.head()
Out[6]: 
                     id_0 name_0       x_0  ...   name_9       x_9       y_9
timestamp                                   ...                             
2000-01-01 00:00:00   977  Alice -0.821225  ...  Charlie -0.957208 -0.757508
2000-01-01 00:01:00  1018    Bob -0.219182  ...    Alice -0.414445 -0.100298
2000-01-01 00:02:00   927  Alice  0.660908  ...  Charlie -0.325838  0.581859
2000-01-01 00:03:00   997    Bob -0.852458  ...      Bob  0.992033 -0.686692
2000-01-01 00:04:00   965    Bob  0.717283  ...  Charlie -0.924556 -0.184161

[5 rows x 40 columns]

In [7]: ts_wide.to_parquet("timeseries_wide.parquet")
```

Example 2 (json):
```json
In [8]: columns = ["id_0", "name_0", "x_0", "y_0"]

In [9]: pd.read_parquet("timeseries_wide.parquet")[columns]
Out[9]: 
                     id_0 name_0       x_0       y_0
timestamp                                           
2000-01-01 00:00:00   977  Alice -0.821225  0.906222
2000-01-01 00:01:00  1018    Bob -0.219182  0.350855
2000-01-01 00:02:00   927  Alice  0.660908 -0.798511
2000-01-01 00:03:00   997    Bob -0.852458  0.735260
2000-01-01 00:04:00   965    Bob  0.717283  0.393391
...                   ...    ...       ...       ...
2000-12-30 23:56:00  1037    Bob -0.814321  0.612836
2000-12-30 23:57:00   980    Bob  0.232195 -0.618828
2000-12-30 23:58:00   965  Alice -0.231131  0.026310
2000-12-30 23:59:00   984  Alice  0.942819  0.853128
2000-12-31 00:00:00  1003  Alice  0.201125 -0.136655

[525601 rows x 4 columns]
```

Example 3 (json):
```json
In [10]: pd.read_parquet("timeseries_wide.parquet", columns=columns)
Out[10]: 
                     id_0 name_0       x_0       y_0
timestamp                                           
2000-01-01 00:00:00   977  Alice -0.821225  0.906222
2000-01-01 00:01:00  1018    Bob -0.219182  0.350855
2000-01-01 00:02:00   927  Alice  0.660908 -0.798511
2000-01-01 00:03:00   997    Bob -0.852458  0.735260
2000-01-01 00:04:00   965    Bob  0.717283  0.393391
...                   ...    ...       ...       ...
2000-12-30 23:56:00  1037    Bob -0.814321  0.612836
2000-12-30 23:57:00   980    Bob  0.232195 -0.618828
2000-12-30 23:58:00   965  Alice -0.231131  0.026310
2000-12-30 23:59:00   984  Alice  0.942819  0.853128
2000-12-31 00:00:00  1003  Alice  0.201125 -0.136655

[525601 rows x 4 columns]
```

Example 4 (json):
```json
In [11]: ts = make_timeseries(freq="30s", seed=0)

In [12]: ts.to_parquet("timeseries.parquet")

In [13]: ts = pd.read_parquet("timeseries.parquet")

In [14]: ts
Out[14]: 
                       id     name         x         y
timestamp                                             
2000-01-01 00:00:00  1041    Alice  0.889987  0.281011
2000-01-01 00:00:30   988      Bob -0.455299  0.488153
2000-01-01 00:01:00  1018    Alice  0.096061  0.580473
2000-01-01 00:01:30   992      Bob  0.142482  0.041665
2000-01-01 00:02:00   960      Bob -0.036235  0.802159
...                   ...      ...       ...       ...
2000-12-30 23:58:00  1022    Alice  0.266191  0.875579
2000-12-30 23:58:30   974    Alice -0.009826  0.413686
2000-12-30 23:59:00  1028  Charlie  0.307108 -0.656789
2000-12-30 23:59:30  1002    Alice  0.202602  0.541335
2000-12-31 00:00:00   987    Alice  0.200832  0.615972

[1051201 rows x 4 columns]
```

---

## Merge, join, concatenate and compare#

**URL:** https://pandas.pydata.org/docs/user_guide/merging.html

**Contents:**
- Merge, join, concatenate and compare#
- concat()#
  - Joining logic of the resulting axis#
  - Ignoring indexes on the concatenation axis#
  - Concatenating Series and DataFrame together#
  - Resulting keys#
  - Appending rows to a DataFrame#
- merge()#
  - Merge types#
  - Merge key uniqueness#

pandas provides various methods for combining and comparing Series or DataFrame.

concat(): Merge multiple Series or DataFrame objects along a shared index or column

DataFrame.join(): Merge multiple DataFrame objects along the columns

DataFrame.combine_first(): Update missing values with non-missing values in the same location

merge(): Combine two Series or DataFrame objects with SQL-style joining

merge_ordered(): Combine two Series or DataFrame objects along an ordered axis

merge_asof(): Combine two Series or DataFrame objects by near instead of exact matching keys

Series.compare() and DataFrame.compare(): Show differences in values between two Series or DataFrame objects

The concat() function concatenates an arbitrary amount of Series or DataFrame objects along an axis while performing optional set logic (union or intersection) of the indexes on the other axes. Like numpy.concatenate, concat() takes a list or dict of homogeneously-typed objects and concatenates them.

concat() makes a full copy of the data, and iteratively reusing concat() can create unnecessary copies. Collect all DataFrame or Series objects in a list before using concat().

When concatenating DataFrame with named axes, pandas will attempt to preserve these index/column names whenever possible. In the case where all inputs share a common name, this name will be assigned to the result. When the input names do not all agree, the result will be unnamed. The same is true for MultiIndex, but the logic is applied separately on a level-by-level basis.

The join keyword specifies how to handle axis values that don’t exist in the first DataFrame.

join='outer' takes the union of all axis values

join='inner' takes the intersection of the axis values

To perform an effective “left” join using the exact index from the original DataFrame, result can be reindexed.

For DataFrame objects which don’t have a meaningful index, the ignore_index ignores overlapping indexes.

You can concatenate a mix of Series and DataFrame objects. The Series will be transformed to DataFrame with the column name as the name of the Series.

Unnamed Series will be numbered consecutively.

ignore_index=True will drop all name references.

The keys argument adds another axis level to the resulting index or column (creating a MultiIndex) associate specific keys with each original DataFrame.

The keys argument cane override the column names when creating a new DataFrame based on existing Series.

You can also pass a dict to concat() in which case the dict keys will be used for the keys argument unless other keys argument is specified:

The MultiIndex created has levels that are constructed from the passed keys and the index of the DataFrame pieces:

levels argument allows specifying resulting levels associated with the keys

If you have a Series that you want to append as a single row to a DataFrame, you can convert the row into a DataFrame and use concat()

merge() performs join operations similar to relational databases like SQL. Users who are familiar with SQL but new to pandas can reference a comparison with SQL.

merge() implements common SQL style joining operations.

one-to-one: joining two DataFrame objects on their indexes which must contain unique values.

many-to-one: joining a unique index to one or more columns in a different DataFrame.

many-to-many : joining columns on columns.

When joining columns on columns, potentially a many-to-many join, any indexes on the passed DataFrame objects will be discarded.

For a many-to-many join, if a key combination appears more than once in both tables, the DataFrame will have the Cartesian product of the associated data.

The how argument to merge() specifies which keys are included in the resulting table. If a key combination does not appear in either the left or right tables, the values in the joined table will be NA. Here is a summary of the how options and their SQL equivalent names:

Use keys from left frame only

Use keys from right frame only

Use union of keys from both frames

Use intersection of keys from both frames

Create the cartesian product of rows of both frames

You can Series and a DataFrame with a MultiIndex if the names of the MultiIndex correspond to the columns from the DataFrame. Transform the Series to a DataFrame using Series.reset_index() before merging

Performing an outer join with duplicate join keys in DataFrame

Merging on duplicate keys significantly increase the dimensions of the result and can cause a memory overflow.

The validate argument checks whether the uniqueness of merge keys. Key uniqueness is checked before merge operations and can protect against memory overflows and unexpected key duplication.

If the user is aware of the duplicates in the right DataFrame but wants to ensure there are no duplicates in the left DataFrame, one can use the validate='one_to_many' argument instead, which will not raise an exception.

merge() accepts the argument indicator. If True, a Categorical-type column called _merge will be added to the output object that takes on values:

Merge key only in 'left' frame

Merge key only in 'right' frame

Merge key in both frames

A string argument to indicator will use the value as the name for the indicator column.

The merge suffixes argument takes a tuple of list of strings to append to overlapping column names in the input DataFrame to disambiguate the result columns:

DataFrame.join() combines the columns of multiple, potentially differently-indexed DataFrame into a single result DataFrame.

DataFrame.join() takes an optional on argument which may be a column or multiple column names that the passed DataFrame is to be aligned.

To join on multiple keys, the passed DataFrame must have a MultiIndex:

The default for DataFrame.join is to perform a left join which uses only the keys found in the calling DataFrame. Other join types can be specified with how.

You can join a DataFrame with a Index to a DataFrame with a MultiIndex on a level. The name of the Index with match the level name of the MultiIndex.

The MultiIndex of the input argument must be completely used in the join and is a subset of the indices in the left argument.

Strings passed as the on, left_on, and right_on parameters may refer to either column names or index level names. This enables merging DataFrame instances on a combination of index levels and columns without resetting indexes.

When DataFrame are joined on a string that matches an index level in both arguments, the index level is preserved as an index level in the resulting DataFrame.

When DataFrame are joined using only some of the levels of a MultiIndex, the extra levels will be dropped from the resulting join. To preserve those levels, use DataFrame.reset_index() on those level names to move those levels to columns prior to the join.

A list or tuple of :class:`DataFrame` can also be passed to join() to join them together on their indexes.

DataFrame.combine_first() update missing values from one DataFrame with the non-missing values in another DataFrame in the corresponding location.

merge_ordered() combines order data such as numeric or time series data with optional filling of missing data with fill_method.

merge_asof() is similar to an ordered left-join except that mactches are on the nearest key rather than equal keys. For each row in the left DataFrame, the last row in the right DataFrame are selected where the on key is less than the left’s key. Both DataFrame must be sorted by the key.

Optionally an merge_asof() can perform a group-wise merge by matching the by key in addition to the nearest match on the on key.

merge_asof() within 2ms between the quote time and the trade time.

merge_asof() within 10ms between the quote time and the trade time and exclude exact matches on time. Note that though we exclude the exact matches (of the quotes), prior quotes do propagate to that point in time.

The Series.compare() and DataFrame.compare() methods allow you to compare two DataFrame or Series, respectively, and summarize their differences.

By default, if two corresponding values are equal, they will be shown as NaN. Furthermore, if all values in an entire row / column, the row / column will be omitted from the result. The remaining differences will be aligned on columns.

Stack the differences on rows.

Keep all original rows and columns with keep_shape=True

Keep all the original values even if they are equal.

**Examples:**

Example 1 (json):
```json
In [1]: df1 = pd.DataFrame(
   ...:     {
   ...:         "A": ["A0", "A1", "A2", "A3"],
   ...:         "B": ["B0", "B1", "B2", "B3"],
   ...:         "C": ["C0", "C1", "C2", "C3"],
   ...:         "D": ["D0", "D1", "D2", "D3"],
   ...:     },
   ...:     index=[0, 1, 2, 3],
   ...: )
   ...: 

In [2]: df2 = pd.DataFrame(
   ...:     {
   ...:         "A": ["A4", "A5", "A6", "A7"],
   ...:         "B": ["B4", "B5", "B6", "B7"],
   ...:         "C": ["C4", "C5", "C6", "C7"],
   ...:         "D": ["D4", "D5", "D6", "D7"],
   ...:     },
   ...:     index=[4, 5, 6, 7],
   ...: )
   ...: 

In [3]: df3 = pd.DataFrame(
   ...:     {
   ...:         "A": ["A8", "A9", "A10", "A11"],
   ...:         "B": ["B8", "B9", "B10", "B11"],
   ...:         "C": ["C8", "C9", "C10", "C11"],
   ...:         "D": ["D8", "D9", "D10", "D11"],
   ...:     },
   ...:     index=[8, 9, 10, 11],
   ...: )
   ...: 

In [4]: frames = [df1, df2, df3]

In [5]: result = pd.concat(frames)

In [6]: result
Out[6]: 
      A    B    C    D
0    A0   B0   C0   D0
1    A1   B1   C1   D1
2    A2   B2   C2   D2
3    A3   B3   C3   D3
4    A4   B4   C4   D4
5    A5   B5   C5   D5
6    A6   B6   C6   D6
7    A7   B7   C7   D7
8    A8   B8   C8   D8
9    A9   B9   C9   D9
10  A10  B10  C10  D10
11  A11  B11  C11  D11
```

Example 2 (bash):
```bash
frames = [process_your_file(f) for f in files]
result = pd.concat(frames)
```

Example 3 (json):
```json
In [7]: df4 = pd.DataFrame(
   ...:     {
   ...:         "B": ["B2", "B3", "B6", "B7"],
   ...:         "D": ["D2", "D3", "D6", "D7"],
   ...:         "F": ["F2", "F3", "F6", "F7"],
   ...:     },
   ...:     index=[2, 3, 6, 7],
   ...: )
   ...: 

In [8]: result = pd.concat([df1, df4], axis=1)

In [9]: result
Out[9]: 
     A    B    C    D    B    D    F
0   A0   B0   C0   D0  NaN  NaN  NaN
1   A1   B1   C1   D1  NaN  NaN  NaN
2   A2   B2   C2   D2   B2   D2   F2
3   A3   B3   C3   D3   B3   D3   F3
6  NaN  NaN  NaN  NaN   B6   D6   F6
7  NaN  NaN  NaN  NaN   B7   D7   F7
```

Example 4 (typescript):
```typescript
In [10]: result = pd.concat([df1, df4], axis=1, join="inner")

In [11]: result
Out[11]: 
    A   B   C   D   B   D   F
2  A2  B2  C2  D2  B2  D2  F2
3  A3  B3  C3  D3  B3  D3  F3
```

---

## Duplicate Labels#

**URL:** https://pandas.pydata.org/docs/user_guide/duplicates.html

**Contents:**
- Duplicate Labels#
- Consequences of Duplicate Labels#
- Duplicate Label Detection#
- Disallowing Duplicate Labels#
  - Duplicate Label Propagation#

Index objects are not required to be unique; you can have duplicate row or column labels. This may be a bit confusing at first. If you’re familiar with SQL, you know that row labels are similar to a primary key on a table, and you would never want duplicates in a SQL table. But one of pandas’ roles is to clean messy, real-world data before it goes to some downstream system. And real-world data has duplicates, even in fields that are supposed to be unique.

This section describes how duplicate labels change the behavior of certain operations, and how prevent duplicates from arising during operations, or to detect them if they do.

Some pandas methods (Series.reindex() for example) just don’t work with duplicates present. The output can’t be determined, and so pandas raises.

Other methods, like indexing, can give very surprising results. Typically indexing with a scalar will reduce dimensionality. Slicing a DataFrame with a scalar will return a Series. Slicing a Series with a scalar will return a scalar. But with duplicates, this isn’t the case.

We have duplicates in the columns. If we slice 'B', we get back a Series

But slicing 'A' returns a DataFrame

This applies to row labels as well

You can check whether an Index (storing the row or column labels) is unique with Index.is_unique:

Checking whether an index is unique is somewhat expensive for large datasets. pandas does cache this result, so re-checking on the same index is very fast.

Index.duplicated() will return a boolean ndarray indicating whether a label is repeated.

Which can be used as a boolean filter to drop duplicate rows.

If you need additional logic to handle duplicate labels, rather than just dropping the repeats, using groupby() on the index is a common trick. For example, we’ll resolve duplicates by taking the average of all rows with the same label.

Added in version 1.2.0.

As noted above, handling duplicates is an important feature when reading in raw data. That said, you may want to avoid introducing duplicates as part of a data processing pipeline (from methods like pandas.concat(), rename(), etc.). Both Series and DataFrame disallow duplicate labels by calling .set_flags(allows_duplicate_labels=False). (the default is to allow them). If there are duplicate labels, an exception will be raised.

This applies to both row and column labels for a DataFrame

This attribute can be checked or set with allows_duplicate_labels, which indicates whether that object can have duplicate labels.

DataFrame.set_flags() can be used to return a new DataFrame with attributes like allows_duplicate_labels set to some value

The new DataFrame returned is a view on the same data as the old DataFrame. Or the property can just be set directly on the same object

When processing raw, messy data you might initially read in the messy data (which potentially has duplicate labels), deduplicate, and then disallow duplicates going forward, to ensure that your data pipeline doesn’t introduce duplicates.

Setting allows_duplicate_labels=False on a Series or DataFrame with duplicate labels or performing an operation that introduces duplicate labels on a Series or DataFrame that disallows duplicates will raise an errors.DuplicateLabelError.

This error message contains the labels that are duplicated, and the numeric positions of all the duplicates (including the “original”) in the Series or DataFrame

In general, disallowing duplicates is “sticky”. It’s preserved through operations.

This is an experimental feature. Currently, many methods fail to propagate the allows_duplicate_labels value. In future versions it is expected that every method taking or returning one or more DataFrame or Series objects will propagate allows_duplicate_labels.

**Examples:**

Example 1 (typescript):
```typescript
In [1]: import pandas as pd

In [2]: import numpy as np
```

Example 2 (yaml):
```yaml
In [3]: s1 = pd.Series([0, 1, 2], index=["a", "b", "b"])

In [4]: s1.reindex(["a", "b", "c"])
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[4], line 1
----> 1 s1.reindex(["a", "b", "c"])

File ~/work/pandas/pandas/pandas/core/series.py:5172, in Series.reindex(self, index, axis, method, copy, level, fill_value, limit, tolerance)
   5155 @doc(
   5156     NDFrame.reindex,  # type: ignore[has-type]
   5157     klass=_shared_doc_kwargs["klass"],
   (...)
   5170     tolerance=None,
   5171 ) -> Series:
-> 5172     return super().reindex(
   5173         index=index,
   5174         method=method,
   5175         copy=copy,
   5176         level=level,
   5177         fill_value=fill_value,
   5178         limit=limit,
   5179         tolerance=tolerance,
   5180     )

File ~/work/pandas/pandas/pandas/core/generic.py:5632, in NDFrame.reindex(self, labels, index, columns, axis, method, copy, level, fill_value, limit, tolerance)
   5629     return self._reindex_multi(axes, copy, fill_value)
   5631 # perform the reindex on the axes
-> 5632 return self._reindex_axes(
   5633     axes, level, limit, tolerance, method, fill_value, copy
   5634 ).__finalize__(self, method="reindex")

File ~/work/pandas/pandas/pandas/core/generic.py:5655, in NDFrame._reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy)
   5652     continue
   5654 ax = self._get_axis(a)
-> 5655 new_index, indexer = ax.reindex(
   5656     labels, level=level, limit=limit, tolerance=tolerance, method=method
   5657 )
   5659 axis = self._get_axis_number(a)
   5660 obj = obj._reindex_with_indexers(
   5661     {axis: [new_index, indexer]},
   5662     fill_value=fill_value,
   5663     copy=copy,
   5664     allow_dups=False,
   5665 )

File ~/work/pandas/pandas/pandas/core/indexes/base.py:4436, in Index.reindex(self, target, method, level, limit, tolerance)
   4433     raise ValueError("cannot handle a non-unique multi-index!")
   4434 elif not self.is_unique:
   4435     # GH#42568
-> 4436     raise ValueError("cannot reindex on an axis with duplicate labels")
   4437 else:
   4438     indexer, _ = self.get_indexer_non_unique(target)

ValueError: cannot reindex on an axis with duplicate labels
```

Example 3 (typescript):
```typescript
In [5]: df1 = pd.DataFrame([[0, 1, 2], [3, 4, 5]], columns=["A", "A", "B"])

In [6]: df1
Out[6]: 
   A  A  B
0  0  1  2
1  3  4  5
```

Example 4 (yaml):
```yaml
In [7]: df1["B"]  # a series
Out[7]: 
0    2
1    5
Name: B, dtype: int64
```

---

## Indexing and selecting data#

**URL:** https://pandas.pydata.org/docs/user_guide/indexing.html

**Contents:**
- Indexing and selecting data#
- Different choices for indexing#
- Basics#
- Attribute access#
- Slicing ranges#
- Selection by label#
  - Slicing with labels#
- Selection by position#
- Selection by callable#
- Combining positional and label-based indexing#

The axis labeling information in pandas objects serves many purposes:

Identifies data (i.e. provides metadata) using known indicators, important for analysis, visualization, and interactive console display.

Enables automatic and explicit data alignment.

Allows intuitive getting and setting of subsets of the data set.

In this section, we will focus on the final point: namely, how to slice, dice, and generally get and set subsets of pandas objects. The primary focus will be on Series and DataFrame as they have received more development attention in this area.

The Python and NumPy indexing operators [] and attribute operator . provide quick and easy access to pandas data structures across a wide range of use cases. This makes interactive work intuitive, as there’s little new to learn if you already know how to deal with Python dictionaries and NumPy arrays. However, since the type of the data to be accessed isn’t known in advance, directly using standard operators has some optimization limits. For production code, we recommended that you take advantage of the optimized pandas data access methods exposed in this chapter.

Whether a copy or a reference is returned for a setting operation, may depend on the context. This is sometimes called chained assignment and should be avoided. See Returning a View versus Copy.

See the MultiIndex / Advanced Indexing for MultiIndex and more advanced indexing documentation.

See the cookbook for some advanced strategies.

Object selection has had a number of user-requested additions in order to support more explicit location based indexing. pandas now supports three types of multi-axis indexing.

.loc is primarily label based, but may also be used with a boolean array. .loc will raise KeyError when the items are not found. Allowed inputs are:

A single label, e.g. 5 or 'a' (Note that 5 is interpreted as a label of the index. This use is not an integer position along the index.).

A list or array of labels ['a', 'b', 'c'].

A slice object with labels 'a':'f' (Note that contrary to usual Python slices, both the start and the stop are included, when present in the index! See Slicing with labels and Endpoints are inclusive.)

A boolean array (any NA values will be treated as False).

A callable function with one argument (the calling Series or DataFrame) and that returns valid output for indexing (one of the above).

A tuple of row (and column) indices whose elements are one of the above inputs.

See more at Selection by Label.

.iloc is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array. .iloc will raise IndexError if a requested indexer is out-of-bounds, except slice indexers which allow out-of-bounds indexing. (this conforms with Python/NumPy slice semantics). Allowed inputs are:

A list or array of integers [4, 3, 0].

A slice object with ints 1:7.

A boolean array (any NA values will be treated as False).

A callable function with one argument (the calling Series or DataFrame) and that returns valid output for indexing (one of the above).

A tuple of row (and column) indices whose elements are one of the above inputs.

See more at Selection by Position, Advanced Indexing and Advanced Hierarchical.

.loc, .iloc, and also [] indexing can accept a callable as indexer. See more at Selection By Callable.

Destructuring tuple keys into row (and column) indexes occurs before callables are applied, so you cannot return a tuple from a callable to index both rows and columns.

Getting values from an object with multi-axes selection uses the following notation (using .loc as an example, but the following applies to .iloc as well). Any of the axes accessors may be the null slice :. Axes left out of the specification are assumed to be :, e.g. p.loc['a'] is equivalent to p.loc['a', :].

As mentioned when introducing the data structures in the last section, the primary function of indexing with [] (a.k.a. __getitem__ for those familiar with implementing class behavior in Python) is selecting out lower-dimensional slices. The following table shows return type values when indexing pandas objects with []:

Series corresponding to colname

Here we construct a simple time series data set to use for illustrating the indexing functionality:

None of the indexing functionality is time series specific unless specifically stated.

Thus, as per above, we have the most basic indexing using []:

You can pass a list of columns to [] to select columns in that order. If a column is not contained in the DataFrame, an exception will be raised. Multiple columns can also be set in this manner:

You may find this useful for applying a transform (in-place) to a subset of the columns.

pandas aligns all AXES when setting Series and DataFrame from .loc.

This will not modify df because the column alignment is before value assignment.

The correct way to swap column values is by using raw values:

However, pandas does not align AXES when setting Series and DataFrame from .iloc because .iloc operates by position.

This will modify df because the column alignment is not done before value assignment.

You may access an index on a Series or column on a DataFrame directly as an attribute:

You can use this access only if the index element is a valid Python identifier, e.g. s.1 is not allowed. See here for an explanation of valid identifiers.

The attribute will not be available if it conflicts with an existing method name, e.g. s.min is not allowed, but s['min'] is possible.

Similarly, the attribute will not be available if it conflicts with any of the following list: index, major_axis, minor_axis, items.

In any of these cases, standard indexing will still work, e.g. s['1'], s['min'], and s['index'] will access the corresponding element or column.

If you are using the IPython environment, you may also use tab-completion to see these accessible attributes.

You can also assign a dict to a row of a DataFrame:

You can use attribute access to modify an existing element of a Series or column of a DataFrame, but be careful; if you try to use attribute access to create a new column, it creates a new attribute rather than a new column and will this raise a UserWarning:

The most robust and consistent way of slicing ranges along arbitrary axes is described in the Selection by Position section detailing the .iloc method. For now, we explain the semantics of slicing using the [] operator.

With Series, the syntax works exactly as with an ndarray, returning a slice of the values and the corresponding labels:

Note that setting works as well:

With DataFrame, slicing inside of [] slices the rows. This is provided largely as a convenience since it is such a common operation.

Whether a copy or a reference is returned for a setting operation, may depend on the context. This is sometimes called chained assignment and should be avoided. See Returning a View versus Copy.

.loc is strict when you present slicers that are not compatible (or convertible) with the index type. For example using integers in a DatetimeIndex. These will raise a TypeError.

String likes in slicing can be convertible to the type of the index and lead to natural slicing.

pandas provides a suite of methods in order to have purely label based indexing. This is a strict inclusion based protocol. Every label asked for must be in the index, or a KeyError will be raised. When slicing, both the start bound AND the stop bound are included, if present in the index. Integers are valid labels, but they refer to the label and not the position.

The .loc attribute is the primary access method. The following are valid inputs:

A single label, e.g. 5 or 'a' (Note that 5 is interpreted as a label of the index. This use is not an integer position along the index.).

A list or array of labels ['a', 'b', 'c'].

A slice object with labels 'a':'f' (Note that contrary to usual Python slices, both the start and the stop are included, when present in the index! See Slicing with labels.

A callable, see Selection By Callable.

Note that setting works as well:

Accessing via label slices:

For getting a cross section using a label (equivalent to df.xs('a')):

For getting values with a boolean array:

NA values in a boolean array propagate as False:

For getting a value explicitly:

When using .loc with slices, if both the start and the stop labels are present in the index, then elements located between the two (including them) are returned:

If at least one of the two is absent, but the index is sorted, and can be compared against start and stop labels, then slicing will still work as expected, by selecting labels which rank between the two:

However, if at least one of the two is absent and the index is not sorted, an error will be raised (since doing otherwise would be computationally expensive, as well as potentially ambiguous for mixed type indexes). For instance, in the above example, s.loc[1:6] would raise KeyError.

For the rationale behind this behavior, see Endpoints are inclusive.

Also, if the index has duplicate labels and either the start or the stop label is duplicated, an error will be raised. For instance, in the above example, s.loc[2:5] would raise a KeyError.

For more information about duplicate labels, see Duplicate Labels.

Whether a copy or a reference is returned for a setting operation, may depend on the context. This is sometimes called chained assignment and should be avoided. See Returning a View versus Copy.

pandas provides a suite of methods in order to get purely integer based indexing. The semantics follow closely Python and NumPy slicing. These are 0-based indexing. When slicing, the start bound is included, while the upper bound is excluded. Trying to use a non-integer, even a valid label will raise an IndexError.

The .iloc attribute is the primary access method. The following are valid inputs:

A list or array of integers [4, 3, 0].

A slice object with ints 1:7.

A callable, see Selection By Callable.

A tuple of row (and column) indexes, whose elements are one of the above types.

Note that setting works as well:

Select via integer slicing:

Select via integer list:

For getting a cross section using an integer position (equiv to df.xs(1)):

Out of range slice indexes are handled gracefully just as in Python/NumPy.

Note that using slices that go out of bounds can result in an empty axis (e.g. an empty DataFrame being returned).

A single indexer that is out of bounds will raise an IndexError. A list of indexers where any element is out of bounds will raise an IndexError.

.loc, .iloc, and also [] indexing can accept a callable as indexer. The callable must be a function with one argument (the calling Series or DataFrame) that returns valid output for indexing.

For .iloc indexing, returning a tuple from the callable is not supported, since tuple destructuring for row and column indexes occurs before applying callables.

You can use callable indexing in Series.

Using these methods / indexers, you can chain data selection operations without using a temporary variable.

If you wish to get the 0th and the 2nd elements from the index in the ‘A’ column, you can do:

This can also be expressed using .iloc, by explicitly getting locations on the indexers, and using positional indexing to select things.

For getting multiple indexers, using .get_indexer:

The idiomatic way to achieve selecting potentially not-found elements is via .reindex(). See also the section on reindexing.

Alternatively, if you want to select only valid keys, the following is idiomatic and efficient; it is guaranteed to preserve the dtype of the selection.

Having a duplicated index will raise for a .reindex():

Generally, you can intersect the desired labels with the current axis, and then reindex.

However, this would still raise if your resulting index is duplicated.

A random selection of rows or columns from a Series or DataFrame with the sample() method. The method will sample rows by default, and accepts a specific number of rows/columns to return, or a fraction of rows.

By default, sample will return each row at most once, but one can also sample with replacement using the replace option:

By default, each row has an equal probability of being selected, but if you want rows to have different probabilities, you can pass the sample function sampling weights as weights. These weights can be a list, a NumPy array, or a Series, but they must be of the same length as the object you are sampling. Missing values will be treated as a weight of zero, and inf values are not allowed. If weights do not sum to 1, they will be re-normalized by dividing all weights by the sum of the weights. For example:

When applied to a DataFrame, you can use a column of the DataFrame as sampling weights (provided you are sampling rows and not columns) by simply passing the name of the column as a string.

sample also allows users to sample columns instead of rows using the axis argument.

Finally, one can also set a seed for sample’s random number generator using the random_state argument, which will accept either an integer (as a seed) or a NumPy RandomState object.

The .loc/[] operations can perform enlargement when setting a non-existent key for that axis.

In the Series case this is effectively an appending operation.

A DataFrame can be enlarged on either axis via .loc.

This is like an append operation on the DataFrame.

Since indexing with [] must handle a lot of cases (single-label access, slicing, boolean indexing, etc.), it has a bit of overhead in order to figure out what you’re asking for. If you only want to access a scalar value, the fastest way is to use the at and iat methods, which are implemented on all of the data structures.

Similarly to loc, at provides label based scalar lookups, while, iat provides integer based lookups analogously to iloc

You can also set using these same indexers.

at may enlarge the object in-place as above if the indexer is missing.

Another common operation is the use of boolean vectors to filter the data. The operators are: | for or, & for and, and ~ for not. These must be grouped by using parentheses, since by default Python will evaluate an expression such as df['A'] > 2 & df['B'] < 3 as df['A'] > (2 & df['B']) < 3, while the desired evaluation order is (df['A'] > 2) & (df['B'] < 3).

Using a boolean vector to index a Series works exactly as in a NumPy ndarray:

You may select rows from a DataFrame using a boolean vector the same length as the DataFrame’s index (for example, something derived from one of the columns of the DataFrame):

List comprehensions and the map method of Series can also be used to produce more complex criteria:

With the choice methods Selection by Label, Selection by Position, and Advanced Indexing you may select along more than one axis using boolean vectors combined with other indexing expressions.

iloc supports two kinds of boolean indexing. If the indexer is a boolean Series, an error will be raised. For instance, in the following example, df.iloc[s.values, 1] is ok. The boolean indexer is an array. But df.iloc[s, 1] would raise ValueError.

Consider the isin() method of Series, which returns a boolean vector that is true wherever the Series elements exist in the passed list. This allows you to select rows where one or more columns have values you want:

The same method is available for Index objects and is useful for the cases when you don’t know which of the sought labels are in fact present:

In addition to that, MultiIndex allows selecting a separate level to use in the membership check:

DataFrame also has an isin() method. When calling isin, pass a set of values as either an array or dict. If values is an array, isin returns a DataFrame of booleans that is the same shape as the original DataFrame, with True wherever the element is in the sequence of values.

Oftentimes you’ll want to match certain values with certain columns. Just make values a dict where the key is the column, and the value is a list of items you want to check for.

To return the DataFrame of booleans where the values are not in the original DataFrame, use the ~ operator:

Combine DataFrame’s isin with the any() and all() methods to quickly select subsets of your data that meet a given criteria. To select a row where each column meets its own criterion:

Selecting values from a Series with a boolean vector generally returns a subset of the data. To guarantee that selection output has the same shape as the original data, you can use the where method in Series and DataFrame.

To return only the selected rows:

To return a Series of the same shape as the original:

Selecting values from a DataFrame with a boolean criterion now also preserves input data shape. where is used under the hood as the implementation. The code below is equivalent to df.where(df < 0).

In addition, where takes an optional other argument for replacement of values where the condition is False, in the returned copy.

You may wish to set values based on some boolean criteria. This can be done intuitively like so:

where returns a modified copy of the data.

The signature for DataFrame.where() differs from numpy.where(). Roughly df1.where(m, df2) is equivalent to np.where(m, df1, df2).

Furthermore, where aligns the input boolean condition (ndarray or DataFrame), such that partial selection with setting is possible. This is analogous to partial setting via .loc (but on the contents rather than the axis labels).

Where can also accept axis and level parameters to align the input when performing the where.

This is equivalent to (but faster than) the following.

where can accept a callable as condition and other arguments. The function must be with one argument (the calling Series or DataFrame) and that returns valid output as condition and other argument.

mask() is the inverse boolean operation of where.

An alternative to where() is to use numpy.where(). Combined with setting a new column, you can use it to enlarge a DataFrame where the values are determined conditionally.

Consider you have two choices to choose from in the following DataFrame. And you want to set a new column color to ‘green’ when the second column has ‘Z’. You can do the following:

If you have multiple conditions, you can use numpy.select() to achieve that. Say corresponding to three conditions there are three choice of colors, with a fourth color as a fallback, you can do the following.

DataFrame objects have a query() method that allows selection using an expression.

You can get the value of the frame where column b has values between the values of columns a and c. For example:

Do the same thing but fall back on a named index if there is no column with the name a.

If instead you don’t want to or cannot name your index, you can use the name index in your query expression:

If the name of your index overlaps with a column name, the column name is given precedence. For example,

You can still use the index in a query expression by using the special identifier ‘index’:

If for some reason you have a column named index, then you can refer to the index as ilevel_0 as well, but at this point you should consider renaming your columns to something less ambiguous.

You can also use the levels of a DataFrame with a MultiIndex as if they were columns in the frame:

If the levels of the MultiIndex are unnamed, you can refer to them using special names:

The convention is ilevel_0, which means “index level 0” for the 0th level of the index.

A use case for query() is when you have a collection of DataFrame objects that have a subset of column names (or index levels/names) in common. You can pass the same query to both frames without having to specify which frame you’re interested in querying

Full numpy-like syntax:

Slightly nicer by removing the parentheses (comparison operators bind tighter than & and |):

Use English instead of symbols:

Pretty close to how you might write it on paper:

query() also supports special use of Python’s in and not in comparison operators, providing a succinct syntax for calling the isin method of a Series or DataFrame.

You can combine this with other expressions for very succinct queries:

Note that in and not in are evaluated in Python, since numexpr has no equivalent of this operation. However, only the in/not in expression itself is evaluated in vanilla Python. For example, in the expression

(b + c + d) is evaluated by numexpr and then the in operation is evaluated in plain Python. In general, any operations that can be evaluated using numexpr will be.

Comparing a list of values to a column using ==/!= works similarly to in/not in.

You can negate boolean expressions with the word not or the ~ operator.

Of course, expressions can be arbitrarily complex too:

DataFrame.query() using numexpr is slightly faster than Python for large frames.

You will only see the performance benefits of using the numexpr engine with DataFrame.query() if your frame has more than approximately 100,000 rows.

This plot was created using a DataFrame with 3 columns each containing floating point values generated using numpy.random.randn().

If you want to identify and remove duplicate rows in a DataFrame, there are two methods that will help: duplicated and drop_duplicates. Each takes as an argument the columns to use to identify duplicated rows.

duplicated returns a boolean vector whose length is the number of rows, and which indicates whether a row is duplicated.

drop_duplicates removes duplicate rows.

By default, the first observed row of a duplicate set is considered unique, but each method has a keep parameter to specify targets to be kept.

keep='first' (default): mark / drop duplicates except for the first occurrence.

keep='last': mark / drop duplicates except for the last occurrence.

keep=False: mark / drop all duplicates.

Also, you can pass a list of columns to identify duplications.

To drop duplicates by index value, use Index.duplicated then perform slicing. The same set of options are available for the keep parameter.

Each of Series or DataFrame have a get method which can return a default value.

Sometimes you want to extract a set of values given a sequence of row labels and column labels, this can be achieved by pandas.factorize and NumPy indexing. For instance:

Formerly this could be achieved with the dedicated DataFrame.lookup method which was deprecated in version 1.2.0 and removed in version 2.0.0.

The pandas Index class and its subclasses can be viewed as implementing an ordered multiset. Duplicates are allowed.

Index also provides the infrastructure necessary for lookups, data alignment, and reindexing. The easiest way to create an Index directly is to pass a list or other sequence to Index:

If no dtype is given, Index tries to infer the dtype from the data. It is also possible to give an explicit dtype when instantiating an Index:

You can also pass a name to be stored in the index:

The name, if set, will be shown in the console display:

Indexes are “mostly immutable”, but it is possible to set and change their name attribute. You can use the rename, set_names to set these attributes directly, and they default to returning a copy.

See Advanced Indexing for usage of MultiIndexes.

set_names, set_levels, and set_codes also take an optional level argument

The two main operations are union and intersection. Difference is provided via the .difference() method.

Also available is the symmetric_difference operation, which returns elements that appear in either idx1 or idx2, but not in both. This is equivalent to the Index created by idx1.difference(idx2).union(idx2.difference(idx1)), with duplicates dropped.

The resulting index from a set operation will be sorted in ascending order.

When performing Index.union() between indexes with different dtypes, the indexes must be cast to a common dtype. Typically, though not always, this is object dtype. The exception is when performing a union between integer and float data. In this case, the integer values are converted to float

Even though Index can hold missing values (NaN), it should be avoided if you do not want any unexpected results. For example, some operations exclude missing values implicitly.

Index.fillna fills missing values with specified scalar value.

Occasionally you will load or create a data set into a DataFrame and want to add an index after you’ve already done so. There are a couple of different ways.

DataFrame has a set_index() method which takes a column name (for a regular Index) or a list of column names (for a MultiIndex). To create a new, re-indexed DataFrame:

The append keyword option allow you to keep the existing index and append the given columns to a MultiIndex:

Other options in set_index allow you not drop the index columns.

As a convenience, there is a new function on DataFrame called reset_index() which transfers the index values into the DataFrame’s columns and sets a simple integer index. This is the inverse operation of set_index().

The output is more similar to a SQL table or a record array. The names for the columns derived from the index are the ones stored in the names attribute.

You can use the level keyword to remove only a portion of the index:

reset_index takes an optional parameter drop which if true simply discards the index, instead of putting index values in the DataFrame’s columns.

You can assign a custom index to the index attribute:

Copy-on-Write will become the new default in pandas 3.0. This means that chained indexing will never work. As a consequence, the SettingWithCopyWarning won’t be necessary anymore. See this section for more context. We recommend turning Copy-on-Write on to leverage the improvements with

` pd.options.mode.copy_on_write = True `

even before pandas 3.0 is available.

When setting values in a pandas object, care must be taken to avoid what is called chained indexing. Here is an example.

Compare these two access methods:

These both yield the same results, so which should you use? It is instructive to understand the order of operations on these and why method 2 (.loc) is much preferred over method 1 (chained []).

dfmi['one'] selects the first level of the columns and returns a DataFrame that is singly-indexed. Then another Python operation dfmi_with_one['second'] selects the series indexed by 'second'. This is indicated by the variable dfmi_with_one because pandas sees these operations as separate events. e.g. separate calls to __getitem__, so it has to treat them as linear operations, they happen one after another.

Contrast this to df.loc[:,('one','second')] which passes a nested tuple of (slice(None),('one','second')) to a single call to __getitem__. This allows pandas to deal with this as a single entity. Furthermore this order of operations can be significantly faster, and allows one to index both axes if so desired.

Copy-on-Write will become the new default in pandas 3.0. This means than chained indexing will never work. As a consequence, the SettingWithCopyWarning won’t be necessary anymore. See this section for more context. We recommend turning Copy-on-Write on to leverage the improvements with

` pd.options.mode.copy_on_write = True `

even before pandas 3.0 is available.

The problem in the previous section is just a performance issue. What’s up with the SettingWithCopy warning? We don’t usually throw warnings around when you do something that might cost a few extra milliseconds!

But it turns out that assigning to the product of chained indexing has inherently unpredictable results. To see this, think about how the Python interpreter executes this code:

But this code is handled differently:

See that __getitem__ in there? Outside of simple cases, it’s very hard to predict whether it will return a view or a copy (it depends on the memory layout of the array, about which pandas makes no guarantees), and therefore whether the __setitem__ will modify dfmi or a temporary object that gets thrown out immediately afterward. That’s what SettingWithCopy is warning you about!

You may be wondering whether we should be concerned about the loc property in the first example. But dfmi.loc is guaranteed to be dfmi itself with modified indexing behavior, so dfmi.loc.__getitem__ / dfmi.loc.__setitem__ operate on dfmi directly. Of course, dfmi.loc.__getitem__(idx) may be a view or a copy of dfmi.

Sometimes a SettingWithCopy warning will arise at times when there’s no obvious chained indexing going on. These are the bugs that SettingWithCopy is designed to catch! pandas is probably trying to warn you that you’ve done this:

Copy-on-Write will become the new default in pandas 3.0. This means than chained indexing will never work. As a consequence, the SettingWithCopyWarning won’t be necessary anymore. See this section for more context. We recommend turning Copy-on-Write on to leverage the improvements with

` pd.options.mode.copy_on_write = True `

even before pandas 3.0 is available.

When you use chained indexing, the order and type of the indexing operation partially determine whether the result is a slice into the original object, or a copy of the slice.

pandas has the SettingWithCopyWarning because assigning to a copy of a slice is frequently not intentional, but a mistake caused by chained indexing returning a copy where a slice was expected.

If you would like pandas to be more or less trusting about assignment to a chained indexing expression, you can set the option mode.chained_assignment to one of these values:

'warn', the default, means a SettingWithCopyWarning is printed.

'raise' means pandas will raise a SettingWithCopyError you have to deal with.

None will suppress the warnings entirely.

This however is operating on a copy and will not work.

A chained assignment can also crop up in setting in a mixed dtype frame.

These setting rules apply to all of .loc/.iloc.

The following is the recommended access method using .loc for multiple items (using mask) and a single item using a fixed index:

The following can work at times, but it is not guaranteed to, and therefore should be avoided:

Last, the subsequent example will not work at all, and so should be avoided:

The chained assignment warnings / exceptions are aiming to inform the user of a possibly invalid assignment. There may be false positives; situations where a chained assignment is inadvertently reported.

**Examples:**

Example 1 (typescript):
```typescript
In [1]: ser = pd.Series(range(5), index=list("abcde"))

In [2]: ser.loc[["a", "c", "e"]]
Out[2]: 
a    0
c    2
e    4
dtype: int64

In [3]: df = pd.DataFrame(np.arange(25).reshape(5, 5), index=list("abcde"), columns=list("abcde"))

In [4]: df.loc[["a", "c", "e"], ["b", "d"]]
Out[4]: 
    b   d
a   1   3
c  11  13
e  21  23
```

Example 2 (typescript):
```typescript
In [5]: dates = pd.date_range('1/1/2000', periods=8)

In [6]: df = pd.DataFrame(np.random.randn(8, 4),
   ...:                   index=dates, columns=['A', 'B', 'C', 'D'])
   ...: 

In [7]: df
Out[7]: 
                   A         B         C         D
2000-01-01  0.469112 -0.282863 -1.509059 -1.135632
2000-01-02  1.212112 -0.173215  0.119209 -1.044236
2000-01-03 -0.861849 -2.104569 -0.494929  1.071804
2000-01-04  0.721555 -0.706771 -1.039575  0.271860
2000-01-05 -0.424972  0.567020  0.276232 -1.087401
2000-01-06 -0.673690  0.113648 -1.478427  0.524988
2000-01-07  0.404705  0.577046 -1.715002 -1.039268
2000-01-08 -0.370647 -1.157892 -1.344312  0.844885
```

Example 3 (typescript):
```typescript
In [8]: s = df['A']

In [9]: s[dates[5]]
Out[9]: -0.6736897080883706
```

Example 4 (unknown):
```unknown
In [10]: df
Out[10]: 
                   A         B         C         D
2000-01-01  0.469112 -0.282863 -1.509059 -1.135632
2000-01-02  1.212112 -0.173215  0.119209 -1.044236
2000-01-03 -0.861849 -2.104569 -0.494929  1.071804
2000-01-04  0.721555 -0.706771 -1.039575  0.271860
2000-01-05 -0.424972  0.567020  0.276232 -1.087401
2000-01-06 -0.673690  0.113648 -1.478427  0.524988
2000-01-07  0.404705  0.577046 -1.715002 -1.039268
2000-01-08 -0.370647 -1.157892 -1.344312  0.844885

In [11]: df[['B', 'A']] = df[['A', 'B']]

In [12]: df
Out[12]: 
                   A         B         C         D
2000-01-01 -0.282863  0.469112 -1.509059 -1.135632
2000-01-02 -0.173215  1.212112  0.119209 -1.044236
2000-01-03 -2.104569 -0.861849 -0.494929  1.071804
2000-01-04 -0.706771  0.721555 -1.039575  0.271860
2000-01-05  0.567020 -0.424972  0.276232 -1.087401
2000-01-06  0.113648 -0.673690 -1.478427  0.524988
2000-01-07  0.577046  0.404705 -1.715002 -1.039268
2000-01-08 -1.157892 -0.370647 -1.344312  0.844885
```

---

## 

**URL:** https://pandas.pydata.org/docs/_sources/user_guide/indexing.rst.txt

---

## 

**URL:** https://pandas.pydata.org/docs/_sources/user_guide/pyarrow.rst.txt

---

## Nullable Boolean data type#

**URL:** https://pandas.pydata.org/docs/user_guide/boolean.html

**Contents:**
- Nullable Boolean data type#
- Indexing with NA values#
- Kleene logical operations#

BooleanArray is currently experimental. Its API or implementation may change without warning.

pandas allows indexing with NA values in a boolean array, which are treated as False.

If you would prefer to keep the NA values you can manually fill them with fillna(True).

arrays.BooleanArray implements Kleene Logic (sometimes called three-value logic) for logical operations like & (and), | (or) and ^ (exclusive-or).

This table demonstrates the results for every combination. These operations are symmetrical, so flipping the left- and right-hand side makes no difference in the result.

When an NA is present in an operation, the output value is NA only if the result cannot be determined solely based on the other input. For example, True | NA is True, because both True | True and True | False are True. In that case, we don’t actually need to consider the value of the NA.

On the other hand, True & NA is NA. The result depends on whether the NA really is True or False, since True & True is True, but True & False is False, so we can’t determine the output.

This differs from how np.nan behaves in logical operations. pandas treated np.nan is always false in the output.

**Examples:**

Example 1 (typescript):
```typescript
In [1]: s = pd.Series([1, 2, 3])

In [2]: mask = pd.array([True, False, pd.NA], dtype="boolean")

In [3]: s[mask]
Out[3]: 
0    1
dtype: int64
```

Example 2 (yaml):
```yaml
In [4]: s[mask.fillna(True)]
Out[4]: 
0    1
2    3
dtype: int64
```

Example 3 (yaml):
```yaml
In [5]: pd.Series([True, False, np.nan], dtype="object") | True
Out[5]: 
0     True
1     True
2    False
dtype: bool

In [6]: pd.Series([True, False, np.nan], dtype="boolean") | True
Out[6]: 
0    True
1    True
2    True
dtype: boolean
```

Example 4 (yaml):
```yaml
In [7]: pd.Series([True, False, np.nan], dtype="object") & True
Out[7]: 
0     True
1    False
2    False
dtype: bool

In [8]: pd.Series([True, False, np.nan], dtype="boolean") & True
Out[8]: 
0     True
1    False
2     <NA>
dtype: boolean
```

---

## Copy-on-Write (CoW)#

**URL:** https://pandas.pydata.org/docs/user_guide/copy_on_write.html

**Contents:**
- Copy-on-Write (CoW)#
- Previous behavior#
- Migrating to Copy-on-Write#
- Description#
- Chained Assignment#
- Read-only NumPy arrays#
- Patterns to avoid#
- Copy-on-Write optimizations#
- How to enable CoW#

Copy-on-Write will become the default in pandas 3.0. We recommend turning it on now to benefit from all improvements.

Copy-on-Write was first introduced in version 1.5.0. Starting from version 2.0 most of the optimizations that become possible through CoW are implemented and supported. All possible optimizations are supported starting from pandas 2.1.

CoW will be enabled by default in version 3.0.

CoW will lead to more predictable behavior since it is not possible to update more than one object with one statement, e.g. indexing operations or methods won’t have side-effects. Additionally, through delaying copies as long as possible, the average performance and memory usage will improve.

pandas indexing behavior is tricky to understand. Some operations return views while other return copies. Depending on the result of the operation, mutating one object might accidentally mutate another:

Mutating subset, e.g. updating its values, also updates df. The exact behavior is hard to predict. Copy-on-Write solves accidentally modifying more than one object, it explicitly disallows this. With CoW enabled, df is unchanged:

The following sections will explain what this means and how it impacts existing applications.

Copy-on-Write will be the default and only mode in pandas 3.0. This means that users need to migrate their code to be compliant with CoW rules.

The default mode in pandas will raise warnings for certain cases that will actively change behavior and thus change user intended behavior.

We added another mode, e.g.

that will warn for every operation that will change behavior with CoW. We expect this mode to be very noisy, since many cases that we don’t expect that they will influence users will also emit a warning. We recommend checking this mode and analyzing the warnings, but it is not necessary to address all of these warning. The first two items of the following lists are the only cases that need to be addressed to make existing code work with CoW.

The following few items describe the user visible changes:

Chained assignment will never work

loc should be used as an alternative. Check the chained assignment section for more details.

Accessing the underlying array of a pandas object will return a read-only view

This example returns a NumPy array that is a view of the Series object. This view can be modified and thus also modify the pandas object. This is not compliant with CoW rules. The returned array is set to non-writeable to protect against this behavior. Creating a copy of this array allows modification. You can also make the array writeable again if you don’t care about the pandas object anymore.

See the section about read-only NumPy arrays for more details.

Only one pandas object is updated at once

The following code snippet updates both df and subset without CoW:

This won’t be possible anymore with CoW, since the CoW rules explicitly forbid this. This includes updating a single column as a Series and relying on the change propagating back to the parent DataFrame. This statement can be rewritten into a single statement with loc or iloc if this behavior is necessary. DataFrame.where() is another suitable alternative for this case.

Updating a column selected from a DataFrame with an inplace method will also not work anymore.

This is another form of chained assignment. This can generally be rewritten in 2 different forms:

A different alternative would be to not use inplace:

Constructors now copy NumPy arrays by default

The Series and DataFrame constructors will now copy NumPy array by default when not otherwise specified. This was changed to avoid mutating a pandas object when the NumPy array is changed inplace outside of pandas. You can set copy=False to avoid this copy.

CoW means that any DataFrame or Series derived from another in any way always behaves as a copy. As a consequence, we can only change the values of an object through modifying the object itself. CoW disallows updating a DataFrame or a Series that shares data with another DataFrame or Series object inplace.

This avoids side-effects when modifying values and hence, most methods can avoid actually copying the data and only trigger a copy when necessary.

The following example will operate inplace with CoW:

The object df does not share any data with any other object and hence no copy is triggered when updating the values. In contrast, the following operation triggers a copy of the data under CoW:

reset_index returns a lazy copy with CoW while it copies the data without CoW. Since both objects, df and df2 share the same data, a copy is triggered when modifying df2. The object df still has the same values as initially while df2 was modified.

If the object df isn’t needed anymore after performing the reset_index operation, you can emulate an inplace-like operation through assigning the output of reset_index to the same variable:

The initial object gets out of scope as soon as the result of reset_index is reassigned and hence df does not share data with any other object. No copy is necessary when modifying the object. This is generally true for all methods listed in Copy-on-Write optimizations.

Previously, when operating on views, the view and the parent object was modified:

CoW triggers a copy when df is changed to avoid mutating view as well:

Chained assignment references a technique where an object is updated through two subsequent indexing operations, e.g.

The column foo is updated where the column bar is greater than 5. This violates the CoW principles though, because it would have to modify the view df["foo"] and df in one step. Hence, chained assignment will consistently never work and raise a ChainedAssignmentError warning with CoW enabled:

With copy on write this can be done by using loc.

Accessing the underlying NumPy array of a DataFrame will return a read-only array if the array shares data with the initial DataFrame:

The array is a copy if the initial DataFrame consists of more than one array:

The array shares data with the DataFrame if the DataFrame consists of only one NumPy array:

This array is read-only, which means that it can’t be modified inplace:

The same holds true for a Series, since a Series always consists of a single array.

There are two potential solution to this:

Trigger a copy manually if you want to avoid updating DataFrames that share memory with your array.

Make the array writeable. This is a more performant solution but circumvents Copy-on-Write rules, so it should be used with caution.

No defensive copy will be performed if two objects share the same data while you are modifying one object inplace.

This creates two objects that share data and thus the setitem operation will trigger a copy. This is not necessary if the initial object df isn’t needed anymore. Simply reassigning to the same variable will invalidate the reference that is held by the object.

No copy is necessary in this example. Creating multiple references keeps unnecessary references alive and thus will hurt performance with Copy-on-Write.

A new lazy copy mechanism that defers the copy until the object in question is modified and only if this object shares data with another object. This mechanism was added to methods that don’t require a copy of the underlying data. Popular examples are DataFrame.drop() for axis=1 and DataFrame.rename().

These methods return views when Copy-on-Write is enabled, which provides a significant performance improvement compared to the regular execution.

Copy-on-Write can be enabled through the configuration option copy_on_write. The option can be turned on __globally__ through either of the following:

**Examples:**

Example 1 (json):
```json
In [1]: df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})

In [2]: subset = df["foo"]

In [3]: subset.iloc[0] = 100

In [4]: df
Out[4]: 
   foo  bar
0  100    4
1    2    5
2    3    6
```

Example 2 (json):
```json
In [5]: pd.options.mode.copy_on_write = True

In [6]: df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})

In [7]: subset = df["foo"]

In [8]: subset.iloc[0] = 100

In [9]: df
Out[9]: 
   foo  bar
0    1    4
1    2    5
2    3    6
```

Example 3 (unknown):
```unknown
pd.options.mode.copy_on_write = "warn"
```

Example 4 (typescript):
```typescript
In [10]: ser = pd.Series([1, 2, 3])

In [11]: ser.to_numpy()
Out[11]: array([1, 2, 3])
```

---

## MultiIndex / advanced indexing#

**URL:** https://pandas.pydata.org/docs/user_guide/advanced.html

**Contents:**
- MultiIndex / advanced indexing#
- Hierarchical indexing (MultiIndex)#
  - Creating a MultiIndex (hierarchical index) object#
  - Reconstructing the level labels#
  - Basic indexing on axis with MultiIndex#
  - Defined levels#
  - Data alignment and using reindex#
- Advanced indexing with hierarchical index#
  - Using slicers#
  - Cross-section#

This section covers indexing with a MultiIndex and other advanced indexing features.

See the Indexing and Selecting Data for general indexing documentation.

Whether a copy or a reference is returned for a setting operation may depend on the context. This is sometimes called chained assignment and should be avoided. See Returning a View versus Copy.

See the cookbook for some advanced strategies.

Hierarchical / Multi-level indexing is very exciting as it opens the door to some quite sophisticated data analysis and manipulation, especially for working with higher dimensional data. In essence, it enables you to store and manipulate data with an arbitrary number of dimensions in lower dimensional data structures like Series (1d) and DataFrame (2d).

In this section, we will show what exactly we mean by “hierarchical” indexing and how it integrates with all of the pandas indexing functionality described above and in prior sections. Later, when discussing group by and pivoting and reshaping data, we’ll show non-trivial applications to illustrate how it aids in structuring data for analysis.

See the cookbook for some advanced strategies.

The MultiIndex object is the hierarchical analogue of the standard Index object which typically stores the axis labels in pandas objects. You can think of MultiIndex as an array of tuples where each tuple is unique. A MultiIndex can be created from a list of arrays (using MultiIndex.from_arrays()), an array of tuples (using MultiIndex.from_tuples()), a crossed set of iterables (using MultiIndex.from_product()), or a DataFrame (using MultiIndex.from_frame()). The Index constructor will attempt to return a MultiIndex when it is passed a list of tuples. The following examples demonstrate different ways to initialize MultiIndexes.

When you want every pairing of the elements in two iterables, it can be easier to use the MultiIndex.from_product() method:

You can also construct a MultiIndex from a DataFrame directly, using the method MultiIndex.from_frame(). This is a complementary method to MultiIndex.to_frame().

As a convenience, you can pass a list of arrays directly into Series or DataFrame to construct a MultiIndex automatically:

All of the MultiIndex constructors accept a names argument which stores string names for the levels themselves. If no names are provided, None will be assigned:

This index can back any axis of a pandas object, and the number of levels of the index is up to you:

We’ve “sparsified” the higher levels of the indexes to make the console output a bit easier on the eyes. Note that how the index is displayed can be controlled using the multi_sparse option in pandas.set_options():

It’s worth keeping in mind that there’s nothing preventing you from using tuples as atomic labels on an axis:

The reason that the MultiIndex matters is that it can allow you to do grouping, selection, and reshaping operations as we will describe below and in subsequent areas of the documentation. As you will see in later sections, you can find yourself working with hierarchically-indexed data without creating a MultiIndex explicitly yourself. However, when loading data from a file, you may wish to generate your own MultiIndex when preparing the data set.

The method get_level_values() will return a vector of the labels for each location at a particular level:

One of the important features of hierarchical indexing is that you can select data by a “partial” label identifying a subgroup in the data. Partial selection “drops” levels of the hierarchical index in the result in a completely analogous way to selecting a column in a regular DataFrame:

See Cross-section with hierarchical index for how to select on a deeper level.

The MultiIndex keeps all the defined levels of an index, even if they are not actually used. When slicing an index, you may notice this. For example:

This is done to avoid a recomputation of the levels in order to make slicing highly performant. If you want to see only the used levels, you can use the get_level_values() method.

To reconstruct the MultiIndex with only the used levels, the remove_unused_levels() method may be used.

Operations between differently-indexed objects having MultiIndex on the axes will work as you expect; data alignment will work the same as an Index of tuples:

The reindex() method of Series/DataFrames can be called with another MultiIndex, or even a list or array of tuples:

Syntactically integrating MultiIndex in advanced indexing with .loc is a bit challenging, but we’ve made every effort to do so. In general, MultiIndex keys take the form of tuples. For example, the following works as you would expect:

Note that df.loc['bar', 'two'] would also work in this example, but this shorthand notation can lead to ambiguity in general.

If you also want to index a specific column with .loc, you must use a tuple like this:

You don’t have to specify all levels of the MultiIndex by passing only the first elements of the tuple. For example, you can use “partial” indexing to get all elements with bar in the first level as follows:

This is a shortcut for the slightly more verbose notation df.loc[('bar',),] (equivalent to df.loc['bar',] in this example).

“Partial” slicing also works quite nicely.

You can slice with a ‘range’ of values, by providing a slice of tuples.

Passing a list of labels or tuples works similar to reindexing:

It is important to note that tuples and lists are not treated identically in pandas when it comes to indexing. Whereas a tuple is interpreted as one multi-level key, a list is used to specify several keys. Or in other words, tuples go horizontally (traversing levels), lists go vertically (scanning levels).

Importantly, a list of tuples indexes several complete MultiIndex keys, whereas a tuple of lists refer to several values within a level:

You can slice a MultiIndex by providing multiple indexers.

You can provide any of the selectors as if you are indexing by label, see Selection by Label, including slices, lists of labels, labels, and boolean indexers.

You can use slice(None) to select all the contents of that level. You do not need to specify all the deeper levels, they will be implied as slice(None).

As usual, both sides of the slicers are included as this is label indexing.

You should specify all axes in the .loc specifier, meaning the indexer for the index and for the columns. There are some ambiguous cases where the passed indexer could be misinterpreted as indexing both axes, rather than into say the MultiIndex for the rows.

You should not do this:

Basic MultiIndex slicing using slices, lists, and labels.

You can use pandas.IndexSlice to facilitate a more natural syntax using :, rather than using slice(None).

It is possible to perform quite complicated selections using this method on multiple axes at the same time.

Using a boolean indexer you can provide selection related to the values.

You can also specify the axis argument to .loc to interpret the passed slicers on a single axis.

Furthermore, you can set the values using the following methods.

You can use a right-hand-side of an alignable object as well.

The xs() method of DataFrame additionally takes a level argument to make selecting data at a particular level of a MultiIndex easier.

You can also select on the columns with xs, by providing the axis argument.

xs also allows selection with multiple keys.

You can pass drop_level=False to xs to retain the level that was selected.

Compare the above with the result using drop_level=True (the default value).

Using the parameter level in the reindex() and align() methods of pandas objects is useful to broadcast values across a level. For instance:

The swaplevel() method can switch the order of two levels:

The reorder_levels() method generalizes the swaplevel method, allowing you to permute the hierarchical index levels in one step:

The rename() method is used to rename the labels of a MultiIndex, and is typically used to rename the columns of a DataFrame. The columns argument of rename allows a dictionary to be specified that includes only the columns you wish to rename.

This method can also be used to rename specific labels of the main index of the DataFrame.

The rename_axis() method is used to rename the name of a Index or MultiIndex. In particular, the names of the levels of a MultiIndex can be specified, which is useful if reset_index() is later used to move the values from the MultiIndex to a column.

Note that the columns of a DataFrame are an index, so that using rename_axis with the columns argument will change the name of that index.

Both rename and rename_axis support specifying a dictionary, Series or a mapping function to map labels/names to new values.

When working with an Index object directly, rather than via a DataFrame, Index.set_names() can be used to change the names.

You cannot set the names of the MultiIndex via a level.

Use Index.set_names() instead.

For MultiIndex-ed objects to be indexed and sliced effectively, they need to be sorted. As with any index, you can use sort_index().

You may also pass a level name to sort_index if the MultiIndex levels are named.

On higher dimensional objects, you can sort any of the other axes by level if they have a MultiIndex:

Indexing will work even if the data are not sorted, but will be rather inefficient (and show a PerformanceWarning). It will also return a copy of the data rather than a view:

Furthermore, if you try to index something that is not fully lexsorted, this can raise:

The is_monotonic_increasing() method on a MultiIndex shows if the index is sorted:

And now selection works as expected.

Similar to NumPy ndarrays, pandas Index, Series, and DataFrame also provides the take() method that retrieves elements along a given axis at the given indices. The given indices must be either a list or an ndarray of integer index positions. take will also accept negative integers as relative positions to the end of the object.

For DataFrames, the given indices should be a 1d list or ndarray that specifies row or column positions.

It is important to note that the take method on pandas objects are not intended to work on boolean indices and may return unexpected results.

Finally, as a small note on performance, because the take method handles a narrower range of inputs, it can offer performance that is a good deal faster than fancy indexing.

We have discussed MultiIndex in the previous sections pretty extensively. Documentation about DatetimeIndex and PeriodIndex are shown here, and documentation about TimedeltaIndex is found here.

In the following sub-sections we will highlight some other index types.

CategoricalIndex is a type of index that is useful for supporting indexing with duplicates. This is a container around a Categorical and allows efficient indexing and storage of an index with a large number of duplicated elements.

Setting the index will create a CategoricalIndex.

Indexing with __getitem__/.iloc/.loc works similarly to an Index with duplicates. The indexers must be in the category or the operation will raise a KeyError.

The CategoricalIndex is preserved after indexing:

Sorting the index will sort by the order of the categories (recall that we created the index with CategoricalDtype(list('cab')), so the sorted order is cab).

Groupby operations on the index will preserve the index nature as well.

Reindexing operations will return a resulting index based on the type of the passed indexer. Passing a list will return a plain-old Index; indexing with a Categorical will return a CategoricalIndex, indexed according to the categories of the passed Categorical dtype. This allows one to arbitrarily index these even with values not in the categories, similarly to how you can reindex any pandas index.

Reshaping and Comparison operations on a CategoricalIndex must have the same categories or a TypeError will be raised.

RangeIndex is a sub-class of Index that provides the default index for all DataFrame and Series objects. RangeIndex is an optimized version of Index that can represent a monotonic ordered set. These are analogous to Python range types. A RangeIndex will always have an int64 dtype.

RangeIndex is the default index for all DataFrame and Series objects:

A RangeIndex will behave similarly to a Index with an int64 dtype and operations on a RangeIndex, whose result cannot be represented by a RangeIndex, but should have an integer dtype, will be converted to an Index with int64. For example:

IntervalIndex together with its own dtype, IntervalDtype as well as the Interval scalar type, allow first-class support in pandas for interval notation.

The IntervalIndex allows some unique indexing and is also used as a return type for the categories in cut() and qcut().

An IntervalIndex can be used in Series and in DataFrame as the index.

Label based indexing via .loc along the edges of an interval works as you would expect, selecting that particular interval.

If you select a label contained within an interval, this will also select the interval.

Selecting using an Interval will only return exact matches.

Trying to select an Interval that is not exactly contained in the IntervalIndex will raise a KeyError.

Selecting all Intervals that overlap a given Interval can be performed using the overlaps() method to create a boolean indexer.

cut() and qcut() both return a Categorical object, and the bins they create are stored as an IntervalIndex in its .categories attribute.

cut() also accepts an IntervalIndex for its bins argument, which enables a useful pandas idiom. First, We call cut() with some data and bins set to a fixed number, to generate the bins. Then, we pass the values of .categories as the bins argument in subsequent calls to cut(), supplying new data which will be binned into the same bins.

Any value which falls outside all bins will be assigned a NaN value.

If we need intervals on a regular frequency, we can use the interval_range() function to create an IntervalIndex using various combinations of start, end, and periods. The default frequency for interval_range is a 1 for numeric intervals, and calendar day for datetime-like intervals:

The freq parameter can used to specify non-default frequencies, and can utilize a variety of frequency aliases with datetime-like intervals:

Additionally, the closed parameter can be used to specify which side(s) the intervals are closed on. Intervals are closed on the right side by default.

Specifying start, end, and periods will generate a range of evenly spaced intervals from start to end inclusively, with periods number of elements in the resulting IntervalIndex:

Label-based indexing with integer axis labels is a thorny topic. It has been discussed heavily on mailing lists and among various members of the scientific Python community. In pandas, our general viewpoint is that labels matter more than integer locations. Therefore, with an integer axis index only label-based indexing is possible with the standard tools like .loc. The following code will generate exceptions:

This deliberate decision was made to prevent ambiguities and subtle bugs (many users reported finding bugs when the API change was made to stop “falling back” on position-based indexing).

If the index of a Series or DataFrame is monotonically increasing or decreasing, then the bounds of a label-based slice can be outside the range of the index, much like slice indexing a normal Python list. Monotonicity of an index can be tested with the is_monotonic_increasing() and is_monotonic_decreasing() attributes.

On the other hand, if the index is not monotonic, then both slice bounds must be unique members of the index.

Index.is_monotonic_increasing and Index.is_monotonic_decreasing only check that an index is weakly monotonic. To check for strict monotonicity, you can combine one of those with the is_unique() attribute.

Compared with standard Python sequence slicing in which the slice endpoint is not inclusive, label-based slicing in pandas is inclusive. The primary reason for this is that it is often not possible to easily determine the “successor” or next element after a particular label in an index. For example, consider the following Series:

Suppose we wished to slice from c to e, using integers this would be accomplished as such:

However, if you only had c and e, determining the next element in the index can be somewhat complicated. For example, the following does not work:

A very common use case is to limit a time series to start and end at two specific dates. To enable this, we made the design choice to make label-based slicing include both endpoints:

This is most definitely a “practicality beats purity” sort of thing, but it is something to watch out for if you expect label-based slicing to behave exactly in the way that standard Python integer slicing works.

The different indexing operation can potentially change the dtype of a Series.

This is because the (re)indexing operations above silently inserts NaNs and the dtype changes accordingly. This can cause some issues when using numpy ufuncs such as numpy.logical_and.

See the GH 2388 for a more detailed discussion.

**Examples:**

Example 1 (json):
```json
In [1]: arrays = [
   ...:     ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
   ...:     ["one", "two", "one", "two", "one", "two", "one", "two"],
   ...: ]
   ...: 

In [2]: tuples = list(zip(*arrays))

In [3]: tuples
Out[3]: 
[('bar', 'one'),
 ('bar', 'two'),
 ('baz', 'one'),
 ('baz', 'two'),
 ('foo', 'one'),
 ('foo', 'two'),
 ('qux', 'one'),
 ('qux', 'two')]

In [4]: index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])

In [5]: index
Out[5]: 
MultiIndex([('bar', 'one'),
            ('bar', 'two'),
            ('baz', 'one'),
            ('baz', 'two'),
            ('foo', 'one'),
            ('foo', 'two'),
            ('qux', 'one'),
            ('qux', 'two')],
           names=['first', 'second'])

In [6]: s = pd.Series(np.random.randn(8), index=index)

In [7]: s
Out[7]: 
first  second
bar    one       0.469112
       two      -0.282863
baz    one      -1.509059
       two      -1.135632
foo    one       1.212112
       two      -0.173215
qux    one       0.119209
       two      -1.044236
dtype: float64
```

Example 2 (typescript):
```typescript
In [8]: iterables = [["bar", "baz", "foo", "qux"], ["one", "two"]]

In [9]: pd.MultiIndex.from_product(iterables, names=["first", "second"])
Out[9]: 
MultiIndex([('bar', 'one'),
            ('bar', 'two'),
            ('baz', 'one'),
            ('baz', 'two'),
            ('foo', 'one'),
            ('foo', 'two'),
            ('qux', 'one'),
            ('qux', 'two')],
           names=['first', 'second'])
```

Example 3 (typescript):
```typescript
In [10]: df = pd.DataFrame(
   ....:     [["bar", "one"], ["bar", "two"], ["foo", "one"], ["foo", "two"]],
   ....:     columns=["first", "second"],
   ....: )
   ....: 

In [11]: pd.MultiIndex.from_frame(df)
Out[11]: 
MultiIndex([('bar', 'one'),
            ('bar', 'two'),
            ('foo', 'one'),
            ('foo', 'two')],
           names=['first', 'second'])
```

Example 4 (typescript):
```typescript
In [12]: arrays = [
   ....:     np.array(["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"]),
   ....:     np.array(["one", "two", "one", "two", "one", "two", "one", "two"]),
   ....: ]
   ....: 

In [13]: s = pd.Series(np.random.randn(8), index=arrays)

In [14]: s
Out[14]: 
bar  one   -0.861849
     two   -2.104569
baz  one   -0.494929
     two    1.071804
foo  one    0.721555
     two   -0.706771
qux  one   -1.039575
     two    0.271860
dtype: float64

In [15]: df = pd.DataFrame(np.random.randn(8, 4), index=arrays)

In [16]: df
Out[16]: 
                0         1         2         3
bar one -0.424972  0.567020  0.276232 -1.087401
    two -0.673690  0.113648 -1.478427  0.524988
baz one  0.404705  0.577046 -1.715002 -1.039268
    two -0.370647 -1.157892 -1.344312  0.844885
foo one  1.075770 -0.109050  1.643563 -1.469388
    two  0.357021 -0.674600 -1.776904 -0.968914
qux one -1.294524  0.413738  0.276662 -0.472035
    two -0.013960 -0.362543 -0.006154 -0.923061
```

---

## User Guide#

**URL:** https://pandas.pydata.org/docs/user_guide/index.html

**Contents:**
- User Guide#
- How to read these guides#
- Guides#

The User Guide covers all of pandas by topic area. Each of the subsections introduces a topic (such as “working with missing data”), and discusses how pandas approaches the problem, with many examples throughout.

Users brand-new to pandas should start with 10 minutes to pandas.

For a high level summary of the pandas fundamentals, see Intro to data structures and Essential basic functionality.

Further information on any specific method can be obtained in the API reference.

In these guides you will see input code inside code blocks such as:

The first block is a standard python input, while in the second the In [1]: indicates the input is inside a notebook. In Jupyter Notebooks the last line is printed and plots are shown inline.

**Examples:**

Example 1 (typescript):
```typescript
import pandas as pd
pd.DataFrame({'A': [1, 2, 3]})
```

Example 2 (python):
```python
In [1]: import pandas as pd

In [2]: pd.DataFrame({'A': [1, 2, 3]})
Out[2]: 
   A
0  1
1  2
2  3
```

Example 3 (typescript):
```typescript
In [3]: a = 1

In [4]: a
Out[4]: 1
```

Example 4 (unknown):
```unknown
a = 1
print(a)
```

---

## Working with text data#

**URL:** https://pandas.pydata.org/docs/user_guide/text.html

**Contents:**
- Working with text data#
- Text data types#
  - Behavior differences#
- String methods#
- Splitting and replacing strings#
- Concatenation#
  - Concatenating a single Series into a string#
  - Concatenating a Series and something list-like into a Series#
  - Concatenating a Series and something array-like into a Series#
  - Concatenating a Series and an indexed object into a Series, with alignment#

There are two ways to store text data in pandas:

object -dtype NumPy array.

StringDtype extension type.

We recommend using StringDtype to store text data.

Prior to pandas 1.0, object dtype was the only option. This was unfortunate for many reasons:

You can accidentally store a mixture of strings and non-strings in an object dtype array. It’s better to have a dedicated dtype.

object dtype breaks dtype-specific operations like DataFrame.select_dtypes(). There isn’t a clear way to select just text while excluding non-text but still object-dtype columns.

When reading code, the contents of an object dtype array is less clear than 'string'.

Currently, the performance of object dtype arrays of strings and arrays.StringArray are about the same. We expect future enhancements to significantly increase the performance and lower the memory overhead of StringArray.

StringArray is currently considered experimental. The implementation and parts of the API may change without warning.

For backwards-compatibility, object dtype remains the default type we infer a list of strings to

To explicitly request string dtype, specify the dtype

Or astype after the Series or DataFrame is created

You can also use StringDtype/"string" as the dtype on non-string data and it will be converted to string dtype:

or convert from existing pandas data:

These are places where the behavior of StringDtype objects differ from object dtype

For StringDtype, string accessor methods that return numeric output will always return a nullable integer dtype, rather than either int or float dtype, depending on the presence of NA values. Methods returning boolean output will return a nullable boolean dtype.

Both outputs are Int64 dtype. Compare that with object-dtype

When NA values are present, the output dtype is float64. Similarly for methods returning boolean values.

Some string methods, like Series.str.decode() are not available on StringArray because StringArray only holds strings, not bytes.

In comparison operations, arrays.StringArray and Series backed by a StringArray will return an object with BooleanDtype, rather than a bool dtype object. Missing values in a StringArray will propagate in comparison operations, rather than always comparing unequal like numpy.nan.

Everything else that follows in the rest of this document applies equally to string and object dtype.

Series and Index are equipped with a set of string processing methods that make it easy to operate on each element of the array. Perhaps most importantly, these methods exclude missing/NA values automatically. These are accessed via the str attribute and generally have names matching the equivalent (scalar) built-in string methods:

The string methods on Index are especially useful for cleaning up or transforming DataFrame columns. For instance, you may have columns with leading or trailing whitespace:

Since df.columns is an Index object, we can use the .str accessor

These string methods can then be used to clean up the columns as needed. Here we are removing leading and trailing whitespaces, lower casing all names, and replacing any remaining whitespaces with underscores:

If you have a Series where lots of elements are repeated (i.e. the number of unique elements in the Series is a lot smaller than the length of the Series), it can be faster to convert the original Series to one of type category and then use .str.<method> or .dt.<property> on that. The performance difference comes from the fact that, for Series of type category, the string operations are done on the .categories and not on each element of the Series.

Please note that a Series of type category with string .categories has some limitations in comparison to Series of type string (e.g. you can’t add strings to each other: s + " " + s won’t work if s is a Series of type category). Also, .str methods which operate on elements of type list are not available on such a Series.

The type of the Series is inferred and the allowed types (i.e. strings).

Generally speaking, the .str accessor is intended to work only on strings. With very few exceptions, other uses are not supported, and may be disabled at a later point.

Methods like split return a Series of lists:

Elements in the split lists can be accessed using get or [] notation:

It is easy to expand this to return a DataFrame using expand.

When original Series has StringDtype, the output columns will all be StringDtype as well.

It is also possible to limit the number of splits:

rsplit is similar to split except it works in the reverse direction, i.e., from the end of the string to the beginning of the string:

replace optionally uses regular expressions:

Changed in version 2.0.

Single character pattern with regex=True will also be treated as regular expressions:

If you want literal replacement of a string (equivalent to str.replace()), you can set the optional regex parameter to False, rather than escaping each character. In this case both pat and repl must be strings:

The replace method can also take a callable as replacement. It is called on every pat using re.sub(). The callable should expect one positional argument (a regex object) and return a string.

The replace method also accepts a compiled regular expression object from re.compile() as a pattern. All flags should be included in the compiled regular expression object.

Including a flags argument when calling replace with a compiled regular expression object will raise a ValueError.

removeprefix and removesuffix have the same effect as str.removeprefix and str.removesuffix added in Python 3.9 <https://docs.python.org/3/library/stdtypes.html#str.removeprefix>`__:

Added in version 1.4.0.

There are several ways to concatenate a Series or Index, either with itself or others, all based on cat(), resp. Index.str.cat.

The content of a Series (or Index) can be concatenated:

If not specified, the keyword sep for the separator defaults to the empty string, sep='':

By default, missing values are ignored. Using na_rep, they can be given a representation:

The first argument to cat() can be a list-like object, provided that it matches the length of the calling Series (or Index).

Missing values on either side will result in missing values in the result as well, unless na_rep is specified:

The parameter others can also be two-dimensional. In this case, the number or rows must match the lengths of the calling Series (or Index).

For concatenation with a Series or DataFrame, it is possible to align the indexes before concatenation by setting the join-keyword.

The usual options are available for join (one of 'left', 'outer', 'inner', 'right'). In particular, alignment also means that the different lengths do not need to coincide anymore.

The same alignment can be used when others is a DataFrame:

Several array-like items (specifically: Series, Index, and 1-dimensional variants of np.ndarray) can be combined in a list-like container (including iterators, dict-views, etc.).

All elements without an index (e.g. np.ndarray) within the passed list-like must match in length to the calling Series (or Index), but Series and Index may have arbitrary length (as long as alignment is not disabled with join=None):

If using join='right' on a list-like of others that contains different indexes, the union of these indexes will be used as the basis for the final concatenation:

You can use [] notation to directly index by position locations. If you index past the end of the string, the result will be a NaN.

The extract method accepts a regular expression with at least one capture group.

Extracting a regular expression with more than one group returns a DataFrame with one column per group.

Elements that do not match return a row filled with NaN. Thus, a Series of messy strings can be “converted” into a like-indexed Series or DataFrame of cleaned-up or more useful strings, without necessitating get() to access tuples or re.match objects. The dtype of the result is always object, even if no match is found and the result only contains NaN.

and optional groups like

can also be used. Note that any capture group names in the regular expression will be used for column names; otherwise capture group numbers will be used.

Extracting a regular expression with one group returns a DataFrame with one column if expand=True.

It returns a Series if expand=False.

Calling on an Index with a regex with exactly one capture group returns a DataFrame with one column if expand=True.

It returns an Index if expand=False.

Calling on an Index with a regex with more than one capture group returns a DataFrame if expand=True.

It raises ValueError if expand=False.

The table below summarizes the behavior of extract(expand=False) (input subject in first column, number of groups in regex in first row)

Unlike extract (which returns only the first match),

the extractall method returns every match. The result of extractall is always a DataFrame with a MultiIndex on its rows. The last level of the MultiIndex is named match and indicates the order in the subject.

When each subject string in the Series has exactly one match,

then extractall(pat).xs(0, level='match') gives the same result as extract(pat).

Index also supports .str.extractall. It returns a DataFrame which has the same result as a Series.str.extractall with a default index (starts from 0).

You can check whether elements contain a pattern:

Or whether elements match a pattern:

The distinction between match, fullmatch, and contains is strictness: fullmatch tests whether the entire string matches the regular expression; match tests whether there is a match of the regular expression that begins at the first character of the string; and contains tests whether there is a match of the regular expression at any position within the string.

The corresponding functions in the re package for these three match modes are re.fullmatch, re.match, and re.search, respectively.

Methods like match, fullmatch, contains, startswith, and endswith take an extra na argument so missing values can be considered True or False:

You can extract dummy variables from string columns. For example if they are separated by a '|':

String Index also supports get_dummies which returns a MultiIndex.

See also get_dummies().

Split strings on delimiter

Split strings on delimiter working from the end of the string

Index into each element (retrieve i-th element)

Join strings in each element of the Series with passed separator

Split strings on the delimiter returning DataFrame of dummy variables

Return boolean array if each string contains pattern/regex

Replace occurrences of pattern/regex/string with some other string or the return value of a callable given the occurrence

Remove prefix from string i.e. only remove if string starts with prefix.

Remove suffix from string i.e. only remove if string ends with suffix.

Duplicate values (s.str.repeat(3) equivalent to x * 3)

Add whitespace to the sides of strings

Equivalent to str.center

Equivalent to str.ljust

Equivalent to str.rjust

Equivalent to str.zfill

Split long strings into lines with length less than a given width

Slice each string in the Series

Replace slice in each string with passed value

Count occurrences of pattern

Equivalent to str.startswith(pat) for each element

Equivalent to str.endswith(pat) for each element

Compute list of all occurrences of pattern/regex for each string

Call re.match on each element returning matched groups as list

Call re.search on each element returning DataFrame with one row for each element and one column for each regex capture group

Call re.findall on each element returning DataFrame with one row for each match and one column for each regex capture group

Compute string lengths

Equivalent to str.strip

Equivalent to str.rstrip

Equivalent to str.lstrip

Equivalent to str.partition

Equivalent to str.rpartition

Equivalent to str.lower

Equivalent to str.casefold

Equivalent to str.upper

Equivalent to str.find

Equivalent to str.rfind

Equivalent to str.index

Equivalent to str.rindex

Equivalent to str.capitalize

Equivalent to str.swapcase

Return Unicode normal form. Equivalent to unicodedata.normalize

Equivalent to str.translate

Equivalent to str.isalnum

Equivalent to str.isalpha

Equivalent to str.isdigit

Equivalent to str.isspace

Equivalent to str.islower

Equivalent to str.isupper

Equivalent to str.istitle

Equivalent to str.isnumeric

Equivalent to str.isdecimal

**Examples:**

Example 1 (yaml):
```yaml
In [1]: pd.Series(["a", "b", "c"])
Out[1]: 
0    a
1    b
2    c
dtype: object
```

Example 2 (yaml):
```yaml
In [2]: pd.Series(["a", "b", "c"], dtype="string")
Out[2]: 
0    a
1    b
2    c
dtype: string

In [3]: pd.Series(["a", "b", "c"], dtype=pd.StringDtype())
Out[3]: 
0    a
1    b
2    c
dtype: string
```

Example 3 (typescript):
```typescript
In [4]: s = pd.Series(["a", "b", "c"])

In [5]: s
Out[5]: 
0    a
1    b
2    c
dtype: object

In [6]: s.astype("string")
Out[6]: 
0    a
1    b
2    c
dtype: string
```

Example 4 (typescript):
```typescript
In [7]: s = pd.Series(["a", 2, np.nan], dtype="string")

In [8]: s
Out[8]: 
0       a
1       2
2    <NA>
dtype: string

In [9]: type(s[1])
Out[9]: str
```

---

## 

**URL:** https://pandas.pydata.org/docs/_sources/user_guide/basics.rst.txt

---

## IO tools (text, CSV, HDF5, …)#

**URL:** https://pandas.pydata.org/docs/user_guide/io.html

**Contents:**
- IO tools (text, CSV, HDF5, …)#
- CSV & text files#
  - Parsing options#
    - Basic#
    - Column and index locations and names#
    - General parsing configuration#
    - NA and missing data handling#
    - Datetime handling#
    - Iteration#
    - Quoting, compression, and file format#

The pandas I/O API is a set of top level reader functions accessed like pandas.read_csv() that generally return a pandas object. The corresponding writer functions are object methods that are accessed like DataFrame.to_csv(). Below is a table containing available readers and writers.

Fixed-Width Text File

Google BigQuery;:ref:read_gbq<io.bigquery>;:ref:to_gbq<io.bigquery>

Here is an informal performance comparison for some of these IO methods.

For examples that use the StringIO class, make sure you import it with from io import StringIO for Python 3.

The workhorse function for reading text files (a.k.a. flat files) is read_csv(). See the cookbook for some advanced strategies.

read_csv() accepts the following common arguments:

Either a path to a file (a str, pathlib.Path, or py:py._path.local.LocalPath), URL (including http, ftp, and S3 locations), or any object with a read() method (such as an open file or StringIO).

Delimiter to use. If sep is None, the C engine cannot automatically detect the separator, but the Python parsing engine can, meaning the latter will be used and automatically detect the separator by Python’s builtin sniffer tool, csv.Sniffer. In addition, separators longer than 1 character and different from '\s+' will be interpreted as regular expressions and will also force the use of the Python parsing engine. Note that regex delimiters are prone to ignoring quoted data. Regex example: '\\r\\t'.

Alternative argument name for sep.

Specifies whether or not whitespace (e.g. ' ' or '\t') will be used as the delimiter. Equivalent to setting sep='\s+'. If this option is set to True, nothing should be passed in for the delimiter parameter.

Row number(s) to use as the column names, and the start of the data. Default behavior is to infer the column names: if no names are passed the behavior is identical to header=0 and column names are inferred from the first line of the file, if column names are passed explicitly then the behavior is identical to header=None. Explicitly pass header=0 to be able to replace existing names.

The header can be a list of ints that specify row locations for a MultiIndex on the columns e.g. [0,1,3]. Intervening rows that are not specified will be skipped (e.g. 2 in this example is skipped). Note that this parameter ignores commented lines and empty lines if skip_blank_lines=True, so header=0 denotes the first line of data rather than the first line of the file.

List of column names to use. If file contains no header row, then you should explicitly pass header=None. Duplicates in this list are not allowed.

Column(s) to use as the row labels of the DataFrame, either given as string name or column index. If a sequence of int / str is given, a MultiIndex is used.

index_col=False can be used to force pandas to not use the first column as the index, e.g. when you have a malformed file with delimiters at the end of each line.

The default value of None instructs pandas to guess. If the number of fields in the column header row is equal to the number of fields in the body of the data file, then a default index is used. If it is larger, then the first columns are used as index so that the remaining number of fields in the body are equal to the number of fields in the header.

The first row after the header is used to determine the number of columns, which will go into the index. If the subsequent rows contain less columns than the first row, they are filled with NaN.

This can be avoided through usecols. This ensures that the columns are taken as is and the trailing data are ignored.

Return a subset of the columns. If list-like, all elements must either be positional (i.e. integer indices into the document columns) or strings that correspond to column names provided either by the user in names or inferred from the document header row(s). If names are given, the document header row(s) are not taken into account. For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].

Element order is ignored, so usecols=[0, 1] is the same as [1, 0]. To instantiate a DataFrame from data with element order preserved use pd.read_csv(data, usecols=['foo', 'bar'])[['foo', 'bar']] for columns in ['foo', 'bar'] order or pd.read_csv(data, usecols=['foo', 'bar'])[['bar', 'foo']] for ['bar', 'foo'] order.

If callable, the callable function will be evaluated against the column names, returning names where the callable function evaluates to True:

Using this parameter results in much faster parsing time and lower memory usage when using the c engine. The Python engine loads the data first before deciding which columns to drop.

Data type for data or columns. E.g. {'a': np.float64, 'b': np.int32, 'c': 'Int64'} Use str or object together with suitable na_values settings to preserve and not interpret dtype. If converters are specified, they will be applied INSTEAD of dtype conversion.

Added in version 1.5.0: Support for defaultdict was added. Specify a defaultdict as input where the default determines the dtype of the columns which are not explicitly listed.

Which dtype_backend to use, e.g. whether a DataFrame should have NumPy arrays, nullable dtypes are used for all dtypes that have a nullable implementation when “numpy_nullable” is set, pyarrow is used for all dtypes if “pyarrow” is set.

The dtype_backends are still experimential.

Added in version 2.0.

Parser engine to use. The C and pyarrow engines are faster, while the python engine is currently more feature-complete. Multithreading is currently only supported by the pyarrow engine.

Added in version 1.4.0: The “pyarrow” engine was added as an experimental engine, and some features are unsupported, or may not work correctly, with this engine.

Dict of functions for converting values in certain columns. Keys can either be integers or column labels.

Values to consider as True.

Values to consider as False.

Skip spaces after delimiter.

Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file.

If callable, the callable function will be evaluated against the row indices, returning True if the row should be skipped and False otherwise:

Number of lines at bottom of file to skip (unsupported with engine=’c’).

Number of rows of file to read. Useful for reading pieces of large files.

Internally process the file in chunks, resulting in lower memory use while parsing, but possibly mixed type inference. To ensure no mixed types either set False, or specify the type with the dtype parameter. Note that the entire file is read into a single DataFrame regardless, use the chunksize or iterator parameter to return the data in chunks. (Only valid with C parser)

If a filepath is provided for filepath_or_buffer, map the file object directly onto memory and access the data directly from there. Using this option can improve performance because there is no longer any I/O overhead.

Additional strings to recognize as NA/NaN. If dict passed, specific per-column NA values. See na values const below for a list of the values interpreted as NaN by default.

Whether or not to include the default NaN values when parsing the data. Depending on whether na_values is passed in, the behavior is as follows:

If keep_default_na is True, and na_values are specified, na_values is appended to the default NaN values used for parsing.

If keep_default_na is True, and na_values are not specified, only the default NaN values are used for parsing.

If keep_default_na is False, and na_values are specified, only the NaN values specified na_values are used for parsing.

If keep_default_na is False, and na_values are not specified, no strings will be parsed as NaN.

Note that if na_filter is passed in as False, the keep_default_na and na_values parameters will be ignored.

Detect missing value markers (empty strings and the value of na_values). In data without any NAs, passing na_filter=False can improve the performance of reading a large file.

Indicate number of NA values placed in non-numeric columns.

If True, skip over blank lines rather than interpreting as NaN values.

If True -> try parsing the index.

If [1, 2, 3] -> try parsing columns 1, 2, 3 each as a separate date column.

If [[1, 3]] -> combine columns 1 and 3 and parse as a single date column.

If {'foo': [1, 3]} -> parse columns 1, 3 as date and call result ‘foo’.

A fast-path exists for iso8601-formatted dates.

If True and parse_dates is enabled for a column, attempt to infer the datetime format to speed up the processing.

Deprecated since version 2.0.0: A strict version of this argument is now the default, passing it has no effect.

If True and parse_dates specifies combining multiple columns then keep the original columns.

Function to use for converting a sequence of string columns to an array of datetime instances. The default uses dateutil.parser.parser to do the conversion. pandas will try to call date_parser in three different ways, advancing to the next if an exception occurs: 1) Pass one or more arrays (as defined by parse_dates) as arguments; 2) concatenate (row-wise) the string values from the columns defined by parse_dates into a single array and pass that; and 3) call date_parser once for each row using one or more strings (corresponding to the columns defined by parse_dates) as arguments.

Deprecated since version 2.0.0: Use date_format instead, or read in as object and then apply to_datetime() as-needed.

If used in conjunction with parse_dates, will parse dates according to this format. For anything more complex, please read in as object and then apply to_datetime() as-needed.

Added in version 2.0.0.

DD/MM format dates, international and European format.

If True, use a cache of unique, converted dates to apply the datetime conversion. May produce significant speed-up when parsing duplicate date strings, especially ones with timezone offsets.

Return TextFileReader object for iteration or getting chunks with get_chunk().

Return TextFileReader object for iteration. See iterating and chunking below.

For on-the-fly decompression of on-disk data. If ‘infer’, then use gzip, bz2, zip, xz, or zstandard if filepath_or_buffer is path-like ending in ‘.gz’, ‘.bz2’, ‘.zip’, ‘.xz’, ‘.zst’, respectively, and no decompression otherwise. If using ‘zip’, the ZIP file must contain only one data file to be read in. Set to None for no decompression. Can also be a dict with key 'method' set to one of {'zip', 'gzip', 'bz2', 'zstd'} and other key-value pairs are forwarded to zipfile.ZipFile, gzip.GzipFile, bz2.BZ2File, or zstandard.ZstdDecompressor. As an example, the following could be passed for faster compression and to create a reproducible gzip archive: compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}.

Changed in version 1.2.0: Previous versions forwarded dict entries for ‘gzip’ to gzip.open.

Character to recognize as decimal point. E.g. use ',' for European data.

Specifies which converter the C engine should use for floating-point values. The options are None for the ordinary converter, high for the high-precision converter, and round_trip for the round-trip converter.

Character to break file into lines. Only valid with C parser.

The character used to denote the start and end of a quoted item. Quoted items can include the delimiter and it will be ignored.

Control field quoting behavior per csv.QUOTE_* constants. Use one of QUOTE_MINIMAL (0), QUOTE_ALL (1), QUOTE_NONNUMERIC (2) or QUOTE_NONE (3).

When quotechar is specified and quoting is not QUOTE_NONE, indicate whether or not to interpret two consecutive quotechar elements inside a field as a single quotechar element.

One-character string used to escape delimiter when quoting is QUOTE_NONE.

Indicates remainder of line should not be parsed. If found at the beginning of a line, the line will be ignored altogether. This parameter must be a single character. Like empty lines (as long as skip_blank_lines=True), fully commented lines are ignored by the parameter header but not by skiprows. For example, if comment='#', parsing ‘#empty\na,b,c\n1,2,3’ with header=0 will result in ‘a,b,c’ being treated as the header.

Encoding to use for UTF when reading/writing (e.g. 'utf-8'). List of Python standard encodings.

If provided, this parameter will override values (default or not) for the following parameters: delimiter, doublequote, escapechar, skipinitialspace, quotechar, and quoting. If it is necessary to override values, a ParserWarning will be issued. See csv.Dialect documentation for more details.

Specifies what to do upon encountering a bad line (a line with too many fields). Allowed values are :

‘error’, raise an ParserError when a bad line is encountered.

‘warn’, print a warning when a bad line is encountered and skip that line.

‘skip’, skip bad lines without raising or warning when they are encountered.

Added in version 1.3.0.

You can indicate the data type for the whole DataFrame or individual columns:

Fortunately, pandas offers more than one way to ensure that your column(s) contain only one dtype. If you’re unfamiliar with these concepts, you can see here to learn more about dtypes, and here to learn more about object conversion in pandas.

For instance, you can use the converters argument of read_csv():

Or you can use the to_numeric() function to coerce the dtypes after reading in the data,

which will convert all valid parsing to floats, leaving the invalid parsing as NaN.

Ultimately, how you deal with reading in columns containing mixed dtypes depends on your specific needs. In the case above, if you wanted to NaN out the data anomalies, then to_numeric() is probably your best option. However, if you wanted for all the data to be coerced, no matter the type, then using the converters argument of read_csv() would certainly be worth trying.

In some cases, reading in abnormal data with columns containing mixed dtypes will result in an inconsistent dataset. If you rely on pandas to infer the dtypes of your columns, the parsing engine will go and infer the dtypes for different chunks of the data, rather than the whole dataset at once. Consequently, you can end up with column(s) with mixed dtypes. For example,

will result with mixed_df containing an int dtype for certain chunks of the column, and str for others due to the mixed dtypes from the data that was read in. It is important to note that the overall column will be marked with a dtype of object, which is used for columns with mixed dtypes.

Setting dtype_backend="numpy_nullable" will result in nullable dtypes for every column.

Categorical columns can be parsed directly by specifying dtype='category' or dtype=CategoricalDtype(categories, ordered).

Individual columns can be parsed as a Categorical using a dict specification:

Specifying dtype='category' will result in an unordered Categorical whose categories are the unique values observed in the data. For more control on the categories and order, create a CategoricalDtype ahead of time, and pass that for that column’s dtype.

When using dtype=CategoricalDtype, “unexpected” values outside of dtype.categories are treated as missing values.

This matches the behavior of Categorical.set_categories().

With dtype='category', the resulting categories will always be parsed as strings (object dtype). If the categories are numeric they can be converted using the to_numeric() function, or as appropriate, another converter such as to_datetime().

When dtype is a CategoricalDtype with homogeneous categories ( all numeric, all datetimes, etc.), the conversion is done automatically.

A file may or may not have a header row. pandas assumes the first row should be used as the column names:

By specifying the names argument in conjunction with header you can indicate other names to use and whether or not to throw away the header row (if any):

If the header is in a row other than the first, pass the row number to header. This will skip the preceding rows:

Default behavior is to infer the column names: if no names are passed the behavior is identical to header=0 and column names are inferred from the first non-blank line of the file, if column names are passed explicitly then the behavior is identical to header=None.

If the file or header contains duplicate names, pandas will by default distinguish between them so as to prevent overwriting data:

There is no more duplicate data because duplicate columns ‘X’, …, ‘X’ become ‘X’, ‘X.1’, …, ‘X.N’.

The usecols argument allows you to select any subset of the columns in a file, either using the column names, position numbers or a callable:

The usecols argument can also be used to specify which columns not to use in the final result:

In this case, the callable is specifying that we exclude the “a” and “c” columns from the output.

If the comment parameter is specified, then completely commented lines will be ignored. By default, completely blank lines will be ignored as well.

If skip_blank_lines=False, then read_csv will not ignore blank lines:

The presence of ignored lines might create ambiguities involving line numbers; the parameter header uses row numbers (ignoring commented/empty lines), while skiprows uses line numbers (including commented/empty lines):

If both header and skiprows are specified, header will be relative to the end of skiprows. For example:

Sometimes comments or meta data may be included in a file:

By default, the parser includes the comments in the output:

We can suppress the comments using the comment keyword:

The encoding argument should be used for encoded unicode data, which will result in byte strings being decoded to unicode in the result:

Some formats which encode all characters as multiple bytes, like UTF-16, won’t parse correctly at all without specifying the encoding. Full list of Python standard encodings.

If a file has one more column of data than the number of column names, the first column will be used as the DataFrame’s row names:

Ordinarily, you can achieve this behavior using the index_col option.

There are some exception cases when a file has been prepared with delimiters at the end of each data line, confusing the parser. To explicitly disable the index column inference and discard the last column, pass index_col=False:

If a subset of data is being parsed using the usecols option, the index_col specification is based on that subset, not the original data.

To better facilitate working with datetime data, read_csv() uses the keyword arguments parse_dates and date_format to allow users to specify a variety of columns and date/time formats to turn the input text data into datetime objects.

The simplest case is to just pass in parse_dates=True:

It is often the case that we may want to store date and time data separately, or store various date fields separately. the parse_dates keyword can be used to specify a combination of columns to parse the dates and/or times from.

You can specify a list of column lists to parse_dates, the resulting date columns will be prepended to the output (so as to not affect the existing column order) and the new column names will be the concatenation of the component column names:

By default the parser removes the component date columns, but you can choose to retain them via the keep_date_col keyword:

Note that if you wish to combine multiple columns into a single date column, a nested list must be used. In other words, parse_dates=[1, 2] indicates that the second and third columns should each be parsed as separate date columns while parse_dates=[[1, 2]] means the two columns should be parsed into a single column.

You can also use a dict to specify custom name columns:

It is important to remember that if multiple text columns are to be parsed into a single date column, then a new column is prepended to the data. The index_col specification is based off of this new set of columns rather than the original data columns:

If a column or index contains an unparsable date, the entire column or index will be returned unaltered as an object data type. For non-standard datetime parsing, use to_datetime() after pd.read_csv.

read_csv has a fast_path for parsing datetime strings in iso8601 format, e.g “2000-01-01T00:01:02+00:00” and similar variations. If you can arrange for your data to store datetimes in this format, load times will be significantly faster, ~20x has been observed.

Deprecated since version 2.2.0: Combining date columns inside read_csv is deprecated. Use pd.to_datetime on the relevant result columns instead.

Finally, the parser allows you to specify a custom date_format. Performance-wise, you should try these methods of parsing dates in order:

If you know the format, use date_format, e.g.: date_format="%d/%m/%Y" or date_format={column_name: "%d/%m/%Y"}.

If you different formats for different columns, or want to pass any extra options (such as utc) to to_datetime, then you should read in your data as object dtype, and then use to_datetime.

pandas cannot natively represent a column or index with mixed timezones. If your CSV file contains columns with a mixture of timezones, the default result will be an object-dtype column with strings, even with parse_dates. To parse the mixed-timezone values as a datetime column, read in as object dtype and then call to_datetime() with utc=True.

Here are some examples of datetime strings that can be guessed (all representing December 30th, 2011 at 00:00:00):

“12/30/2011 00:00:00”

“30/Dec/2011 00:00:00”

“30/December/2011 00:00:00”

Note that format inference is sensitive to dayfirst. With dayfirst=True, it will guess “01/12/2011” to be December 1st. With dayfirst=False (default) it will guess “01/12/2011” to be January 12th.

If you try to parse a column of date strings, pandas will attempt to guess the format from the first non-NaN element, and will then parse the rest of the column with that format. If pandas fails to guess the format (for example if your first string is '01 December US/Pacific 2000'), then a warning will be raised and each row will be parsed individually by dateutil.parser.parse. The safest way to parse dates is to explicitly set format=.

In the case that you have mixed datetime formats within the same column, you can pass format='mixed'

or, if your datetime formats are all ISO8601 (possibly not identically-formatted):

While US date formats tend to be MM/DD/YYYY, many international formats use DD/MM/YYYY instead. For convenience, a dayfirst keyword is provided:

Added in version 1.2.0.

df.to_csv(..., mode="wb") allows writing a CSV to a file object opened binary mode. In most cases, it is not necessary to specify mode as Pandas will auto-detect whether the file object is opened in text or binary mode.

The parameter float_precision can be specified in order to use a specific floating-point converter during parsing with the C engine. The options are the ordinary converter, the high-precision converter, and the round-trip converter (which is guaranteed to round-trip values after writing to a file). For example:

For large numbers that have been written with a thousands separator, you can set the thousands keyword to a string of length 1 so that integers will be parsed correctly:

By default, numbers with a thousands separator will be parsed as strings:

The thousands keyword allows integers to be parsed correctly:

To control which values are parsed as missing values (which are signified by NaN), specify a string in na_values. If you specify a list of strings, then all values in it are considered to be missing values. If you specify a number (a float, like 5.0 or an integer like 5), the corresponding equivalent values will also imply a missing value (in this case effectively [5.0, 5] are recognized as NaN).

To completely override the default values that are recognized as missing, specify keep_default_na=False.

The default NaN recognized values are ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a', 'NA', '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', 'None', ''].

Let us consider some examples:

In the example above 5 and 5.0 will be recognized as NaN, in addition to the defaults. A string will first be interpreted as a numerical 5, then as a NaN.

Above, only an empty field will be recognized as NaN.

Above, both NA and 0 as strings are NaN.

The default values, in addition to the string "Nope" are recognized as NaN.

inf like values will be parsed as np.inf (positive infinity), and -inf as -np.inf (negative infinity). These will ignore the case of the value, meaning Inf, will also be parsed as np.inf.

The common values True, False, TRUE, and FALSE are all recognized as boolean. Occasionally you might want to recognize other values as being boolean. To do this, use the true_values and false_values options as follows:

Some files may have malformed lines with too few fields or too many. Lines with too few fields will have NA values filled in the trailing fields. Lines with too many fields will raise an error by default:

You can elect to skip bad lines:

Added in version 1.4.0.

Or pass a callable function to handle the bad line if engine="python". The bad line will be a list of strings that was split by the sep:

The callable function will handle only a line with too many fields. Bad lines caused by other errors will be silently skipped.

The line was not processed in this case, as a “bad line” here is caused by an escape character.

You can also use the usecols parameter to eliminate extraneous column data that appear in some lines but not others:

In case you want to keep all data including the lines with too many fields, you can specify a sufficient number of names. This ensures that lines with not enough fields are filled with NaN.

The dialect keyword gives greater flexibility in specifying the file format. By default it uses the Excel dialect but you can specify either the dialect name or a csv.Dialect instance.

Suppose you had data with unenclosed quotes:

By default, read_csv uses the Excel dialect and treats the double quote as the quote character, which causes it to fail when it finds a newline before it finds the closing double quote.

We can get around this using dialect:

All of the dialect options can be specified separately by keyword arguments:

Another common dialect option is skipinitialspace, to skip any whitespace after a delimiter:

The parsers make every attempt to “do the right thing” and not be fragile. Type inference is a pretty big deal. If a column can be coerced to integer dtype without altering the contents, the parser will do so. Any non-numeric columns will come through as object dtype as with the rest of pandas objects.

Quotes (and other escape characters) in embedded fields can be handled in any number of ways. One way is to use backslashes; to properly parse this data, you should pass the escapechar option:

While read_csv() reads delimited data, the read_fwf() function works with data files that have known and fixed column widths. The function parameters to read_fwf are largely the same as read_csv with two extra parameters, and a different usage of the delimiter parameter:

colspecs: A list of pairs (tuples) giving the extents of the fixed-width fields of each line as half-open intervals (i.e., [from, to[ ). String value ‘infer’ can be used to instruct the parser to try detecting the column specifications from the first 100 rows of the data. Default behavior, if not specified, is to infer.

widths: A list of field widths which can be used instead of ‘colspecs’ if the intervals are contiguous.

delimiter: Characters to consider as filler characters in the fixed-width file. Can be used to specify the filler character of the fields if it is not spaces (e.g., ‘~’).

Consider a typical fixed-width data file:

In order to parse this file into a DataFrame, we simply need to supply the column specifications to the read_fwf function along with the file name:

Note how the parser automatically picks column names X.<column number> when header=None argument is specified. Alternatively, you can supply just the column widths for contiguous columns:

The parser will take care of extra white spaces around the columns so it’s ok to have extra separation between the columns in the file.

By default, read_fwf will try to infer the file’s colspecs by using the first 100 rows of the file. It can do it only in cases when the columns are aligned and correctly separated by the provided delimiter (default delimiter is whitespace).

read_fwf supports the dtype parameter for specifying the types of parsed columns to be different from the inferred type.

Consider a file with one less entry in the header than the number of data column:

In this special case, read_csv assumes that the first column is to be used as the index of the DataFrame:

Note that the dates weren’t automatically parsed. In that case you would need to do as before:

Suppose you have data indexed by two columns:

The index_col argument to read_csv can take a list of column numbers to turn multiple columns into a MultiIndex for the index of the returned object:

By specifying list of row locations for the header argument, you can read in a MultiIndex for the columns. Specifying non-consecutive rows will skip the intervening rows.

read_csv is also able to interpret a more common format of multi-columns indices.

If an index_col is not specified (e.g. you don’t have an index, or wrote it with df.to_csv(..., index=False), then any names on the columns index will be lost.

read_csv is capable of inferring delimited (not necessarily comma-separated) files, as pandas uses the csv.Sniffer class of the csv module. For this, you have to specify sep=None.

It’s best to use concat() to combine multiple files. See the cookbook for an example.

Suppose you wish to iterate through a (potentially very large) file lazily rather than reading the entire file into memory, such as the following:

By specifying a chunksize to read_csv, the return value will be an iterable object of type TextFileReader:

Changed in version 1.2: read_csv/json/sas return a context-manager when iterating through a file.

Specifying iterator=True will also return the TextFileReader object:

Pandas currently supports three engines, the C engine, the python engine, and an experimental pyarrow engine (requires the pyarrow package). In general, the pyarrow engine is fastest on larger workloads and is equivalent in speed to the C engine on most other workloads. The python engine tends to be slower than the pyarrow and C engines on most workloads. However, the pyarrow engine is much less robust than the C engine, which lacks a few features compared to the Python engine.

Where possible, pandas uses the C parser (specified as engine='c'), but it may fall back to Python if C-unsupported options are specified.

Currently, options unsupported by the C and pyarrow engines include:

sep other than a single character (e.g. regex separators)

sep=None with delim_whitespace=False

Specifying any of the above options will produce a ParserWarning unless the python engine is selected explicitly using engine='python'.

Options that are unsupported by the pyarrow engine which are not covered by the list above include:

infer_datetime_format

Specifying these options with engine='pyarrow' will raise a ValueError.

You can pass in a URL to read or write remote files to many of pandas’ IO functions - the following example shows reading a CSV file:

Added in version 1.3.0.

A custom header can be sent alongside HTTP(s) requests by passing a dictionary of header key value mappings to the storage_options keyword argument as shown below:

All URLs which are not local files or HTTP(s) are handled by fsspec, if installed, and its various filesystem implementations (including Amazon S3, Google Cloud, SSH, FTP, webHDFS…). Some of these implementations will require additional packages to be installed, for example S3 URLs require the s3fs library:

When dealing with remote storage systems, you might need extra configuration with environment variables or config files in special locations. For example, to access data in your S3 bucket, you will need to define credentials in one of the several ways listed in the S3Fs documentation. The same is true for several of the storage backends, and you should follow the links at fsimpl1 for implementations built into fsspec and fsimpl2 for those not included in the main fsspec distribution.

You can also pass parameters directly to the backend driver. Since fsspec does not utilize the AWS_S3_HOST environment variable, we can directly define a dictionary containing the endpoint_url and pass the object into the storage option parameter:

More sample configurations and documentation can be found at S3Fs documentation.

If you do not have S3 credentials, you can still access public data by specifying an anonymous connection, such as

Added in version 1.2.0.

fsspec also allows complex URLs, for accessing data in compressed archives, local caching of files, and more. To locally cache the above example, you would modify the call to

where we specify that the “anon” parameter is meant for the “s3” part of the implementation, not to the caching implementation. Note that this caches to a temporary directory for the duration of the session only, but you can also specify a permanent store.

The Series and DataFrame objects have an instance method to_csv which allows storing the contents of the object as a comma-separated-values file. The function takes a number of arguments. Only the first is required.

path_or_buf: A string path to the file to write or a file object. If a file object it must be opened with newline=''

sep : Field delimiter for the output file (default “,”)

na_rep: A string representation of a missing value (default ‘’)

float_format: Format string for floating point numbers

columns: Columns to write (default None)

header: Whether to write out the column names (default True)

index: whether to write row (index) names (default True)

index_label: Column label(s) for index column(s) if desired. If None (default), and header and index are True, then the index names are used. (A sequence should be given if the DataFrame uses MultiIndex).

mode : Python write mode, default ‘w’

encoding: a string representing the encoding to use if the contents are non-ASCII, for Python versions prior to 3

lineterminator: Character sequence denoting line end (default os.linesep)

quoting: Set quoting rules as in csv module (default csv.QUOTE_MINIMAL). Note that if you have set a float_format then floats are converted to strings and csv.QUOTE_NONNUMERIC will treat them as non-numeric

quotechar: Character used to quote fields (default ‘”’)

doublequote: Control quoting of quotechar in fields (default True)

escapechar: Character used to escape sep and quotechar when appropriate (default None)

chunksize: Number of rows to write at a time

date_format: Format string for datetime objects

The DataFrame object has an instance method to_string which allows control over the string representation of the object. All arguments are optional:

buf default None, for example a StringIO object

columns default None, which columns to write

col_space default None, minimum width of each column.

na_rep default NaN, representation of NA value

formatters default None, a dictionary (by column) of functions each of which takes a single argument and returns a formatted string

float_format default None, a function which takes a single (float) argument and returns a formatted string; to be applied to floats in the DataFrame.

sparsify default True, set to False for a DataFrame with a hierarchical index to print every MultiIndex key at each row.

index_names default True, will print the names of the indices

index default True, will print the index (ie, row labels)

header default True, will print the column labels

justify default left, will print column headers left- or right-justified

The Series object also has a to_string method, but with only the buf, na_rep, float_format arguments. There is also a length argument which, if set to True, will additionally output the length of the Series.

Read and write JSON format files and strings.

A Series or DataFrame can be converted to a valid JSON string. Use to_json with optional parameters:

path_or_buf : the pathname or buffer to write the output. This can be None in which case a JSON string is returned.

allowed values are {split, records, index}

allowed values are {split, records, index, columns, values, table}

The format of the JSON string

dict like {index -> [index]; columns -> [columns]; data -> [values]}

list like [{column -> value}; … ]

dict like {index -> {column -> value}}

dict like {column -> {index -> value}}

just the values array

adhering to the JSON Table Schema

date_format : string, type of date conversion, ‘epoch’ for timestamp, ‘iso’ for ISO8601.

double_precision : The number of decimal places to use when encoding floating point values, default 10.

force_ascii : force encoded string to be ASCII, default True.

date_unit : The time unit to encode to, governs timestamp and ISO8601 precision. One of ‘s’, ‘ms’, ‘us’ or ‘ns’ for seconds, milliseconds, microseconds and nanoseconds respectively. Default ‘ms’.

default_handler : The handler to call if an object cannot otherwise be converted to a suitable format for JSON. Takes a single argument, which is the object to convert, and returns a serializable object.

lines : If records orient, then will write each record per line as json.

mode : string, writer mode when writing to path. ‘w’ for write, ‘a’ for append. Default ‘w’

Note NaN’s, NaT’s and None will be converted to null and datetime objects will be converted based on the date_format and date_unit parameters.

There are a number of different options for the format of the resulting JSON file / string. Consider the following DataFrame and Series:

Column oriented (the default for DataFrame) serializes the data as nested JSON objects with column labels acting as the primary index:

Index oriented (the default for Series) similar to column oriented but the index labels are now primary:

Record oriented serializes the data to a JSON array of column -> value records, index labels are not included. This is useful for passing DataFrame data to plotting libraries, for example the JavaScript library d3.js:

Value oriented is a bare-bones option which serializes to nested JSON arrays of values only, column and index labels are not included:

Split oriented serializes to a JSON object containing separate entries for values, index and columns. Name is also included for Series:

Table oriented serializes to the JSON Table Schema, allowing for the preservation of metadata including but not limited to dtypes and index names.

Any orient option that encodes to a JSON object will not preserve the ordering of index and column labels during round-trip serialization. If you wish to preserve label ordering use the split option as it uses ordered containers.

Writing in ISO date format:

Writing in ISO date format, with microseconds:

Epoch timestamps, in seconds:

Writing to a file, with a date index and a date column:

If the JSON serializer cannot handle the container contents directly it will fall back in the following manner:

if the dtype is unsupported (e.g. np.complex_) then the default_handler, if provided, will be called for each value, otherwise an exception is raised.

if an object is unsupported it will attempt the following:

check if the object has defined a toDict method and call it. A toDict method should return a dict which will then be JSON serialized.

invoke the default_handler if one was provided.

convert the object to a dict by traversing its contents. However this will often fail with an OverflowError or give unexpected results.

In general the best approach for unsupported objects or dtypes is to provide a default_handler. For example:

can be dealt with by specifying a simple default_handler:

Reading a JSON string to pandas object can take a number of parameters. The parser will try to parse a DataFrame if typ is not supplied or is None. To explicitly force Series parsing, pass typ=series

filepath_or_buffer : a VALID JSON string or file handle / StringIO. The string could be a URL. Valid URL schemes include http, ftp, S3, and file. For file URLs, a host is expected. For instance, a local file could be file ://localhost/path/to/table.json

typ : type of object to recover (series or frame), default ‘frame’

allowed values are {split, records, index}

allowed values are {split, records, index, columns, values, table}

The format of the JSON string

dict like {index -> [index]; columns -> [columns]; data -> [values]}

list like [{column -> value} …]

dict like {index -> {column -> value}}

dict like {column -> {index -> value}}

just the values array

adhering to the JSON Table Schema

dtype : if True, infer dtypes, if a dict of column to dtype, then use those, if False, then don’t infer dtypes at all, default is True, apply only to the data.

convert_axes : boolean, try to convert the axes to the proper dtypes, default is True

convert_dates : a list of columns to parse for dates; If True, then try to parse date-like columns, default is True.

keep_default_dates : boolean, default True. If parsing dates, then parse the default date-like columns.

precise_float : boolean, default False. Set to enable usage of higher precision (strtod) function when decoding string to double values. Default (False) is to use fast but less precise builtin functionality.

date_unit : string, the timestamp unit to detect if converting dates. Default None. By default the timestamp precision will be detected, if this is not desired then pass one of ‘s’, ‘ms’, ‘us’ or ‘ns’ to force timestamp precision to seconds, milliseconds, microseconds or nanoseconds respectively.

lines : reads file as one json object per line.

encoding : The encoding to use to decode py3 bytes.

chunksize : when used in combination with lines=True, return a pandas.api.typing.JsonReader which reads in chunksize lines per iteration.

engine: Either "ujson", the built-in JSON parser, or "pyarrow" which dispatches to pyarrow’s pyarrow.json.read_json. The "pyarrow" is only available when lines=True

The parser will raise one of ValueError/TypeError/AssertionError if the JSON is not parseable.

If a non-default orient was used when encoding to JSON be sure to pass the same option here so that decoding produces sensible results, see Orient Options for an overview.

The default of convert_axes=True, dtype=True, and convert_dates=True will try to parse the axes, and all of the data into appropriate types, including dates. If you need to override specific dtypes, pass a dict to dtype. convert_axes should only be set to False if you need to preserve string-like numbers (e.g. ‘1’, ‘2’) in an axes.

Large integer values may be converted to dates if convert_dates=True and the data and / or column labels appear ‘date-like’. The exact threshold depends on the date_unit specified. ‘date-like’ means that the column label meets one of the following criteria:

it begins with 'timestamp'

When reading JSON data, automatic coercing into dtypes has some quirks:

an index can be reconstructed in a different order from serialization, that is, the returned order is not guaranteed to be the same as before serialization

a column that was float data will be converted to integer if it can be done safely, e.g. a column of 1.

bool columns will be converted to integer on reconstruction

Thus there are times where you may want to specify specific dtypes via the dtype keyword argument.

Reading from a JSON string:

Don’t convert any data (but still convert axes and dates):

Specify dtypes for conversion:

Preserve string indices:

Dates written in nanoseconds need to be read back in nanoseconds:

By setting the dtype_backend argument you can control the default dtypes used for the resulting DataFrame.

pandas provides a utility function to take a dict or list of dicts and normalize this semi-structured data into a flat table.

The max_level parameter provides more control over which level to end normalization. With max_level=1 the following snippet normalizes until 1st nesting level of the provided dict.

pandas is able to read and write line-delimited json files that are common in data processing pipelines using Hadoop or Spark.

For line-delimited json files, pandas can also return an iterator which reads in chunksize lines at a time. This can be useful for large files or to read from a stream.

Line-limited json can also be read using the pyarrow reader by specifying engine="pyarrow".

Added in version 2.0.0.

Table Schema is a spec for describing tabular datasets as a JSON object. The JSON includes information on the field names, types, and other attributes. You can use the orient table to build a JSON string with two fields, schema and data.

The schema field contains the fields key, which itself contains a list of column name to type pairs, including the Index or MultiIndex (see below for a list of types). The schema field also contains a primaryKey field if the (Multi)index is unique.

The second field, data, contains the serialized data with the records orient. The index is included, and any datetimes are ISO 8601 formatted, as required by the Table Schema spec.

The full list of types supported are described in the Table Schema spec. This table shows the mapping from pandas types:

A few notes on the generated table schema:

The schema object contains a pandas_version field. This contains the version of pandas’ dialect of the schema, and will be incremented with each revision.

All dates are converted to UTC when serializing. Even timezone naive values, which are treated as UTC with an offset of 0.

datetimes with a timezone (before serializing), include an additional field tz with the time zone name (e.g. 'US/Central').

Periods are converted to timestamps before serialization, and so have the same behavior of being converted to UTC. In addition, periods will contain and additional field freq with the period’s frequency, e.g. 'A-DEC'.

Categoricals use the any type and an enum constraint listing the set of possible values. Additionally, an ordered field is included:

A primaryKey field, containing an array of labels, is included if the index is unique:

The primaryKey behavior is the same with MultiIndexes, but in this case the primaryKey is an array:

The default naming roughly follows these rules:

For series, the object.name is used. If that’s none, then the name is values

For DataFrames, the stringified version of the column name is used

For Index (not MultiIndex), index.name is used, with a fallback to index if that is None.

For MultiIndex, mi.names is used. If any level has no name, then level_<i> is used.

read_json also accepts orient='table' as an argument. This allows for the preservation of metadata such as dtypes and index names in a round-trippable manner.

Please note that the literal string ‘index’ as the name of an Index is not round-trippable, nor are any names beginning with 'level_' within a MultiIndex. These are used by default in DataFrame.to_json() to indicate missing values and the subsequent read cannot distinguish the intent.

When using orient='table' along with user-defined ExtensionArray, the generated schema will contain an additional extDtype key in the respective fields element. This extra key is not standard but does enable JSON roundtrips for extension types (e.g. read_json(df.to_json(orient="table"), orient="table")).

The extDtype key carries the name of the extension, if you have properly registered the ExtensionDtype, pandas will use said name to perform a lookup into the registry and re-convert the serialized data into your custom dtype.

We highly encourage you to read the HTML Table Parsing gotchas below regarding the issues surrounding the BeautifulSoup4/html5lib/lxml parsers.

The top-level read_html() function can accept an HTML string/file/URL and will parse HTML tables into list of pandas DataFrames. Let’s look at a few examples.

read_html returns a list of DataFrame objects, even if there is only a single table contained in the HTML content.

Read a URL with no options:

The data from the above URL changes every Monday so the resulting data above may be slightly different.

Read a URL while passing headers alongside the HTTP request:

We see above that the headers we passed are reflected in the HTTP request.

Read in the content of the file from the above URL and pass it to read_html as a string:

You can even pass in an instance of StringIO if you so desire:

The following examples are not run by the IPython evaluator due to the fact that having so many network-accessing functions slows down the documentation build. If you spot an error or an example that doesn’t run, please do not hesitate to report it over on pandas GitHub issues page.

Read a URL and match a table that contains specific text:

Specify a header row (by default <th> or <td> elements located within a <thead> are used to form the column index, if multiple rows are contained within <thead> then a MultiIndex is created); if specified, the header row is taken from the data minus the parsed header elements (<th> elements).

Specify an index column:

Specify a number of rows to skip:

Specify a number of rows to skip using a list (range works as well):

Specify an HTML attribute:

Specify values that should be converted to NaN:

Specify whether to keep the default set of NaN values:

Specify converters for columns. This is useful for numerical text data that has leading zeros. By default columns that are numerical are cast to numeric types and the leading zeros are lost. To avoid this, we can convert these columns to strings.

Use some combination of the above:

Read in pandas to_html output (with some loss of floating point precision):

The lxml backend will raise an error on a failed parse if that is the only parser you provide. If you only have a single parser you can provide just a string, but it is considered good practice to pass a list with one string if, for example, the function expects a sequence of strings. You may use:

Or you could pass flavor='lxml' without a list:

However, if you have bs4 and html5lib installed and pass None or ['lxml', 'bs4'] then the parse will most likely succeed. Note that as soon as a parse succeeds, the function will return.

Links can be extracted from cells along with the text using extract_links="all".

Added in version 1.5.0.

DataFrame objects have an instance method to_html which renders the contents of the DataFrame as an HTML table. The function arguments are as in the method to_string described above.

Not all of the possible options for DataFrame.to_html are shown here for brevity’s sake. See DataFrame.to_html() for the full set of options.

In an HTML-rendering supported environment like a Jupyter Notebook, display(HTML(...))` will render the raw HTML into the environment.

The columns argument will limit the columns shown:

float_format takes a Python callable to control the precision of floating point values:

bold_rows will make the row labels bold by default, but you can turn that off:

The classes argument provides the ability to give the resulting HTML table CSS classes. Note that these classes are appended to the existing 'dataframe' class.

The render_links argument provides the ability to add hyperlinks to cells that contain URLs.

Finally, the escape argument allows you to control whether the “<”, “>” and “&” characters escaped in the resulting HTML (by default it is True). So to get the HTML without escaped characters pass escape=False

Some browsers may not show a difference in the rendering of the previous two HTML tables.

There are some versioning issues surrounding the libraries that are used to parse HTML tables in the top-level pandas io function read_html.

lxml requires Cython to install correctly.

lxml does not make any guarantees about the results of its parse unless it is given strictly valid markup.

In light of the above, we have chosen to allow you, the user, to use the lxml backend, but this backend will use html5lib if lxml fails to parse

It is therefore highly recommended that you install both BeautifulSoup4 and html5lib, so that you will still get a valid result (provided everything else is valid) even if lxml fails.

Issues with BeautifulSoup4 using lxml as a backend

The above issues hold here as well since BeautifulSoup4 is essentially just a wrapper around a parser backend.

Issues with BeautifulSoup4 using html5lib as a backend

html5lib is far more lenient than lxml and consequently deals with real-life markup in a much saner way rather than just, e.g., dropping an element without notifying you.

html5lib generates valid HTML5 markup from invalid markup automatically. This is extremely important for parsing HTML tables, since it guarantees a valid document. However, that does NOT mean that it is “correct”, since the process of fixing markup does not have a single definition.

html5lib is pure Python and requires no additional build steps beyond its own installation.

The biggest drawback to using html5lib is that it is slow as molasses. However consider the fact that many tables on the web are not big enough for the parsing algorithm runtime to matter. It is more likely that the bottleneck will be in the process of reading the raw text from the URL over the web, i.e., IO (input-output). For very large tables, this might not be true.

Added in version 1.3.0.

Currently there are no methods to read from LaTeX, only output methods.

DataFrame and Styler objects currently have a to_latex method. We recommend using the Styler.to_latex() method over DataFrame.to_latex() due to the former’s greater flexibility with conditional styling, and the latter’s possible future deprecation.

Review the documentation for Styler.to_latex, which gives examples of conditional styling and explains the operation of its keyword arguments.

For simple application the following pattern is sufficient.

To format values before output, chain the Styler.format method.

Added in version 1.3.0.

The top-level read_xml() function can accept an XML string/file/URL and will parse nodes and attributes into a pandas DataFrame.

Since there is no standard XML structure where design types can vary in many ways, read_xml works best with flatter, shallow versions. If an XML document is deeply nested, use the stylesheet feature to transform XML into a flatter version.

Let’s look at a few examples.

Read a URL with no options:

Read in the content of the “books.xml” file and pass it to read_xml as a string:

Read in the content of the “books.xml” as instance of StringIO or BytesIO and pass it to read_xml:

Even read XML from AWS S3 buckets such as NIH NCBI PMC Article Datasets providing Biomedical and Life Science Jorurnals:

With lxml as default parser, you access the full-featured XML library that extends Python’s ElementTree API. One powerful tool is ability to query nodes selectively or conditionally with more expressive XPath:

Specify only elements or only attributes to parse:

XML documents can have namespaces with prefixes and default namespaces without prefixes both of which are denoted with a special attribute xmlns. In order to parse by node under a namespace context, xpath must reference a prefix.

For example, below XML contains a namespace with prefix, doc, and URI at https://example.com. In order to parse doc:row nodes, namespaces must be used.

Similarly, an XML document can have a default namespace without prefix. Failing to assign a temporary prefix will return no nodes and raise a ValueError. But assigning any temporary name to correct URI allows parsing by nodes.

However, if XPath does not reference node names such as default, /*, then namespaces is not required.

Since xpath identifies the parent of content to be parsed, only immediate desendants which include child nodes or current attributes are parsed. Therefore, read_xml will not parse the text of grandchildren or other descendants and will not parse attributes of any descendant. To retrieve lower level content, adjust xpath to lower level. For example,

shows the attribute sides on shape element was not parsed as expected since this attribute resides on the child of row element and not row element itself. In other words, sides attribute is a grandchild level descendant of row element. However, the xpath targets row element which covers only its children and attributes.

With lxml as parser, you can flatten nested XML documents with an XSLT script which also can be string/file/URL types. As background, XSLT is a special-purpose language written in a special XML file that can transform original XML documents into other XML, HTML, even text (CSV, JSON, etc.) using an XSLT processor.

For example, consider this somewhat nested structure of Chicago “L” Rides where station and rides elements encapsulate data in their own sections. With below XSLT, lxml can transform original nested document into a flatter output (as shown below for demonstration) for easier parse into DataFrame:

For very large XML files that can range in hundreds of megabytes to gigabytes, pandas.read_xml() supports parsing such sizeable files using lxml’s iterparse and etree’s iterparse which are memory-efficient methods to iterate through an XML tree and extract specific elements and attributes. without holding entire tree in memory.

Added in version 1.5.0.

To use this feature, you must pass a physical XML file path into read_xml and use the iterparse argument. Files should not be compressed or point to online sources but stored on local disk. Also, iterparse should be a dictionary where the key is the repeating nodes in document (which become the rows) and the value is a list of any element or attribute that is a descendant (i.e., child, grandchild) of repeating node. Since XPath is not used in this method, descendants do not need to share same relationship with one another. Below shows example of reading in Wikipedia’s very large (12 GB+) latest article data dump.

Added in version 1.3.0.

DataFrame objects have an instance method to_xml which renders the contents of the DataFrame as an XML document.

This method does not support special properties of XML including DTD, CData, XSD schemas, processing instructions, comments, and others. Only namespaces at the root level is supported. However, stylesheet allows design changes after initial output.

Let’s look at a few examples.

Write an XML without options:

Write an XML with new root and row name:

Write an attribute-centric XML:

Write a mix of elements and attributes:

Any DataFrames with hierarchical columns will be flattened for XML element names with levels delimited by underscores:

Write an XML with default namespace:

Write an XML with namespace prefix:

Write an XML without declaration or pretty print:

Write an XML and transform with stylesheet:

All XML documents adhere to W3C specifications. Both etree and lxml parsers will fail to parse any markup document that is not well-formed or follows XML syntax rules. Do be aware HTML is not an XML document unless it follows XHTML specs. However, other popular markup types including KML, XAML, RSS, MusicML, MathML are compliant XML schemas.

For above reason, if your application builds XML prior to pandas operations, use appropriate DOM libraries like etree and lxml to build the necessary document and not by string concatenation or regex adjustments. Always remember XML is a special text file with markup rules.

With very large XML files (several hundred MBs to GBs), XPath and XSLT can become memory-intensive operations. Be sure to have enough available RAM for reading and writing to large XML files (roughly about 5 times the size of text).

Because XSLT is a programming language, use it with caution since such scripts can pose a security risk in your environment and can run large or infinite recursive operations. Always test scripts on small fragments before full run.

The etree parser supports all functionality of both read_xml and to_xml except for complex XPath and any XSLT. Though limited in features, etree is still a reliable and capable parser and tree builder. Its performance may trail lxml to a certain degree for larger files but relatively unnoticeable on small to medium size files.

The read_excel() method can read Excel 2007+ (.xlsx) files using the openpyxl Python module. Excel 2003 (.xls) files can be read using xlrd. Binary Excel (.xlsb) files can be read using pyxlsb. All formats can be read using calamine engine. The to_excel() instance method is used for saving a DataFrame to Excel. Generally the semantics are similar to working with csv data. See the cookbook for some advanced strategies.

When engine=None, the following logic will be used to determine the engine:

If path_or_buffer is an OpenDocument format (.odf, .ods, .odt), then odf will be used.

Otherwise if path_or_buffer is an xls format, xlrd will be used.

Otherwise if path_or_buffer is in xlsb format, pyxlsb will be used.

Otherwise openpyxl will be used.

In the most basic use-case, read_excel takes a path to an Excel file, and the sheet_name indicating which sheet to parse.

When using the engine_kwargs parameter, pandas will pass these arguments to the engine. For this, it is important to know which function pandas is using internally.

For the engine openpyxl, pandas is using openpyxl.load_workbook() to read in (.xlsx) and (.xlsm) files.

For the engine xlrd, pandas is using xlrd.open_workbook() to read in (.xls) files.

For the engine pyxlsb, pandas is using pyxlsb.open_workbook() to read in (.xlsb) files.

For the engine odf, pandas is using odf.opendocument.load() to read in (.ods) files.

For the engine calamine, pandas is using python_calamine.load_workbook() to read in (.xlsx), (.xlsm), (.xls), (.xlsb), (.ods) files.

To facilitate working with multiple sheets from the same file, the ExcelFile class can be used to wrap the file and can be passed into read_excel There will be a performance benefit for reading multiple sheets as the file is read into memory only once.

The ExcelFile class can also be used as a context manager.

The sheet_names property will generate a list of the sheet names in the file.

The primary use-case for an ExcelFile is parsing multiple sheets with different parameters:

Note that if the same parsing parameters are used for all sheets, a list of sheet names can simply be passed to read_excel with no loss in performance.

ExcelFile can also be called with a xlrd.book.Book object as a parameter. This allows the user to control how the excel file is read. For example, sheets can be loaded on demand by calling xlrd.open_workbook() with on_demand=True.

The second argument is sheet_name, not to be confused with ExcelFile.sheet_names.

An ExcelFile’s attribute sheet_names provides access to a list of sheets.

The arguments sheet_name allows specifying the sheet or sheets to read.

The default value for sheet_name is 0, indicating to read the first sheet

Pass a string to refer to the name of a particular sheet in the workbook.

Pass an integer to refer to the index of a sheet. Indices follow Python convention, beginning at 0.

Pass a list of either strings or integers, to return a dictionary of specified sheets.

Pass a None to return a dictionary of all available sheets.

Using the sheet index:

Using all default values:

Using None to get all sheets:

Using a list to get multiple sheets:

read_excel can read more than one sheet, by setting sheet_name to either a list of sheet names, a list of sheet positions, or None to read all sheets. Sheets can be specified by sheet index or sheet name, using an integer or string, respectively.

read_excel can read a MultiIndex index, by passing a list of columns to index_col and a MultiIndex column by passing a list of rows to header. If either the index or columns have serialized level names those will be read in as well by specifying the rows/columns that make up the levels.

For example, to read in a MultiIndex index without names:

If the index has level names, they will parsed as well, using the same parameters.

If the source file has both MultiIndex index and columns, lists specifying each should be passed to index_col and header:

Missing values in columns specified in index_col will be forward filled to allow roundtripping with to_excel for merged_cells=True. To avoid forward filling the missing values use set_index after reading the data instead of index_col.

It is often the case that users will insert columns to do temporary computations in Excel and you may not want to read in those columns. read_excel takes a usecols keyword to allow you to specify a subset of columns to parse.

You can specify a comma-delimited set of Excel columns and ranges as a string:

If usecols is a list of integers, then it is assumed to be the file column indices to be parsed.

Element order is ignored, so usecols=[0, 1] is the same as [1, 0].

If usecols is a list of strings, it is assumed that each string corresponds to a column name provided either by the user in names or inferred from the document header row(s). Those strings define which columns will be parsed:

Element order is ignored, so usecols=['baz', 'joe'] is the same as ['joe', 'baz'].

If usecols is callable, the callable function will be evaluated against the column names, returning names where the callable function evaluates to True.

Datetime-like values are normally automatically converted to the appropriate dtype when reading the excel file. But if you have a column of strings that look like dates (but are not actually formatted as dates in excel), you can use the parse_dates keyword to parse those strings to datetimes:

It is possible to transform the contents of Excel cells via the converters option. For instance, to convert a column to boolean:

This options handles missing values and treats exceptions in the converters as missing data. Transformations are applied cell by cell rather than to the column as a whole, so the array dtype is not guaranteed. For instance, a column of integers with missing values cannot be transformed to an array with integer dtype, because NaN is strictly a float. You can manually mask missing data to recover integer dtype:

As an alternative to converters, the type for an entire column can be specified using the dtype keyword, which takes a dictionary mapping column names to types. To interpret data with no type inference, use the type str or object.

To write a DataFrame object to a sheet of an Excel file, you can use the to_excel instance method. The arguments are largely the same as to_csv described above, the first argument being the name of the excel file, and the optional second argument the name of the sheet to which the DataFrame should be written. For example:

Files with a .xlsx extension will be written using xlsxwriter (if available) or openpyxl.

The DataFrame will be written in a way that tries to mimic the REPL output. The index_label will be placed in the second row instead of the first. You can place it in the first row by setting the merge_cells option in to_excel() to False:

In order to write separate DataFrames to separate sheets in a single Excel file, one can pass an ExcelWriter.

When using the engine_kwargs parameter, pandas will pass these arguments to the engine. For this, it is important to know which function pandas is using internally.

For the engine openpyxl, pandas is using openpyxl.Workbook() to create a new sheet and openpyxl.load_workbook() to append data to an existing sheet. The openpyxl engine writes to (.xlsx) and (.xlsm) files.

For the engine xlsxwriter, pandas is using xlsxwriter.Workbook() to write to (.xlsx) files.

For the engine odf, pandas is using odf.opendocument.OpenDocumentSpreadsheet() to write to (.ods) files.

pandas supports writing Excel files to buffer-like objects such as StringIO or BytesIO using ExcelWriter.

engine is optional but recommended. Setting the engine determines the version of workbook produced. Setting engine='xlrd' will produce an Excel 2003-format workbook (xls). Using either 'openpyxl' or 'xlsxwriter' will produce an Excel 2007-format workbook (xlsx). If omitted, an Excel 2007-formatted workbook is produced.

pandas chooses an Excel writer via two methods:

the engine keyword argument

the filename extension (via the default specified in config options)

By default, pandas uses the XlsxWriter for .xlsx, openpyxl for .xlsm. If you have multiple engines installed, you can set the default engine through setting the config options io.excel.xlsx.writer and io.excel.xls.writer. pandas will fall back on openpyxl for .xlsx files if Xlsxwriter is not available.

To specify which writer you want to use, you can pass an engine keyword argument to to_excel and to ExcelWriter. The built-in engines are:

openpyxl: version 2.4 or higher is required

The look and feel of Excel worksheets created from pandas can be modified using the following parameters on the DataFrame’s to_excel method.

float_format : Format string for floating point numbers (default None).

freeze_panes : A tuple of two integers representing the bottommost row and rightmost column to freeze. Each of these parameters is one-based, so (1, 1) will freeze the first row and first column (default None).

Using the Xlsxwriter engine provides many options for controlling the format of an Excel worksheet created with the to_excel method. Excellent examples can be found in the Xlsxwriter documentation here: https://xlsxwriter.readthedocs.io/working_with_pandas.html

The io methods for Excel files also support reading and writing OpenDocument spreadsheets using the odfpy module. The semantics and features for reading and writing OpenDocument spreadsheets match what can be done for Excel files using engine='odf'. The optional dependency ‘odfpy’ needs to be installed.

The read_excel() method can read OpenDocument spreadsheets

Similarly, the to_excel() method can write OpenDocument spreadsheets

The read_excel() method can also read binary Excel files using the pyxlsb module. The semantics and features for reading binary Excel files mostly match what can be done for Excel files using engine='pyxlsb'. pyxlsb does not recognize datetime types in files and will return floats instead (you can use calamine if you need recognize datetime types).

Currently pandas only supports reading binary Excel files. Writing is not implemented.

The read_excel() method can read Excel file (.xlsx, .xlsm, .xls, .xlsb) and OpenDocument spreadsheets (.ods) using the python-calamine module. This module is a binding for Rust library calamine and is faster than other engines in most cases. The optional dependency ‘python-calamine’ needs to be installed.

A handy way to grab data is to use the read_clipboard() method, which takes the contents of the clipboard buffer and passes them to the read_csv method. For instance, you can copy the following text to the clipboard (CTRL-C on many operating systems):

And then import the data directly to a DataFrame by calling:

The to_clipboard method can be used to write the contents of a DataFrame to the clipboard. Following which you can paste the clipboard contents into other applications (CTRL-V on many operating systems). Here we illustrate writing a DataFrame into clipboard and reading it back.

We can see that we got the same content back, which we had earlier written to the clipboard.

You may need to install xclip or xsel (with PyQt5, PyQt4 or qtpy) on Linux to use these methods.

All pandas objects are equipped with to_pickle methods which use Python’s cPickle module to save data structures to disk using the pickle format.

The read_pickle function in the pandas namespace can be used to load any pickled pandas object (or any other pickled object) from file:

Loading pickled data received from untrusted sources can be unsafe.

See: https://docs.python.org/3/library/pickle.html

read_pickle() is only guaranteed backwards compatible back to a few minor release.

read_pickle(), DataFrame.to_pickle() and Series.to_pickle() can read and write compressed pickle files. The compression types of gzip, bz2, xz, zstd are supported for reading and writing. The zip file format only supports reading and must contain only one data file to be read.

The compression type can be an explicit parameter or be inferred from the file extension. If ‘infer’, then use gzip, bz2, zip, xz, zstd if filename ends in '.gz', '.bz2', '.zip', '.xz', or '.zst', respectively.

The compression parameter can also be a dict in order to pass options to the compression protocol. It must have a 'method' key set to the name of the compression protocol, which must be one of {'zip', 'gzip', 'bz2', 'xz', 'zstd'}. All other key-value pairs are passed to the underlying compression library.

Using an explicit compression type:

Inferring compression type from the extension:

The default is to ‘infer’:

Passing options to the compression protocol in order to speed up compression:

pandas support for msgpack has been removed in version 1.0.0. It is recommended to use pickle instead.

Alternatively, you can also the Arrow IPC serialization format for on-the-wire transmission of pandas objects. For documentation on pyarrow, see here.

HDFStore is a dict-like object which reads and writes pandas using the high performance HDF5 format using the excellent PyTables library. See the cookbook for some advanced strategies

pandas uses PyTables for reading and writing HDF5 files, which allows serializing object-dtype data with pickle. Loading pickled data received from untrusted sources can be unsafe.

See: https://docs.python.org/3/library/pickle.html for more.

Objects can be written to the file just like adding key-value pairs to a dict:

In a current or later Python session, you can retrieve stored objects:

Deletion of the object specified by the key:

Closing a Store and using a context manager:

HDFStore supports a top-level API using read_hdf for reading and to_hdf for writing, similar to how read_csv and to_csv work.

HDFStore will by default not drop rows that are all missing. This behavior can be changed by setting dropna=True.

The examples above show storing using put, which write the HDF5 to PyTables in a fixed array format, called the fixed format. These types of stores are not appendable once written (though you can simply remove them and rewrite). Nor are they queryable; they must be retrieved in their entirety. They also do not support dataframes with non-unique column names. The fixed format stores offer very fast writing and slightly faster reading than table stores. This format is specified by default when using put or to_hdf or by format='fixed' or format='f'.

A fixed format will raise a TypeError if you try to retrieve using a where:

HDFStore supports another PyTables format on disk, the table format. Conceptually a table is shaped very much like a DataFrame, with rows and columns. A table may be appended to in the same or other sessions. In addition, delete and query type operations are supported. This format is specified by format='table' or format='t' to append or put or to_hdf.

This format can be set as an option as well pd.set_option('io.hdf.default_format','table') to enable put/append/to_hdf to by default store in the table format.

You can also create a table by passing format='table' or format='t' to a put operation.

Keys to a store can be specified as a string. These can be in a hierarchical path-name like format (e.g. foo/bar/bah), which will generate a hierarchy of sub-stores (or Groups in PyTables parlance). Keys can be specified without the leading ‘/’ and are always absolute (e.g. ‘foo’ refers to ‘/foo’). Removal operations can remove everything in the sub-store and below, so be careful.

You can walk through the group hierarchy using the walk method which will yield a tuple for each group key along with the relative keys of its contents.

Hierarchical keys cannot be retrieved as dotted (attribute) access as described above for items stored under the root node.

Instead, use explicit string based keys:

Storing mixed-dtype data is supported. Strings are stored as a fixed-width using the maximum size of the appended column. Subsequent attempts at appending longer strings will raise a ValueError.

Passing min_itemsize={`values`: size} as a parameter to append will set a larger minimum for the string columns. Storing floats, strings, ints, bools, datetime64 are currently supported. For string columns, passing nan_rep = 'nan' to append will change the default nan representation on disk (which converts to/from np.nan), this defaults to nan.

Storing MultiIndex DataFrames as tables is very similar to storing/selecting from homogeneous index DataFrames.

The index keyword is reserved and cannot be use as a level name.

select and delete operations have an optional criterion that can be specified to select/delete only a subset of the data. This allows one to have a very large on-disk table and retrieve only a portion of the data.

A query is specified using the Term class under the hood, as a boolean expression.

index and columns are supported indexers of DataFrames.

if data_columns are specified, these can be used as additional indexers.

level name in a MultiIndex, with default name level_0, level_1, … if not provided.

Valid comparison operators are:

=, ==, !=, >, >=, <, <=

Valid boolean expressions are combined with:

( and ) : for grouping

These rules are similar to how boolean expressions are used in pandas for indexing.

= will be automatically expanded to the comparison operator ==

~ is the not operator, but can only be used in very limited circumstances

If a list/tuple of expressions is passed they will be combined via &

The following are valid expressions:

"columns = ['A', 'D']"

"columns in ['A', 'D']"

"~(columns = ['A', 'B'])"

'index > df.index[3] & string = "bar"'

'(index > df.index[3] & index <= df.index[6]) | string = "bar"'

"ts >= Timestamp('2012-02-01')"

"major_axis>=20130101"

The indexers are on the left-hand side of the sub-expression:

columns, major_axis, ts

The right-hand side of the sub-expression (after a comparison operator) can be:

functions that will be evaluated, e.g. Timestamp('2012-02-01')

date-like, e.g. 20130101, or "20130101"

lists, e.g. "['A', 'B']"

variables that are defined in the local names space, e.g. date

Passing a string to a query by interpolating it into the query expression is not recommended. Simply assign the string of interest to a variable and use that variable in an expression. For example, do this

The latter will not work and will raise a SyntaxError.Note that there’s a single quote followed by a double quote in the string variable.

If you must interpolate, use the '%r' format specifier

which will quote string.

Here are some examples:

Use boolean expressions, with in-line function evaluation.

Use inline column reference.

The columns keyword can be supplied to select a list of columns to be returned, this is equivalent to passing a 'columns=list_of_columns_to_filter':

start and stop parameters can be specified to limit the total search space. These are in terms of the total number of rows in a table.

select will raise a ValueError if the query expression has an unknown variable reference. Usually this means that you are trying to select on a column that is not a data_column.

select will raise a SyntaxError if the query expression is not valid.

You can store and query using the timedelta64[ns] type. Terms can be specified in the format: <float>(<unit>), where float may be signed (and fractional), and unit can be D,s,ms,us,ns for the timedelta. Here’s an example:

Selecting from a MultiIndex can be achieved by using the name of the level.

If the MultiIndex levels names are None, the levels are automatically made available via the level_n keyword with n the level of the MultiIndex you want to select from.

You can create/modify an index for a table with create_table_index after data is already in the table (after and append/put operation). Creating a table index is highly encouraged. This will speed your queries a great deal when you use a select with the indexed dimension as the where.

Indexes are automagically created on the indexables and any data columns you specify. This behavior can be turned off by passing index=False to append.

Oftentimes when appending large amounts of data to a store, it is useful to turn off index creation for each append, then recreate at the end.

Then create the index when finished appending.

See here for how to create a completely-sorted-index (CSI) on an existing store.

You can designate (and index) certain columns that you want to be able to perform queries (other than the indexable columns, which you can always query). For instance say you want to perform this common operation, on-disk, and return just the frame that matches this query. You can specify data_columns = True to force all columns to be data_columns.

There is some performance degradation by making lots of columns into data columns, so it is up to the user to designate these. In addition, you cannot change data columns (nor indexables) after the first append/put operation (Of course you can simply read in the data and create a new table!).

You can pass iterator=True or chunksize=number_in_a_chunk to select and select_as_multiple to return an iterator on the results. The default is 50,000 rows returned in a chunk.

You can also use the iterator with read_hdf which will open, then automatically close the store when finished iterating.

Note, that the chunksize keyword applies to the source rows. So if you are doing a query, then the chunksize will subdivide the total rows in the table and the query applied, returning an iterator on potentially unequal sized chunks.

Here is a recipe for generating a query and using it to create equal sized return chunks.

To retrieve a single indexable or data column, use the method select_column. This will, for example, enable you to get the index very quickly. These return a Series of the result, indexed by the row number. These do not currently accept the where selector.

Sometimes you want to get the coordinates (a.k.a the index locations) of your query. This returns an Index of the resulting locations. These coordinates can also be passed to subsequent where operations.

Sometime your query can involve creating a list of rows to select. Usually this mask would be a resulting index from an indexing operation. This example selects the months of a datetimeindex which are 5.

If you want to inspect the stored object, retrieve via get_storer. You could use this programmatically to say get the number of rows in an object.

The methods append_to_multiple and select_as_multiple can perform appending/selecting from multiple tables at once. The idea is to have one table (call it the selector table) that you index most/all of the columns, and perform your queries. The other table(s) are data tables with an index matching the selector table’s index. You can then perform a very fast query on the selector table, yet get lots of data back. This method is similar to having a very wide table, but enables more efficient queries.

The append_to_multiple method splits a given single DataFrame into multiple tables according to d, a dictionary that maps the table names to a list of ‘columns’ you want in that table. If None is used in place of a list, that table will have the remaining unspecified columns of the given DataFrame. The argument selector defines which table is the selector table (which you can make queries from). The argument dropna will drop rows from the input DataFrame to ensure tables are synchronized. This means that if a row for one of the tables being written to is entirely np.nan, that row will be dropped from all tables.

If dropna is False, THE USER IS RESPONSIBLE FOR SYNCHRONIZING THE TABLES. Remember that entirely np.Nan rows are not written to the HDFStore, so if you choose to call dropna=False, some tables may have more rows than others, and therefore select_as_multiple may not work or it may return unexpected results.

You can delete from a table selectively by specifying a where. In deleting rows, it is important to understand the PyTables deletes rows by erasing the rows, then moving the following data. Thus deleting can potentially be a very expensive operation depending on the orientation of your data. To get optimal performance, it’s worthwhile to have the dimension you are deleting be the first of the indexables.

Data is ordered (on the disk) in terms of the indexables. Here’s a simple use case. You store panel-type data, with dates in the major_axis and ids in the minor_axis. The data is then interleaved like this:

It should be clear that a delete operation on the major_axis will be fairly quick, as one chunk is removed, then the following data moved. On the other hand a delete operation on the minor_axis will be very expensive. In this case it would almost certainly be faster to rewrite the table using a where that selects all but the missing data.

Please note that HDF5 DOES NOT RECLAIM SPACE in the h5 files automatically. Thus, repeatedly deleting (or removing nodes) and adding again, WILL TEND TO INCREASE THE FILE SIZE.

To repack and clean the file, use ptrepack.

PyTables allows the stored data to be compressed. This applies to all kinds of stores, not just tables. Two parameters are used to control compression: complevel and complib.

complevel specifies if and how hard data is to be compressed. complevel=0 and complevel=None disables compression and 0<complevel<10 enables compression.

complib specifies which compression library to use. If nothing is specified the default library zlib is used. A compression library usually optimizes for either good compression rates or speed and the results will depend on the type of data. Which type of compression to choose depends on your specific needs and data. The list of supported compression libraries:

zlib: The default compression library. A classic in terms of compression, achieves good compression rates but is somewhat slow.

lzo: Fast compression and decompression.

bzip2: Good compression rates.

blosc: Fast compression and decompression.

Support for alternative blosc compressors:

blosc:blosclz This is the default compressor for blosc

blosc:lz4: A compact, very popular and fast compressor.

blosc:lz4hc: A tweaked version of LZ4, produces better compression ratios at the expense of speed.

blosc:snappy: A popular compressor used in many places.

blosc:zlib: A classic; somewhat slower than the previous ones, but achieving better compression ratios.

blosc:zstd: An extremely well balanced codec; it provides the best compression ratios among the others above, and at reasonably fast speed.

If complib is defined as something other than the listed libraries a ValueError exception is issued.

If the library specified with the complib option is missing on your platform, compression defaults to zlib without further ado.

Enable compression for all objects within the file:

Or on-the-fly compression (this only applies to tables) in stores where compression is not enabled:

PyTables offers better write performance when tables are compressed after they are written, as opposed to turning on compression at the very beginning. You can use the supplied PyTables utility ptrepack. In addition, ptrepack can change compression levels after the fact.

Furthermore ptrepack in.h5 out.h5 will repack the file to allow you to reuse previously deleted space. Alternatively, one can simply remove the file and write again, or use the copy method.

HDFStore is not-threadsafe for writing. The underlying PyTables only supports concurrent reads (via threading or processes). If you need reading and writing at the same time, you need to serialize these operations in a single thread in a single process. You will corrupt your data otherwise. See the (GH 2397) for more information.

If you use locks to manage write access between multiple processes, you may want to use fsync() before releasing write locks. For convenience you can use store.flush(fsync=True) to do this for you.

Once a table is created columns (DataFrame) are fixed; only exactly the same columns can be appended

Be aware that timezones (e.g., pytz.timezone('US/Eastern')) are not necessarily equal across timezone versions. So if data is localized to a specific timezone in the HDFStore using one version of a timezone library and that data is updated with another version, the data will be converted to UTC since these timezones are not considered equal. Either use the same version of timezone library or use tz_convert with the updated timezone definition.

PyTables will show a NaturalNameWarning if a column name cannot be used as an attribute selector. Natural identifiers contain only letters, numbers, and underscores, and may not begin with a number. Other identifiers cannot be used in a where clause and are generally a bad idea.

HDFStore will map an object dtype to the PyTables underlying dtype. This means the following types are known to work:

Represents missing values

floating : float64, float32, float16

integer : int64, int32, int8, uint64,uint32, uint8

categorical : see the section below

unicode columns are not supported, and WILL FAIL.

You can write data that contains category dtypes to a HDFStore. Queries work the same as if it was an object array. However, the category dtyped data is stored in a more efficient manner.

The underlying implementation of HDFStore uses a fixed column width (itemsize) for string columns. A string column itemsize is calculated as the maximum of the length of data (for that column) that is passed to the HDFStore, in the first append. Subsequent appends, may introduce a string for a column larger than the column can hold, an Exception will be raised (otherwise you could have a silent truncation of these columns, leading to loss of information). In the future we may relax this and allow a user-specified truncation to occur.

Pass min_itemsize on the first table creation to a-priori specify the minimum length of a particular string column. min_itemsize can be an integer, or a dict mapping a column name to an integer. You can pass values as a key to allow all indexables or data_columns to have this min_itemsize.

Passing a min_itemsize dict will cause all passed columns to be created as data_columns automatically.

If you are not passing any data_columns, then the min_itemsize will be the maximum of the length of any string passed

String columns will serialize a np.nan (a missing value) with the nan_rep string representation. This defaults to the string value nan. You could inadvertently turn an actual nan value into a missing value.

tables format come with a writing performance penalty as compared to fixed stores. The benefit is the ability to append/delete and query (potentially very large amounts of data). Write times are generally longer as compared with regular stores. Query times can be quite fast, especially on an indexed axis.

You can pass chunksize=<int> to append, specifying the write chunksize (default is 50000). This will significantly lower your memory usage on writing.

You can pass expectedrows=<int> to the first append, to set the TOTAL number of rows that PyTables will expect. This will optimize read/write performance.

Duplicate rows can be written to tables, but are filtered out in selection (with the last items being selected; thus a table is unique on major, minor pairs)

A PerformanceWarning will be raised if you are attempting to store types that will be pickled by PyTables (rather than stored as endemic types). See Here for more information and some solutions.

Feather provides binary columnar serialization for data frames. It is designed to make reading and writing data frames efficient, and to make sharing data across data analysis languages easy.

Feather is designed to faithfully serialize and de-serialize DataFrames, supporting all of the pandas dtypes, including extension dtypes such as categorical and datetime with tz.

The format will NOT write an Index, or MultiIndex for the DataFrame and will raise an error if a non-default one is provided. You can .reset_index() to store the index or .reset_index(drop=True) to ignore it.

Duplicate column names and non-string columns names are not supported

Actual Python objects in object dtype columns are not supported. These will raise a helpful error message on an attempt at serialization.

See the Full Documentation.

Write to a feather file.

Read from a feather file.

Apache Parquet provides a partitioned binary columnar serialization for data frames. It is designed to make reading and writing data frames efficient, and to make sharing data across data analysis languages easy. Parquet can use a variety of compression techniques to shrink the file size as much as possible while still maintaining good read performance.

Parquet is designed to faithfully serialize and de-serialize DataFrame s, supporting all of the pandas dtypes, including extension dtypes such as datetime with tz.

Duplicate column names and non-string columns names are not supported.

The pyarrow engine always writes the index to the output, but fastparquet only writes non-default indexes. This extra column can cause problems for non-pandas consumers that are not expecting it. You can force including or omitting indexes with the index argument, regardless of the underlying engine.

Index level names, if specified, must be strings.

In the pyarrow engine, categorical dtypes for non-string types can be serialized to parquet, but will de-serialize as their primitive dtype.

The pyarrow engine preserves the ordered flag of categorical dtypes with string types. fastparquet does not preserve the ordered flag.

Non supported types include Interval and actual Python object types. These will raise a helpful error message on an attempt at serialization. Period type is supported with pyarrow >= 0.16.0.

The pyarrow engine preserves extension data types such as the nullable integer and string data type (requiring pyarrow >= 0.16.0, and requiring the extension type to implement the needed protocols, see the extension types documentation).

You can specify an engine to direct the serialization. This can be one of pyarrow, or fastparquet, or auto. If the engine is NOT specified, then the pd.options.io.parquet.engine option is checked; if this is also auto, then pyarrow is tried, and falling back to fastparquet.

See the documentation for pyarrow and fastparquet.

These engines are very similar and should read/write nearly identical parquet format files. pyarrow>=8.0.0 supports timedelta data, fastparquet>=0.1.4 supports timezone aware datetimes. These libraries differ by having different underlying dependencies (fastparquet by using numba, while pyarrow uses a c-library).

Write to a parquet file.

Read from a parquet file.

By setting the dtype_backend argument you can control the default dtypes used for the resulting DataFrame.

Note that this is not supported for fastparquet.

Read only certain columns of a parquet file.

Serializing a DataFrame to parquet may include the implicit index as one or more columns in the output file. Thus, this code:

creates a parquet file with three columns if you use pyarrow for serialization: a, b, and __index_level_0__. If you’re using fastparquet, the index may or may not be written to the file.

This unexpected extra column causes some databases like Amazon Redshift to reject the file, because that column doesn’t exist in the target table.

If you want to omit a dataframe’s indexes when writing, pass index=False to to_parquet():

This creates a parquet file with just the two expected columns, a and b. If your DataFrame has a custom index, you won’t get it back when you load this file into a DataFrame.

Passing index=True will always write the index, even if that’s not the underlying engine’s default behavior.

Parquet supports partitioning of data based on the values of one or more columns.

The path specifies the parent directory to which data will be saved. The partition_cols are the column names by which the dataset will be partitioned. Columns are partitioned in the order they are given. The partition splits are determined by the unique values in the partition columns. The above example creates a partitioned dataset that may look like:

Similar to the parquet format, the ORC Format is a binary columnar serialization for data frames. It is designed to make reading data frames efficient. pandas provides both the reader and the writer for the ORC format, read_orc() and to_orc(). This requires the pyarrow library.

It is highly recommended to install pyarrow using conda due to some issues occurred by pyarrow.

to_orc() requires pyarrow>=7.0.0.

read_orc() and to_orc() are not supported on Windows yet, you can find valid environments on install optional dependencies.

For supported dtypes please refer to supported ORC features in Arrow.

Currently timezones in datetime columns are not preserved when a dataframe is converted into ORC files.

Write to an orc file.

Read from an orc file.

Read only certain columns of an orc file.

The pandas.io.sql module provides a collection of query wrappers to both facilitate data retrieval and to reduce dependency on DB-specific API.

Where available, users may first want to opt for Apache Arrow ADBC drivers. These drivers should provide the best performance, null handling, and type detection.

Added in version 2.2.0: Added native support for ADBC drivers

For a full list of ADBC drivers and their development status, see the ADBC Driver Implementation Status documentation.

Where an ADBC driver is not available or may be missing functionality, users should opt for installing SQLAlchemy alongside their database driver library. Examples of such drivers are psycopg2 for PostgreSQL or pymysql for MySQL. For SQLite this is included in Python’s standard library by default. You can find an overview of supported drivers for each SQL dialect in the SQLAlchemy docs.

If SQLAlchemy is not installed, you can use a sqlite3.Connection in place of a SQLAlchemy engine, connection, or URI string.

See also some cookbook examples for some advanced strategies.

The key functions are:

read_sql_table(table_name, con[, schema, ...])

Read SQL database table into a DataFrame.

read_sql_query(sql, con[, index_col, ...])

Read SQL query into a DataFrame.

read_sql(sql, con[, index_col, ...])

Read SQL query or database table into a DataFrame.

DataFrame.to_sql(name, con, *[, schema, ...])

Write records stored in a DataFrame to a SQL database.

The function read_sql() is a convenience wrapper around read_sql_table() and read_sql_query() (and for backward compatibility) and will delegate to specific function depending on the provided input (database table name or sql query). Table names do not need to be quoted if they have special characters.

In the following example, we use the SQlite SQL database engine. You can use a temporary SQLite database where data are stored in “memory”.

To connect using an ADBC driver you will want to install the adbc_driver_sqlite using your package manager. Once installed, you can use the DBAPI interface provided by the ADBC driver to connect to your database.

To connect with SQLAlchemy you use the create_engine() function to create an engine object from database URI. You only need to create the engine once per database you are connecting to. For more information on create_engine() and the URI formatting, see the examples below and the SQLAlchemy documentation

If you want to manage your own connections you can pass one of those instead. The example below opens a connection to the database using a Python context manager that automatically closes the connection after the block has completed. See the SQLAlchemy docs for an explanation of how the database connection is handled.

When you open a connection to a database you are also responsible for closing it. Side effects of leaving a connection open may include locking the database or other breaking behaviour.

Assuming the following data is in a DataFrame data, we can insert it into the database using to_sql().

With some databases, writing large DataFrames can result in errors due to packet size limitations being exceeded. This can be avoided by setting the chunksize parameter when calling to_sql. For example, the following writes data to the database in batches of 1000 rows at a time:

Ensuring consistent data type management across SQL databases is challenging. Not every SQL database offers the same types, and even when they do the implementation of a given type can vary in ways that have subtle effects on how types can be preserved.

For the best odds at preserving database types users are advised to use ADBC drivers when available. The Arrow type system offers a wider array of types that more closely match database types than the historical pandas/NumPy type system. To illustrate, note this (non-exhaustive) listing of types available in different databases and pandas backends:

month_day_nano_interval

Not implemented as of writing, but theoretically possible

If you are interested in preserving database types as best as possible throughout the lifecycle of your DataFrame, users are encouraged to leverage the dtype_backend="pyarrow" argument of read_sql()

This will prevent your data from being converted to the traditional pandas/NumPy type system, which often converts SQL types in ways that make them impossible to round-trip.

In case an ADBC driver is not available, to_sql() will try to map your data to an appropriate SQL data type based on the dtype of the data. When you have columns of dtype object, pandas will try to infer the data type.

You can always override the default type by specifying the desired SQL type of any of the columns by using the dtype argument. This argument needs a dictionary mapping column names to SQLAlchemy types (or strings for the sqlite3 fallback mode). For example, specifying to use the sqlalchemy String type instead of the default Text type for string columns:

Due to the limited support for timedelta’s in the different database flavors, columns with type timedelta64 will be written as integer values as nanoseconds to the database and a warning will be raised. The only exception to this is when using the ADBC PostgreSQL driver in which case a timedelta will be written to the database as an INTERVAL

Columns of category dtype will be converted to the dense representation as you would get with np.asarray(categorical) (e.g. for string categories this gives an array of strings). Because of this, reading the database table back in does not generate a categorical.

Using ADBC or SQLAlchemy, to_sql() is capable of writing datetime data that is timezone naive or timezone aware. However, the resulting data stored in the database ultimately depends on the supported data type for datetime data of the database system being used.

The following table lists supported data types for datetime data for some common databases. Other database dialects may have different data types for datetime data.

TIMESTAMP or DATETIME

TIMESTAMP or TIMESTAMP WITH TIME ZONE

When writing timezone aware data to databases that do not support timezones, the data will be written as timezone naive timestamps that are in local time with respect to the timezone.

read_sql_table() is also capable of reading datetime data that is timezone aware or naive. When reading TIMESTAMP WITH TIME ZONE types, pandas will convert the data to UTC.

The parameter method controls the SQL insertion clause used. Possible values are:

None: Uses standard SQL INSERT clause (one per row).

'multi': Pass multiple values in a single INSERT clause. It uses a special SQL syntax not supported by all backends. This usually provides better performance for analytic databases like Presto and Redshift, but has worse performance for traditional SQL backend if the table contains many columns. For more information check the SQLAlchemy documentation.

callable with signature (pd_table, conn, keys, data_iter): This can be used to implement a more performant insertion method based on specific backend dialect features.

Example of a callable using PostgreSQL COPY clause:

read_sql_table() will read a database table given the table name and optionally a subset of columns to read.

In order to use read_sql_table(), you must have the ADBC driver or SQLAlchemy optional dependency installed.

ADBC drivers will map database types directly back to arrow types. For other drivers note that pandas infers column dtypes from query outputs, and not by looking up data types in the physical database schema. For example, assume userid is an integer column in a table. Then, intuitively, select userid ... will return integer-valued series, while select cast(userid as text) ... will return object-valued (str) series. Accordingly, if the query output is empty, then all resulting columns will be returned as object-valued (since they are most general). If you foresee that your query will sometimes generate an empty result, you may want to explicitly typecast afterwards to ensure dtype integrity.

You can also specify the name of the column as the DataFrame index, and specify a subset of columns to be read.

And you can explicitly force columns to be parsed as dates:

If needed you can explicitly specify a format string, or a dict of arguments to pass to pandas.to_datetime():

You can check if a table exists using has_table()

Reading from and writing to different schema’s is supported through the schema keyword in the read_sql_table() and to_sql() functions. Note however that this depends on the database flavor (sqlite does not have schema’s). For example:

You can query using raw SQL in the read_sql_query() function. In this case you must use the SQL variant appropriate for your database. When using SQLAlchemy, you can also pass SQLAlchemy Expression language constructs, which are database-agnostic.

Of course, you can specify a more “complex” query.

The read_sql_query() function supports a chunksize argument. Specifying this will return an iterator through chunks of the query result:

To connect with SQLAlchemy you use the create_engine() function to create an engine object from database URI. You only need to create the engine once per database you are connecting to.

For more information see the examples the SQLAlchemy documentation

You can use SQLAlchemy constructs to describe your query.

Use sqlalchemy.text() to specify query parameters in a backend-neutral way

If you have an SQLAlchemy description of your database you can express where conditions using SQLAlchemy expressions

You can combine SQLAlchemy expressions with parameters passed to read_sql() using sqlalchemy.bindparam()

The use of sqlite is supported without using SQLAlchemy. This mode requires a Python database adapter which respect the Python DB-API.

You can create connections like so:

And then issue the following queries:

The pandas-gbq package provides functionality to read/write from Google BigQuery.

pandas integrates with this external package. if pandas-gbq is installed, you can use the pandas methods pd.read_gbq and DataFrame.to_gbq, which will call the respective functions from pandas-gbq.

Full documentation can be found here.

The method DataFrame.to_stata() will write a DataFrame into a .dta file. The format version of this file is always 115 (Stata 12).

Stata data files have limited data type support; only strings with 244 or fewer characters, int8, int16, int32, float32 and float64 can be stored in .dta files. Additionally, Stata reserves certain values to represent missing data. Exporting a non-missing value that is outside of the permitted range in Stata for a particular data type will retype the variable to the next larger size. For example, int8 values are restricted to lie between -127 and 100 in Stata, and so variables with values above 100 will trigger a conversion to int16. nan values in floating points data types are stored as the basic missing data type (. in Stata).

It is not possible to export missing data values for integer data types.

The Stata writer gracefully handles other data types including int64, bool, uint8, uint16, uint32 by casting to the smallest supported type that can represent the data. For example, data with a type of uint8 will be cast to int8 if all values are less than 100 (the upper bound for non-missing int8 data in Stata), or, if values are outside of this range, the variable is cast to int16.

Conversion from int64 to float64 may result in a loss of precision if int64 values are larger than 2**53.

StataWriter and DataFrame.to_stata() only support fixed width strings containing up to 244 characters, a limitation imposed by the version 115 dta file format. Attempting to write Stata dta files with strings longer than 244 characters raises a ValueError.

The top-level function read_stata will read a dta file and return either a DataFrame or a pandas.api.typing.StataReader that can be used to read the file incrementally.

Specifying a chunksize yields a pandas.api.typing.StataReader instance that can be used to read chunksize lines from the file at a time. The StataReader object can be used as an iterator.

For more fine-grained control, use iterator=True and specify chunksize with each call to read().

Currently the index is retrieved as a column.

The parameter convert_categoricals indicates whether value labels should be read and used to create a Categorical variable from them. Value labels can also be retrieved by the function value_labels, which requires read() to be called before use.

The parameter convert_missing indicates whether missing value representations in Stata should be preserved. If False (the default), missing values are represented as np.nan. If True, missing values are represented using StataMissingValue objects, and columns containing missing values will have object data type.

read_stata() and StataReader support .dta formats 113-115 (Stata 10-12), 117 (Stata 13), and 118 (Stata 14).

Setting preserve_dtypes=False will upcast to the standard pandas data types: int64 for all integer types and float64 for floating point data. By default, the Stata data types are preserved when importing.

All StataReader objects, whether created by read_stata() (when using iterator=True or chunksize) or instantiated by hand, must be used as context managers (e.g. the with statement). While the close() method is available, its use is unsupported. It is not part of the public API and will be removed in with future without warning.

Categorical data can be exported to Stata data files as value labeled data. The exported data consists of the underlying category codes as integer data values and the categories as value labels. Stata does not have an explicit equivalent to a Categorical and information about whether the variable is ordered is lost when exporting.

Stata only supports string value labels, and so str is called on the categories when exporting data. Exporting Categorical variables with non-string categories produces a warning, and can result a loss of information if the str representations of the categories are not unique.

Labeled data can similarly be imported from Stata data files as Categorical variables using the keyword argument convert_categoricals (True by default). The keyword argument order_categoricals (True by default) determines whether imported Categorical variables are ordered.

When importing categorical data, the values of the variables in the Stata data file are not preserved since Categorical variables always use integer data types between -1 and n-1 where n is the number of categories. If the original values in the Stata data file are required, these can be imported by setting convert_categoricals=False, which will import original data (but not the variable labels). The original values can be matched to the imported categorical data since there is a simple mapping between the original Stata data values and the category codes of imported Categorical variables: missing values are assigned code -1, and the smallest original value is assigned 0, the second smallest is assigned 1 and so on until the largest original value is assigned the code n-1.

Stata supports partially labeled series. These series have value labels for some but not all data values. Importing a partially labeled series will produce a Categorical with string categories for the values that are labeled and numeric categories for values with no label.

The top-level function read_sas() can read (but not write) SAS XPORT (.xpt) and SAS7BDAT (.sas7bdat) format files.

SAS files only contain two value types: ASCII text and floating point values (usually 8 bytes but sometimes truncated). For xport files, there is no automatic type conversion to integers, dates, or categoricals. For SAS7BDAT files, the format codes may allow date variables to be automatically converted to dates. By default the whole file is read and returned as a DataFrame.

Specify a chunksize or use iterator=True to obtain reader objects (XportReader or SAS7BDATReader) for incrementally reading the file. The reader objects also have attributes that contain additional information about the file and its variables.

Read a SAS7BDAT file:

Obtain an iterator and read an XPORT file 100,000 lines at a time:

The specification for the xport file format is available from the SAS web site.

No official documentation is available for the SAS7BDAT format.

The top-level function read_spss() can read (but not write) SPSS SAV (.sav) and ZSAV (.zsav) format files.

SPSS files contain column names. By default the whole file is read, categorical columns are converted into pd.Categorical, and a DataFrame with all columns is returned.

Specify the usecols parameter to obtain a subset of columns. Specify convert_categoricals=False to avoid converting categorical columns into pd.Categorical.

Extract a subset of columns contained in usecols from an SPSS file and avoid converting categorical columns into pd.Categorical:

More information about the SAV and ZSAV file formats is available here.

pandas itself only supports IO with a limited set of file formats that map cleanly to its tabular data model. For reading and writing other file formats into and from pandas, we recommend these packages from the broader community.

xarray provides data structures inspired by the pandas DataFrame for working with multi-dimensional datasets, with a focus on the netCDF file format and easy conversion to and from pandas.

This is an informal comparison of various IO methods, using pandas 0.24.2. Timings are machine dependent and small differences should be ignored.

The following test functions will be used below to compare the performance of several IO methods:

When writing, the top three functions in terms of speed are test_feather_write, test_hdf_fixed_write and test_hdf_fixed_write_compress.

When reading, the top three functions in terms of speed are test_feather_read, test_pickle_read and test_hdf_fixed_read.

The files test.pkl.compress, test.parquet and test.feather took the least space on disk (in bytes).

**Examples:**

Example 1 (typescript):
```typescript
In [1]: import pandas as pd

In [2]: from io import StringIO

In [3]: data = "col1,col2,col3\na,b,1\na,b,2\nc,d,3"

In [4]: pd.read_csv(StringIO(data))
Out[4]: 
  col1 col2  col3
0    a    b     1
1    a    b     2
2    c    d     3

In [5]: pd.read_csv(StringIO(data), usecols=lambda x: x.upper() in ["COL1", "COL3"])
Out[5]: 
  col1  col3
0    a     1
1    a     2
2    c     3
```

Example 2 (typescript):
```typescript
In [6]: data = "col1,col2,col3\na,b,1\na,b,2\nc,d,3"

In [7]: pd.read_csv(StringIO(data))
Out[7]: 
  col1 col2  col3
0    a    b     1
1    a    b     2
2    c    d     3

In [8]: pd.read_csv(StringIO(data), skiprows=lambda x: x % 2 != 0)
Out[8]: 
  col1 col2  col3
0    a    b     2
```

Example 3 (typescript):
```typescript
In [9]: import numpy as np

In [10]: data = "a,b,c,d\n1,2,3,4\n5,6,7,8\n9,10,11"

In [11]: print(data)
a,b,c,d
1,2,3,4
5,6,7,8
9,10,11

In [12]: df = pd.read_csv(StringIO(data), dtype=object)

In [13]: df
Out[13]: 
   a   b   c    d
0  1   2   3    4
1  5   6   7    8
2  9  10  11  NaN

In [14]: df["a"][0]
Out[14]: '1'

In [15]: df = pd.read_csv(StringIO(data), dtype={"b": object, "c": np.float64, "d": "Int64"})

In [16]: df.dtypes
Out[16]: 
a      int64
b     object
c    float64
d      Int64
dtype: object
```

Example 4 (jsx):
```jsx
In [17]: data = "col_1\n1\n2\n'A'\n4.22"

In [18]: df = pd.read_csv(StringIO(data), converters={"col_1": str})

In [19]: df
Out[19]: 
  col_1
0     1
1     2
2   'A'
3  4.22

In [20]: df["col_1"].apply(type).value_counts()
Out[20]: 
col_1
<class 'str'>    4
Name: count, dtype: int64
```

---

## Frequently Asked Questions (FAQ)#

**URL:** https://pandas.pydata.org/docs/user_guide/gotchas.html

**Contents:**
- Frequently Asked Questions (FAQ)#
- DataFrame memory usage#
- Using if/truth statements with pandas#
  - Bitwise boolean#
  - Using the in operator#
- Mutating with User Defined Function (UDF) methods#
- Missing value representation for NumPy types#
  - np.nan as the NA representation for NumPy types#
  - NA type promotions for NumPy types#
  - Support for integer NA#

The memory usage of a DataFrame (including the index) is shown when calling the info(). A configuration option, display.memory_usage (see the list of options), specifies if the DataFrame memory usage will be displayed when invoking the info() method.

For example, the memory usage of the DataFrame below is shown when calling info():

The + symbol indicates that the true memory usage could be higher, because pandas does not count the memory used by values in columns with dtype=object.

Passing memory_usage='deep' will enable a more accurate memory usage report, accounting for the full usage of the contained objects. This is optional as it can be expensive to do this deeper introspection.

By default the display option is set to True but can be explicitly overridden by passing the memory_usage argument when invoking info().

The memory usage of each column can be found by calling the memory_usage() method. This returns a Series with an index represented by column names and memory usage of each column shown in bytes. For the DataFrame above, the memory usage of each column and the total memory usage can be found with the memory_usage() method:

By default the memory usage of the DataFrame index is shown in the returned Series, the memory usage of the index can be suppressed by passing the index=False argument:

The memory usage displayed by the info() method utilizes the memory_usage() method to determine the memory usage of a DataFrame while also formatting the output in human-readable units (base-2 representation; i.e. 1KB = 1024 bytes).

See also Categorical Memory Usage.

pandas follows the NumPy convention of raising an error when you try to convert something to a bool. This happens in an if-statement or when using the boolean operations: and, or, and not. It is not clear what the result of the following code should be:

Should it be True because it’s not zero-length, or False because there are False values? It is unclear, so instead, pandas raises a ValueError:

You need to explicitly choose what you want to do with the DataFrame, e.g. use any(), all() or empty(). Alternatively, you might want to compare if the pandas object is None:

Below is how to check if any of the values are True:

Bitwise boolean operators like == and != return a boolean Series which performs an element-wise comparison when compared to a scalar.

See boolean comparisons for more examples.

Using the Python in operator on a Series tests for membership in the index, not membership among the values.

If this behavior is surprising, keep in mind that using in on a Python dictionary tests keys, not values, and Series are dict-like. To test for membership in the values, use the method isin():

For DataFrame, likewise, in applies to the column axis, testing for membership in the list of column names.

This section applies to pandas methods that take a UDF. In particular, the methods DataFrame.apply(), DataFrame.aggregate(), DataFrame.transform(), and DataFrame.filter().

It is a general rule in programming that one should not mutate a container while it is being iterated over. Mutation will invalidate the iterator, causing unexpected behavior. Consider the example:

One probably would have expected that the result would be [1, 3, 5]. When using a pandas method that takes a UDF, internally pandas is often iterating over the DataFrame or other pandas object. Therefore, if the UDF mutates (changes) the DataFrame, unexpected behavior can arise.

Here is a similar example with DataFrame.apply():

To resolve this issue, one can make a copy so that the mutation does not apply to the container being iterated over.

For lack of NA (missing) support from the ground up in NumPy and Python in general, NA could have been represented with:

A masked array solution: an array of data and an array of boolean values indicating whether a value is there or is missing.

Using a special sentinel value, bit pattern, or set of sentinel values to denote NA across the dtypes.

The special value np.nan (Not-A-Number) was chosen as the NA value for NumPy types, and there are API functions like DataFrame.isna() and DataFrame.notna() which can be used across the dtypes to detect NA values. However, this choice has a downside of coercing missing integer data as float types as shown in Support for integer NA.

When introducing NAs into an existing Series or DataFrame via reindex() or some other means, boolean and integer types will be promoted to a different dtype in order to store the NAs. The promotions are summarized in this table:

Promotion dtype for storing NAs

In the absence of high performance NA support being built into NumPy from the ground up, the primary casualty is the ability to represent NAs in integer arrays. For example:

This trade-off is made largely for memory and performance reasons, and also so that the resulting Series continues to be “numeric”.

If you need to represent integers with possibly missing values, use one of the nullable-integer extension dtypes provided by pandas or pyarrow

See Nullable integer data type and PyArrow Functionality for more.

Many people have suggested that NumPy should simply emulate the NA support present in the more domain-specific statistical programming language R. Part of the reason is the NumPy type hierarchy.

The R language, by contrast, only has a handful of built-in data types: integer, numeric (floating-point), character, and boolean. NA types are implemented by reserving special bit patterns for each type to be used as the missing value. While doing this with the full NumPy type hierarchy would be possible, it would be a more substantial trade-off (especially for the 8- and 16-bit data types) and implementation undertaking.

However, R NA semantics are now available by using masked NumPy types such as Int64Dtype or PyArrow types (ArrowDtype).

For Series and DataFrame objects, var() normalizes by N-1 to produce unbiased estimates of the population variance, while NumPy’s numpy.var() normalizes by N, which measures the variance of the sample. Note that cov() normalizes by N-1 in both pandas and NumPy.

pandas is not 100% thread safe. The known issues relate to the copy() method. If you are doing a lot of copying of DataFrame objects shared among threads, we recommend holding locks inside the threads where the data copying occurs.

See this link for more information.

Occasionally you may have to deal with data that were created on a machine with a different byte order than the one on which you are running Python. A common symptom of this issue is an error like:

To deal with this issue you should convert the underlying NumPy array to the native system byte order before passing it to Series or DataFrame constructors using something similar to the following:

See the NumPy documentation on byte order for more details.

**Examples:**

Example 1 (yaml):
```yaml
In [1]: dtypes = [
   ...:     "int64",
   ...:     "float64",
   ...:     "datetime64[ns]",
   ...:     "timedelta64[ns]",
   ...:     "complex128",
   ...:     "object",
   ...:     "bool",
   ...: ]
   ...: 

In [2]: n = 5000

In [3]: data = {t: np.random.randint(100, size=n).astype(t) for t in dtypes}

In [4]: df = pd.DataFrame(data)

In [5]: df["categorical"] = df["object"].astype("category")

In [6]: df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5000 entries, 0 to 4999
Data columns (total 8 columns):
 #   Column           Non-Null Count  Dtype          
---  ------           --------------  -----          
 0   int64            5000 non-null   int64          
 1   float64          5000 non-null   float64        
 2   datetime64[ns]   5000 non-null   datetime64[ns] 
 3   timedelta64[ns]  5000 non-null   timedelta64[ns]
 4   complex128       5000 non-null   complex128     
 5   object           5000 non-null   object         
 6   bool             5000 non-null   bool           
 7   categorical      5000 non-null   category       
dtypes: bool(1), category(1), complex128(1), datetime64[ns](1), float64(1), int64(1), object(1), timedelta64[ns](1)
memory usage: 288.2+ KB
```

Example 2 (yaml):
```yaml
In [7]: df.info(memory_usage="deep")
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5000 entries, 0 to 4999
Data columns (total 8 columns):
 #   Column           Non-Null Count  Dtype          
---  ------           --------------  -----          
 0   int64            5000 non-null   int64          
 1   float64          5000 non-null   float64        
 2   datetime64[ns]   5000 non-null   datetime64[ns] 
 3   timedelta64[ns]  5000 non-null   timedelta64[ns]
 4   complex128       5000 non-null   complex128     
 5   object           5000 non-null   object         
 6   bool             5000 non-null   bool           
 7   categorical      5000 non-null   category       
dtypes: bool(1), category(1), complex128(1), datetime64[ns](1), float64(1), int64(1), object(1), timedelta64[ns](1)
memory usage: 424.7 KB
```

Example 3 (yaml):
```yaml
In [8]: df.memory_usage()
Out[8]: 
Index                128
int64              40000
float64            40000
datetime64[ns]     40000
timedelta64[ns]    40000
complex128         80000
object             40000
bool                5000
categorical         9968
dtype: int64

# total memory usage of dataframe
In [9]: df.memory_usage().sum()
Out[9]: 295096
```

Example 4 (yaml):
```yaml
In [10]: df.memory_usage(index=False)
Out[10]: 
int64              40000
float64            40000
datetime64[ns]     40000
timedelta64[ns]    40000
complex128         80000
object             40000
bool                5000
categorical         9968
dtype: int64
```

---

## 

**URL:** https://pandas.pydata.org/docs/_sources/user_guide/dsintro.rst.txt

---

## 

**URL:** https://pandas.pydata.org/docs/_sources/user_guide/advanced.rst.txt

---

## Migration guide for the new string data type (pandas 3.0)#

**URL:** https://pandas.pydata.org/docs/user_guide/migration-3-strings.html

**Contents:**
- Migration guide for the new string data type (pandas 3.0)#
- Background#
- Brief introduction to the new default string dtype#
- Overview of behavior differences and how to address them#
  - The dtype is no longer a numpy “object” dtype#
    - Checking the dtype#
    - Hardcoded use of object dtype#
  - The missing value sentinel is now always NaN#
  - “setitem” operations will now raise an error for non-string data#
  - Invalid unicode input#

The upcoming pandas 3.0 release introduces a new, default string data type. This will most likely cause some work when upgrading to pandas 3.0, and this page provides an overview of the issues you might run into and gives guidance on how to address them.

This new dtype is already available in the pandas 2.3 release, and you can enable it with:

This allows you to test your code before the final 3.0 release.

Historically, pandas has always used the NumPy object dtype as the default to store text data. This has two primary drawbacks. First, object dtype is not specific to strings: any Python object can be stored in an object-dtype array, not just strings, and seeing object as the dtype for a column with strings is confusing for users. Second, this is not always very efficient (both performance wise and for memory usage).

Since pandas 1.0, an opt-in string data type has been available, but this has not yet been made the default, and uses the pd.NA scalar to represent missing values.

Pandas 3.0 changes the default dtype for strings to a new string data type, a variant of the existing optional string data type but using NaN as the missing value indicator, to be consistent with the other default data types.

To improve performance, the new string data type will use the pyarrow package by default, if installed (and otherwise it uses object dtype under the hood as a fallback).

See PDEP-14: Dedicated string data type for pandas 3.0 for more background and details.

By default, pandas will infer this new string dtype instead of object dtype for string data (when creating pandas objects, such as in constructors or IO functions).

Being a default dtype means that the string dtype will be used in IO methods or constructors when the dtype is being inferred and the input is inferred to be string data:

It can also be specified explicitly using the "str" alias:

Similarly, functions like read_csv(), read_parquet(), and others will now use the new string dtype when reading string data.

In contrast to the current object dtype, the new string dtype will only store strings. This also means that it will raise an error if you try to store a non-string value in it (see below for more details).

Missing values with the new string dtype are always represented as NaN (np.nan), and the missing value behavior is similar to other default dtypes.

This new string dtype should otherwise behave the same as the existing object dtype users are used to. For example, all string-specific methods through the str accessor will work the same:

The new default string dtype is an instance of the pandas.StringDtype class. The dtype can be constructed as pd.StringDtype(na_value=np.nan), but for general usage we recommend to use the shorter "str" alias.

When inferring or reading string data, the data type of the resulting DataFrame column or Series will silently start being the new "str" dtype instead of the numpy "object" dtype, and this can have some impact on your code.

The new string dtype is a pandas data type (“extension dtype”), and no longer a numpy np.dtype instance. Therefore, passing the dtype of a string column to numpy functions will no longer work (e.g. passing it to a dtype= argument of a numpy function, or using np.issubdtype to check the dtype).

When checking the dtype, code might currently do something like:

to check for columns with string data (by checking for the dtype being "object"). This will no longer work in pandas 3+, since ser.dtype will now be "str" with the new default string dtype, and the above check will return False.

To check for columns with string data, you should instead use:

How to write compatible code

For code that should work on both pandas 2.x and 3.x, you can use the pandas.api.types.is_string_dtype() function:

This will return True for both the object dtype and the string dtypes.

If you have code where the dtype is hardcoded in constructors, like

this will keep using the object dtype. You will want to update this code to ensure you get the benefits of the new string dtype.

How to write compatible code?

First, in many cases it can be sufficient to remove the specific data type, and let pandas do the inference. But if you want to be specific, you can specify the "str" dtype:

This is actually compatible with pandas 2.x as well, since in pandas < 3, dtype="str" was essentially treated as an alias for object dtype.

While using dtype="str" in constructors is compatible with pandas 2.x, specifying it as the dtype in astype() runs into the issue of also stringifying missing values in pandas 2.x. See the section astype(str) preserving missing values for more details.

For selecting string columns with select_dtypes() in a pandas 2.x and 3.x compatible way, it is not possible to use "str". While this works for pandas 3.x, it raises an error in pandas 2.x. As an alternative, you can select both object (for pandas 2.x) and "string" (for pandas 3.x; which will also select the default str dtype and does not error on pandas 2.x):

When using object dtype, multiple possible missing value sentinels are supported, including None and np.nan. With the new default string dtype, the missing value sentinel is always NaN (np.nan):

Generally this should be no problem when relying on missing value behavior in pandas methods (for example, ser.isna() will give the same result as before). But when you relied on the exact value of None being present, that can impact your code.

How to write compatible code?

When checking for a missing value, instead of checking for the exact value of None or np.nan, you should use the pandas.isna() function. This is the most robust way to check for missing values, as it will work regardless of the dtype and the exact missing value sentinel:

One caveat: this function works both on scalars and on array-likes, and in the latter case it will return an array of bools. When using it in a Boolean context (for example, if pd.isna(..): ..) be sure to only pass a scalar to it.

With the new string dtype, any attempt to set a non-string value in a Series or DataFrame will raise an error:

If you relied on the flexible nature of object dtype being able to hold any Python object, but your initial data was inferred as strings, your code might be impacted by this change.

How to write compatible code?

You can update your code to ensure you only set string values in such columns, or otherwise you can explicitly ensure the column has object dtype first. This can be done by specifying the dtype explicitly in the constructor, or by using the astype() method:

This astype("object") call will be redundant when using pandas 2.x, but this code will work for all versions.

Python allows to have a built-in str object that represents invalid unicode data. And since the object dtype can hold any Python object, you can have a pandas Series with such invalid unicode data:

However, when using the string dtype using pyarrow under the hood, this can only store valid unicode data, and otherwise it will raise an error:

If you want to keep the previous behaviour, you can explicitly specify dtype=object to keep working with object dtype.

When you have byte data that you want to convert to strings using decode(), the decode() method now has a dtype parameter to be able to specify object dtype instead of the default of string dtype for this use case.

With object dtype, using .values on a Series will return the underlying NumPy array.

However with the new string dtype, the underlying ExtensionArray is returned instead.

If your code requires a NumPy array, you should use Series.to_numpy().

In general, you should always prefer Series.to_numpy() to get a NumPy array or Series.array() to get an ExtensionArray over using Series.values().

The stringifying of missing values is a long standing “bug” or misfeature, as discussed in pandas-dev/pandas#25353, but fixing it introduces a significant behaviour change.

With pandas < 3, when using astype(str) or astype("str"), the operation would convert every element to a string, including the missing values:

Note how NaN (np.nan) was converted to the string "nan". This was not the intended behavior, and it was inconsistent with how other dtypes handled missing values.

With pandas 3, this behavior has been fixed, and now astype("str") will cast to the new string dtype, which preserves the missing values:

If you want to preserve the old behaviour of converting every object to a string, you can use ser.map(str) instead. If you want do such conversion while preserving the missing values in a way that works with both pandas 2.x and 3.x, you can use ser.map(str, na_action="ignore") (for pandas 3.x only, you can do ser.astype("str")).

If you want to convert to object or string dtype for pandas 2.x and 3.x, respectively, without needing to stringify each individual element, you will have to use a conditional check on the pandas version. For example, to convert a categorical Series with string categories to its dense non-categorical version with object or string dtype:

In pandas < 3, calling the prod() method on a Series with string data would generally raise an error, except when the Series was empty or contained only a single string (potentially with missing values):

When the Series contains multiple strings, it will raise a TypeError. This behaviour stays the same in pandas 3 when using the flexible object dtype. But by virtue of using the new string dtype, this will generally consistently raise an error regardless of the number of strings:

**Examples:**

Example 1 (unknown):
```unknown
pd.options.future.infer_string = True
```

Example 2 (yaml):
```yaml
>>> pd.Series(["a", "b", None])
0      a
1      b
2    NaN
dtype: str
```

Example 3 (yaml):
```yaml
>>> pd.Series(["a", "b", None], dtype="str")
0      a
1      b
2    NaN
dtype: str
```

Example 4 (yaml):
```yaml
>>> ser = pd.Series(["a", "b", None], dtype="str")
>>> ser.str.upper()
0    A
1    B
2  NaN
dtype: str
```

---

## 

**URL:** https://pandas.pydata.org/docs/_sources/user_guide/index.rst.txt

---

## Group by: split-apply-combine#

**URL:** https://pandas.pydata.org/docs/user_guide/groupby.html

**Contents:**
- Group by: split-apply-combine#
- Splitting an object into groups#
  - GroupBy sorting#
    - GroupBy dropna#
  - GroupBy object attributes#
  - GroupBy with MultiIndex#
  - Grouping DataFrame with Index levels and columns#
  - DataFrame column selection in GroupBy#
- Iterating through groups#
- Selecting a group#

By “group by” we are referring to a process involving one or more of the following steps:

Splitting the data into groups based on some criteria.

Applying a function to each group independently.

Combining the results into a data structure.

Out of these, the split step is the most straightforward. In the apply step, we might wish to do one of the following:

Aggregation: compute a summary statistic (or statistics) for each group. Some examples:

Compute group sums or means.

Compute group sizes / counts.

Transformation: perform some group-specific computations and return a like-indexed object. Some examples:

Standardize data (zscore) within a group.

Filling NAs within groups with a value derived from each group.

Filtration: discard some groups, according to a group-wise computation that evaluates to True or False. Some examples:

Discard data that belong to groups with only a few members.

Filter out data based on the group sum or mean.

Many of these operations are defined on GroupBy objects. These operations are similar to those of the aggregating API, window API, and resample API.

It is possible that a given operation does not fall into one of these categories or is some combination of them. In such a case, it may be possible to compute the operation using GroupBy’s apply method. This method will examine the results of the apply step and try to sensibly combine them into a single result if it doesn’t fit into either of the above three categories.

An operation that is split into multiple steps using built-in GroupBy operations will be more efficient than using the apply method with a user-defined Python function.

The name GroupBy should be quite familiar to those who have used a SQL-based tool (or itertools), in which you can write code like:

We aim to make operations like this natural and easy to express using pandas. We’ll address each area of GroupBy functionality, then provide some non-trivial examples / use cases.

See the cookbook for some advanced strategies.

The abstract definition of grouping is to provide a mapping of labels to group names. To create a GroupBy object (more on what the GroupBy object is later), you may do the following:

The mapping can be specified many different ways:

A Python function, to be called on each of the index labels.

A list or NumPy array of the same length as the index.

A dict or Series, providing a label -> group name mapping.

For DataFrame objects, a string indicating either a column name or an index level name to be used to group.

A list of any of the above things.

Collectively we refer to the grouping objects as the keys. For example, consider the following DataFrame:

A string passed to groupby may refer to either a column or an index level. If a string matches both a column name and an index level name, a ValueError will be raised.

On a DataFrame, we obtain a GroupBy object by calling groupby(). This method returns a pandas.api.typing.DataFrameGroupBy instance. We could naturally group by either the A or B columns, or both:

df.groupby('A') is just syntactic sugar for df.groupby(df['A']).

If we also have a MultiIndex on columns A and B, we can group by all the columns except the one we specify:

The above GroupBy will split the DataFrame on its index (rows). To split by columns, first do a transpose:

pandas Index objects support duplicate values. If a non-unique index is used as the group key in a groupby operation, all values for the same index value will be considered to be in one group and thus the output of aggregation functions will only contain unique index values:

Note that no splitting occurs until it’s needed. Creating the GroupBy object only verifies that you’ve passed a valid mapping.

Many kinds of complicated data manipulations can be expressed in terms of GroupBy operations (though it can’t be guaranteed to be the most efficient implementation). You can get quite creative with the label mapping functions.

By default the group keys are sorted during the groupby operation. You may however pass sort=False for potential speedups. With sort=False the order among group-keys follows the order of appearance of the keys in the original dataframe:

Note that groupby will preserve the order in which observations are sorted within each group. For example, the groups created by groupby() below are in the order they appeared in the original DataFrame:

By default NA values are excluded from group keys during the groupby operation. However, in case you want to include NA values in group keys, you could pass dropna=False to achieve it.

The default setting of dropna argument is True which means NA are not included in group keys.

The groups attribute is a dictionary whose keys are the computed unique groups and corresponding values are the axis labels belonging to each group. In the above example we have:

Calling the standard Python len function on the GroupBy object returns the number of groups, which is the same as the length of the groups dictionary:

GroupBy will tab complete column names, GroupBy operations, and other attributes:

With hierarchically-indexed data, it’s quite natural to group by one of the levels of the hierarchy.

Let’s create a Series with a two-level MultiIndex.

We can then group by one of the levels in s.

If the MultiIndex has names specified, these can be passed instead of the level number:

Grouping with multiple levels is supported.

Index level names may be supplied as keys.

More on the sum function and aggregation later.

A DataFrame may be grouped by a combination of columns and index levels. You can specify both column and index names, or use a Grouper.

Let’s first create a DataFrame with a MultiIndex:

Then we group df by the second index level and the A column.

Index levels may also be specified by name.

Index level names may be specified as keys directly to groupby.

Once you have created the GroupBy object from a DataFrame, you might want to do something different for each of the columns. Thus, by using [] on the GroupBy object in a similar way as the one used to get a column from a DataFrame, you can do:

This is mainly syntactic sugar for the alternative, which is much more verbose:

Additionally, this method avoids recomputing the internal grouping information derived from the passed key.

You can also include the grouping columns if you want to operate on them.

With the GroupBy object in hand, iterating through the grouped data is very natural and functions similarly to itertools.groupby():

In the case of grouping by multiple keys, the group name will be a tuple:

See Iterating through groups.

A single group can be selected using DataFrameGroupBy.get_group():

Or for an object grouped on multiple columns:

An aggregation is a GroupBy operation that reduces the dimension of the grouping object. The result of an aggregation is, or at least is treated as, a scalar value for each column in a group. For example, producing the sum of each column in a group of values.

In the result, the keys of the groups appear in the index by default. They can be instead included in the columns by passing as_index=False.

Many common aggregations are built-in to GroupBy objects as methods. Of the methods listed below, those with a * do not have an efficient, GroupBy-specific, implementation.

Compute whether any of the values in the groups are truthy

Compute whether all of the values in the groups are truthy

Compute the number of non-NA values in the groups

Compute the covariance of the groups

Compute the first occurring value in each group

Compute the index of the maximum value in each group

Compute the index of the minimum value in each group

Compute the last occurring value in each group

Compute the maximum value in each group

Compute the mean of each group

Compute the median of each group

Compute the minimum value in each group

Compute the number of unique values in each group

Compute the product of the values in each group

Compute a given quantile of the values in each group

Compute the standard error of the mean of the values in each group

Compute the number of values in each group

Compute the skew of the values in each group

Compute the standard deviation of the values in each group

Compute the sum of the values in each group

Compute the variance of the values in each group

Another aggregation example is to compute the size of each group. This is included in GroupBy as the size method. It returns a Series whose index consists of the group names and the values are the sizes of each group.

While the DataFrameGroupBy.describe() method is not itself a reducer, it can be used to conveniently produce a collection of summary statistics about each of the groups.

Another aggregation example is to compute the number of unique values of each group. This is similar to the DataFrameGroupBy.value_counts() function, except that it only counts the number of unique values.

Aggregation functions will not return the groups that you are aggregating over as named columns when as_index=True, the default. The grouped columns will be the indices of the returned object.

Passing as_index=False will return the groups that you are aggregating over as named columns, regardless if they are named indices or columns in the inputs.

The aggregate() method can accept many different types of inputs. This section details using string aliases for various GroupBy methods; other inputs are detailed in the sections below.

Any reduction method that pandas implements can be passed as a string to aggregate(). Users are encouraged to use the shorthand, agg. It will operate as if the corresponding method was called.

The result of the aggregation will have the group names as the new index. In the case of multiple keys, the result is a MultiIndex by default. As mentioned above, this can be changed by using the as_index option:

Note that you could use the DataFrame.reset_index() DataFrame function to achieve the same result as the column names are stored in the resulting MultiIndex, although this will make an extra copy.

Users can also provide their own User-Defined Functions (UDFs) for custom aggregations.

When aggregating with a UDF, the UDF should not mutate the provided Series. See Mutating with User Defined Function (UDF) methods for more information.

Aggregating with a UDF is often less performant than using the pandas built-in methods on GroupBy. Consider breaking up a complex operation into a chain of operations that utilize the built-in methods.

The resulting dtype will reflect that of the aggregating function. If the results from different groups have different dtypes, then a common dtype will be determined in the same way as DataFrame construction.

On a grouped Series, you can pass a list or dict of functions to SeriesGroupBy.agg(), outputting a DataFrame:

On a grouped DataFrame, you can pass a list of functions to DataFrameGroupBy.agg() to aggregate each column, which produces an aggregated result with a hierarchical column index:

The resulting aggregations are named after the functions themselves. If you need to rename, then you can add in a chained operation for a Series like this:

For a grouped DataFrame, you can rename in a similar manner:

In general, the output column names should be unique, but pandas will allow you apply to the same function (or two functions with the same name) to the same column.

pandas also allows you to provide multiple lambdas. In this case, pandas will mangle the name of the (nameless) lambda functions, appending _<i> to each subsequent lambda.

To support column-specific aggregation with control over the output column names, pandas accepts the special syntax in DataFrameGroupBy.agg() and SeriesGroupBy.agg(), known as “named aggregation”, where

The keywords are the output column names

The values are tuples whose first element is the column to select and the second element is the aggregation to apply to that column. pandas provides the NamedAgg namedtuple with the fields ['column', 'aggfunc'] to make it clearer what the arguments are. As usual, the aggregation can be a callable or a string alias.

NamedAgg is just a namedtuple. Plain tuples are allowed as well.

If the column names you want are not valid Python keywords, construct a dictionary and unpack the keyword arguments

When using named aggregation, additional keyword arguments are not passed through to the aggregation functions; only pairs of (column, aggfunc) should be passed as **kwargs. If your aggregation functions require additional arguments, apply them partially with functools.partial().

Named aggregation is also valid for Series groupby aggregations. In this case there’s no column selection, so the values are just the functions.

By passing a dict to aggregate you can apply a different aggregation to the columns of a DataFrame:

The function names can also be strings. In order for a string to be valid it must be implemented on GroupBy:

A transformation is a GroupBy operation whose result is indexed the same as the one being grouped. Common examples include cumsum() and diff().

Unlike aggregations, the groupings that are used to split the original object are not included in the result.

Since transformations do not include the groupings that are used to split the result, the arguments as_index and sort in DataFrame.groupby() and Series.groupby() have no effect.

A common use of a transformation is to add the result back into the original DataFrame.

The following methods on GroupBy act as transformations.

Back fill NA values within each group

Compute the cumulative count within each group

Compute the cumulative max within each group

Compute the cumulative min within each group

Compute the cumulative product within each group

Compute the cumulative sum within each group

Compute the difference between adjacent values within each group

Forward fill NA values within each group

Compute the percent change between adjacent values within each group

Compute the rank of each value within each group

Shift values up or down within each group

In addition, passing any built-in aggregation method as a string to transform() (see the next section) will broadcast the result across the group, producing a transformed result. If the aggregation method has an efficient implementation, this will be performant as well.

Similar to the aggregation method, the transform() method can accept string aliases to the built-in transformation methods in the previous section. It can also accept string aliases to the built-in aggregation methods. When an aggregation method is provided, the result will be broadcast across the group.

In addition to string aliases, the transform() method can also accept User-Defined Functions (UDFs). The UDF must:

Return a result that is either the same size as the group chunk or broadcastable to the size of the group chunk (e.g., a scalar, grouped.transform(lambda x: x.iloc[-1])).

Operate column-by-column on the group chunk. The transform is applied to the first group chunk using chunk.apply.

Not perform in-place operations on the group chunk. Group chunks should be treated as immutable, and changes to a group chunk may produce unexpected results. See Mutating with User Defined Function (UDF) methods for more information.

(Optionally) operates on all columns of the entire group chunk at once. If this is supported, a fast path is used starting from the second chunk.

Transforming by supplying transform with a UDF is often less performant than using the built-in methods on GroupBy. Consider breaking up a complex operation into a chain of operations that utilize the built-in methods.

All of the examples in this section can be made more performant by calling built-in methods instead of using UDFs. See below for examples.

Changed in version 2.0.0: When using .transform on a grouped DataFrame and the transformation function returns a DataFrame, pandas now aligns the result’s index with the input’s index. You can call .to_numpy() within the transformation function to avoid alignment.

Similar to The aggregate() method, the resulting dtype will reflect that of the transformation function. If the results from different groups have different dtypes, then a common dtype will be determined in the same way as DataFrame construction.

Suppose we wish to standardize the data within each group:

We would expect the result to now have mean 0 and standard deviation 1 within each group (up to floating-point error), which we can easily check:

We can also visually compare the original and transformed data sets.

Transformation functions that have lower dimension outputs are broadcast to match the shape of the input array.

Another common data transform is to replace missing data with the group mean.

We can verify that the group means have not changed in the transformed data, and that the transformed data contains no NAs.

As mentioned in the note above, each of the examples in this section can be computed more efficiently using built-in methods. In the code below, the inefficient way using a UDF is commented out and the faster alternative appears below.

It is possible to use resample(), expanding() and rolling() as methods on groupbys.

The example below will apply the rolling() method on the samples of the column B, based on the groups of column A.

The expanding() method will accumulate a given operation (sum() in the example) for all the members of each particular group.

Suppose you want to use the resample() method to get a daily frequency in each group of your dataframe, and wish to complete the missing values with the ffill() method.

A filtration is a GroupBy operation that subsets the original grouping object. It may either filter out entire groups, part of groups, or both. Filtrations return a filtered version of the calling object, including the grouping columns when provided. In the following example, class is included in the result.

Unlike aggregations, filtrations do not add the group keys to the index of the result. Because of this, passing as_index=False or sort=True will not affect these methods.

Filtrations will respect subsetting the columns of the GroupBy object.

The following methods on GroupBy act as filtrations. All these methods have an efficient, GroupBy-specific, implementation.

Select the top row(s) of each group

Select the nth row(s) of each group

Select the bottom row(s) of each group

Users can also use transformations along with Boolean indexing to construct complex filtrations within groups. For example, suppose we are given groups of products and their volumes, and we wish to subset the data to only the largest products capturing no more than 90% of the total volume within each group.

Filtering by supplying filter with a User-Defined Function (UDF) is often less performant than using the built-in methods on GroupBy. Consider breaking up a complex operation into a chain of operations that utilize the built-in methods.

The filter method takes a User-Defined Function (UDF) that, when applied to an entire group, returns either True or False. The result of the filter method is then the subset of groups for which the UDF returned True.

Suppose we want to take only elements that belong to groups with a group sum greater than 2.

Another useful operation is filtering out elements that belong to groups with only a couple members.

Alternatively, instead of dropping the offending groups, we can return a like-indexed objects where the groups that do not pass the filter are filled with NaNs.

For DataFrames with multiple columns, filters should explicitly specify a column as the filter criterion.

Some operations on the grouped data might not fit into the aggregation, transformation, or filtration categories. For these, you can use the apply function.

apply has to try to infer from the result whether it should act as a reducer, transformer, or filter, depending on exactly what is passed to it. Thus the grouped column(s) may be included in the output or not. While it tries to intelligently guess how to behave, it can sometimes guess wrong.

All of the examples in this section can be more reliably, and more efficiently, computed using other pandas functionality.

The dimension of the returned result can also change:

apply on a Series can operate on a returned value from the applied function that is itself a series, and possibly upcast the result to a DataFrame:

Similar to The aggregate() method, the resulting dtype will reflect that of the apply function. If the results from different groups have different dtypes, then a common dtype will be determined in the same way as DataFrame construction.

To control whether the grouped column(s) are included in the indices, you can use the argument group_keys which defaults to True. Compare

Added in version 1.1.

If Numba is installed as an optional dependency, the transform and aggregate methods support engine='numba' and engine_kwargs arguments. See enhancing performance with Numba for general usage of the arguments and performance considerations.

The function signature must start with values, index exactly as the data belonging to each group will be passed into values, and the group index will be passed into index.

When using engine='numba', there will be no “fall back” behavior internally. The group data and group index will be passed as NumPy arrays to the JITed user defined function, and no alternative execution attempts will be tried.

Again consider the example DataFrame we’ve been looking at:

Suppose we wish to compute the standard deviation grouped by the A column. There is a slight problem, namely that we don’t care about the data in column B because it is not numeric. You can avoid non-numeric columns by specifying numeric_only=True:

Note that df.groupby('A').colname.std(). is more efficient than df.groupby('A').std().colname. So if the result of an aggregation function is only needed over one column (here colname), it may be filtered before applying the aggregation function.

When using a Categorical grouper (as a single grouper, or as part of multiple groupers), the observed keyword controls whether to return a cartesian product of all possible groupers values (observed=False) or only those that are observed groupers (observed=True).

Show only the observed values:

The returned dtype of the grouped will always include all of the categories that were grouped.

By NA, we are referring to any NA values, including NA, NaN, NaT, and None. If there are any NA values in the grouping key, by default these will be excluded. In other words, any “NA group” will be dropped. You can include NA groups by specifying dropna=False.

Categorical variables represented as instances of pandas’s Categorical class can be used as group keys. If so, the order of the levels will be preserved. When observed=False and sort=False, any unobserved categories will be at the end of the result in order.

You may need to specify a bit more data to properly group. You can use the pd.Grouper to provide this local control.

Groupby a specific column with the desired frequency. This is like resampling.

When freq is specified, the object returned by pd.Grouper will be an instance of pandas.api.typing.TimeGrouper. When there is a column and index with the same name, you can use key to group by the column and level to group by the index.

Just like for a DataFrame or Series you can call head and tail on a groupby:

This shows the first or last n rows from each group.

To select the nth item from each group, use DataFrameGroupBy.nth() or SeriesGroupBy.nth(). Arguments supplied can be any integer, lists of integers, slices, or lists of slices; see below for examples. When the nth element of a group does not exist an error is not raised; instead no corresponding rows are returned.

In general this operation acts as a filtration. In certain cases it will also return one row per group, making it also a reduction. However because in general it can return zero or multiple rows per group, pandas treats it as a filtration in all cases.

If the nth element of a group does not exist, then no corresponding row is included in the result. In particular, if the specified n is larger than any group, the result will be an empty DataFrame.

If you want to select the nth not-null item, use the dropna kwarg. For a DataFrame this should be either 'any' or 'all' just like you would pass to dropna:

You can also select multiple rows from each group by specifying multiple nth values as a list of ints.

You may also use slices or lists of slices.

To see the order in which each row appears within its group, use the cumcount method:

To see the ordering of the groups (as opposed to the order of rows within a group given by cumcount) you can use DataFrameGroupBy.ngroup().

Note that the numbers given to the groups match the order in which the groups would be seen when iterating over the groupby object, not the order they are first observed.

Groupby also works with some plotting methods. In this case, suppose we suspect that the values in column 1 are 3 times higher on average in group “B”.

We can easily visualize this with a boxplot:

The result of calling boxplot is a dictionary whose keys are the values of our grouping column g (“A” and “B”). The values of the resulting dictionary can be controlled by the return_type keyword of boxplot. See the visualization documentation for more.

For historical reasons, df.groupby("g").boxplot() is not equivalent to df.boxplot(by="g"). See here for an explanation.

Similar to the functionality provided by DataFrame and Series, functions that take GroupBy objects can be chained together using a pipe method to allow for a cleaner, more readable syntax. To read about .pipe in general terms, see here.

Combining .groupby and .pipe is often useful when you need to reuse GroupBy objects.

As an example, imagine having a DataFrame with columns for stores, products, revenue and quantity sold. We’d like to do a groupwise calculation of prices (i.e. revenue/quantity) per store and per product. We could do this in a multi-step operation, but expressing it in terms of piping can make the code more readable. First we set the data:

We now find the prices per store/product.

Piping can also be expressive when you want to deliver a grouped object to some arbitrary function, for example:

Here mean takes a GroupBy object and finds the mean of the Revenue and Quantity columns respectively for each Store-Product combination. The mean function can be any function that takes in a GroupBy object; the .pipe will pass the GroupBy object as a parameter into the function you specify.

By using DataFrameGroupBy.ngroup(), we can extract information about the groups in a way similar to factorize() (as described further in the reshaping API) but which applies naturally to multiple columns of mixed type and different sources. This can be useful as an intermediate categorical-like step in processing, when the relationships between the group rows are more important than their content, or as input to an algorithm which only accepts the integer encoding. (For more information about support in pandas for full categorical data, see the Categorical introduction and the API documentation.)

Resampling produces new hypothetical samples (resamples) from already existing observed data or from a model that generates data. These new samples are similar to the pre-existing samples.

In order for resample to work on indices that are non-datetimelike, the following procedure can be utilized.

In the following examples, df.index // 5 returns an integer array which is used to determine what gets selected for the groupby operation.

The example below shows how we can downsample by consolidation of samples into fewer ones. Here by using df.index // 5, we are aggregating the samples in bins. By applying std() function, we aggregate the information contained in many samples into a small subset of values which is their standard deviation thereby reducing the number of samples.

Group DataFrame columns, compute a set of metrics and return a named Series. The Series name is used as the name for the column index. This is especially useful in conjunction with reshaping operations such as stacking, in which the column index name will be used as the name of the inserted column:

**Examples:**

Example 1 (sql):
```sql
SELECT Column1, Column2, mean(Column3), sum(Column4)
FROM SomeTable
GROUP BY Column1, Column2
```

Example 2 (typescript):
```typescript
In [1]: speeds = pd.DataFrame(
   ...:     [
   ...:         ("bird", "Falconiformes", 389.0),
   ...:         ("bird", "Psittaciformes", 24.0),
   ...:         ("mammal", "Carnivora", 80.2),
   ...:         ("mammal", "Primates", np.nan),
   ...:         ("mammal", "Carnivora", 58),
   ...:     ],
   ...:     index=["falcon", "parrot", "lion", "monkey", "leopard"],
   ...:     columns=("class", "order", "max_speed"),
   ...: )
   ...: 

In [2]: speeds
Out[2]: 
          class           order  max_speed
falcon     bird   Falconiformes      389.0
parrot     bird  Psittaciformes       24.0
lion     mammal       Carnivora       80.2
monkey   mammal        Primates        NaN
leopard  mammal       Carnivora       58.0

In [3]: grouped = speeds.groupby("class")

In [4]: grouped = speeds.groupby(["class", "order"])
```

Example 3 (json):
```json
In [5]: df = pd.DataFrame(
   ...:     {
   ...:         "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
   ...:         "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
   ...:         "C": np.random.randn(8),
   ...:         "D": np.random.randn(8),
   ...:     }
   ...: )
   ...: 

In [6]: df
Out[6]: 
     A      B         C         D
0  foo    one  0.469112 -0.861849
1  bar    one -0.282863 -2.104569
2  foo    two -1.509059 -0.494929
3  bar  three -1.135632  1.071804
4  foo    two  1.212112  0.721555
5  bar    two -0.173215 -0.706771
6  foo    one  0.119209 -1.039575
7  foo  three -1.044236  0.271860
```

Example 4 (typescript):
```typescript
In [7]: grouped = df.groupby("A")

In [8]: grouped = df.groupby("B")

In [9]: grouped = df.groupby(["A", "B"])
```

---

## Sparse data structures#

**URL:** https://pandas.pydata.org/docs/user_guide/sparse.html

**Contents:**
- Sparse data structures#
- SparseArray#
- SparseDtype#
- Sparse accessor#
- Sparse calculation#
- Interaction with scipy.sparse#

pandas provides data structures for efficiently storing sparse data. These are not necessarily sparse in the typical “mostly 0”. Rather, you can view these objects as being “compressed” where any data matching a specific value (NaN / missing value, though any value can be chosen, including 0) is omitted. The compressed values are not actually stored in the array.

Notice the dtype, Sparse[float64, nan]. The nan means that elements in the array that are nan aren’t actually stored, only the non-nan elements are. Those non-nan elements have a float64 dtype.

The sparse objects exist for memory efficiency reasons. Suppose you had a large, mostly NA DataFrame:

As you can see, the density (% of values that have not been “compressed”) is extremely low. This sparse object takes up much less memory on disk (pickled) and in the Python interpreter.

Functionally, their behavior should be nearly identical to their dense counterparts.

arrays.SparseArray is a ExtensionArray for storing an array of sparse values (see dtypes for more on extension arrays). It is a 1-dimensional ndarray-like object storing only values distinct from the fill_value:

A sparse array can be converted to a regular (dense) ndarray with numpy.asarray()

The SparseArray.dtype property stores two pieces of information

The dtype of the non-sparse values

The scalar fill value

A SparseDtype may be constructed by passing only a dtype

in which case a default fill value will be used (for NumPy dtypes this is often the “missing” value for that dtype). To override this default an explicit fill value may be passed instead

Finally, the string alias 'Sparse[dtype]' may be used to specify a sparse dtype in many places

pandas provides a .sparse accessor, similar to .str for string data, .cat for categorical data, and .dt for datetime-like data. This namespace provides attributes and methods that are specific to sparse data.

This accessor is available only on data with SparseDtype, and on the Series class itself for creating a Series with sparse data from a scipy COO matrix with.

A .sparse accessor has been added for DataFrame as well. See Sparse accessor for more.

You can apply NumPy ufuncs to arrays.SparseArray and get a arrays.SparseArray as a result.

The ufunc is also applied to fill_value. This is needed to get the correct dense result.

To convert data from sparse to dense, use the .sparse accessors

From dense to sparse, use DataFrame.astype() with a SparseDtype.

Use DataFrame.sparse.from_spmatrix() to create a DataFrame with sparse values from a sparse matrix.

All sparse formats are supported, but matrices that are not in COOrdinate format will be converted, copying data as needed. To convert back to sparse SciPy matrix in COO format, you can use the DataFrame.sparse.to_coo() method:

Series.sparse.to_coo() is implemented for transforming a Series with sparse values indexed by a MultiIndex to a scipy.sparse.coo_matrix.

The method requires a MultiIndex with two or more levels.

In the example below, we transform the Series to a sparse representation of a 2-d array by specifying that the first and second MultiIndex levels define labels for the rows and the third and fourth levels define labels for the columns. We also specify that the column and row labels should be sorted in the final sparse representation.

Specifying different row and column labels (and not sorting them) yields a different sparse matrix:

A convenience method Series.sparse.from_coo() is implemented for creating a Series with sparse values from a scipy.sparse.coo_matrix.

The default behaviour (with dense_index=False) simply returns a Series containing only the non-null entries.

Specifying dense_index=True will result in an index that is the Cartesian product of the row and columns coordinates of the matrix. Note that this will consume a significant amount of memory (relative to dense_index=False) if the sparse matrix is large (and sparse) enough.

**Examples:**

Example 1 (typescript):
```typescript
In [1]: arr = np.random.randn(10)

In [2]: arr[2:-2] = np.nan

In [3]: ts = pd.Series(pd.arrays.SparseArray(arr))

In [4]: ts
Out[4]: 
0    0.469112
1   -0.282863
2         NaN
3         NaN
4         NaN
5         NaN
6         NaN
7         NaN
8   -0.861849
9   -2.104569
dtype: Sparse[float64, nan]
```

Example 2 (typescript):
```typescript
In [5]: df = pd.DataFrame(np.random.randn(10000, 4))

In [6]: df.iloc[:9998] = np.nan

In [7]: sdf = df.astype(pd.SparseDtype("float", np.nan))

In [8]: sdf.head()
Out[8]: 
     0    1    2    3
0  NaN  NaN  NaN  NaN
1  NaN  NaN  NaN  NaN
2  NaN  NaN  NaN  NaN
3  NaN  NaN  NaN  NaN
4  NaN  NaN  NaN  NaN

In [9]: sdf.dtypes
Out[9]: 
0    Sparse[float64, nan]
1    Sparse[float64, nan]
2    Sparse[float64, nan]
3    Sparse[float64, nan]
dtype: object

In [10]: sdf.sparse.density
Out[10]: 0.0002
```

Example 3 (json):
```json
In [11]: 'dense : {:0.2f} bytes'.format(df.memory_usage().sum() / 1e3)
Out[11]: 'dense : 320.13 bytes'

In [12]: 'sparse: {:0.2f} bytes'.format(sdf.memory_usage().sum() / 1e3)
Out[12]: 'sparse: 0.22 bytes'
```

Example 4 (json):
```json
In [13]: arr = np.random.randn(10)

In [14]: arr[2:5] = np.nan

In [15]: arr[7:8] = np.nan

In [16]: sparr = pd.arrays.SparseArray(arr)

In [17]: sparr
Out[17]: 
[-1.9556635297215477, -1.6588664275960427, nan, nan, nan, 1.1589328886422277, 0.14529711373305043, nan, 0.6060271905134522, 1.3342113401317768]
Fill: nan
IntIndex
Indices: array([0, 1, 5, 6, 8, 9], dtype=int32)
```

---

## Nullable integer data type#

**URL:** https://pandas.pydata.org/docs/user_guide/integer_na.html

**Contents:**
- Nullable integer data type#
- Construction#
- Operations#
- Scalar NA Value#

IntegerArray is currently experimental. Its API or implementation may change without warning. Uses pandas.NA as the missing value.

In Working with missing data, we saw that pandas primarily uses NaN to represent missing data. Because NaN is a float, this forces an array of integers with any missing values to become floating point. In some cases, this may not matter much. But if your integer column is, say, an identifier, casting to float can be problematic. Some integers cannot even be represented as floating point numbers.

pandas can represent integer data with possibly missing values using arrays.IntegerArray. This is an extension type implemented within pandas.

Or the string alias "Int64" (note the capital "I") to differentiate from NumPy’s 'int64' dtype:

All NA-like values are replaced with pandas.NA.

This array can be stored in a DataFrame or Series like any NumPy array.

You can also pass the list-like object to the Series constructor with the dtype.

Currently pandas.array() and pandas.Series() use different rules for dtype inference. pandas.array() will infer a nullable-integer dtype

For backwards-compatibility, Series infers these as either integer or float dtype.

We recommend explicitly providing the dtype to avoid confusion.

In the future, we may provide an option for Series to infer a nullable-integer dtype.

Operations involving an integer array will behave similar to NumPy arrays. Missing values will be propagated, and the data will be coerced to another dtype if needed.

These dtypes can operate as part of a DataFrame.

These dtypes can be merged, reshaped & casted.

Reduction and groupby operations such as sum() work as well.

arrays.IntegerArray uses pandas.NA as its scalar missing value. Slicing a single element that’s missing will return pandas.NA

**Examples:**

Example 1 (typescript):
```typescript
In [1]: arr = pd.array([1, 2, None], dtype=pd.Int64Dtype())

In [2]: arr
Out[2]: 
<IntegerArray>
[1, 2, <NA>]
Length: 3, dtype: Int64
```

Example 2 (json):
```json
In [3]: pd.array([1, 2, np.nan], dtype="Int64")
Out[3]: 
<IntegerArray>
[1, 2, <NA>]
Length: 3, dtype: Int64
```

Example 3 (json):
```json
In [4]: pd.array([1, 2, np.nan, None, pd.NA], dtype="Int64")
Out[4]: 
<IntegerArray>
[1, 2, <NA>, <NA>, <NA>]
Length: 5, dtype: Int64
```

Example 4 (yaml):
```yaml
In [5]: pd.Series(arr)
Out[5]: 
0       1
1       2
2    <NA>
dtype: Int64
```

---

## Intro to data structures#

**URL:** https://pandas.pydata.org/docs/user_guide/dsintro.html

**Contents:**
- Intro to data structures#
- Series#
  - Series is ndarray-like#
  - Series is dict-like#
  - Vectorized operations and label alignment with Series#
  - Name attribute#
- DataFrame#
  - From dict of Series or dicts#
  - From dict of ndarrays / lists#
  - From structured or record array#

We’ll start with a quick, non-comprehensive overview of the fundamental data structures in pandas to get you started. The fundamental behavior about data types, indexing, axis labeling, and alignment apply across all of the objects. To get started, import NumPy and load pandas into your namespace:

Fundamentally, data alignment is intrinsic. The link between labels and data will not be broken unless done so explicitly by you.

We’ll give a brief intro to the data structures, then consider all of the broad categories of functionality and methods in separate sections.

Series is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.). The axis labels are collectively referred to as the index. The basic method to create a Series is to call:

Here, data can be many different things:

a scalar value (like 5)

The passed index is a list of axis labels. Thus, this separates into a few cases depending on what data is:

If data is an ndarray, index must be the same length as data. If no index is passed, one will be created having values [0, ..., len(data) - 1].

pandas supports non-unique index values. If an operation that does not support duplicate index values is attempted, an exception will be raised at that time.

Series can be instantiated from dicts:

If an index is passed, the values in data corresponding to the labels in the index will be pulled out.

NaN (not a number) is the standard missing data marker used in pandas.

If data is a scalar value, an index must be provided. The value will be repeated to match the length of index.

Series acts very similarly to a ndarray and is a valid argument to most NumPy functions. However, operations such as slicing will also slice the index.

We will address array-based indexing like s.iloc[[4, 3, 1]] in section on indexing.

Like a NumPy array, a pandas Series has a single dtype.

This is often a NumPy dtype. However, pandas and 3rd-party libraries extend NumPy’s type system in a few places, in which case the dtype would be an ExtensionDtype. Some examples within pandas are Categorical data and Nullable integer data type. See dtypes for more.

If you need the actual array backing a Series, use Series.array.

Accessing the array can be useful when you need to do some operation without the index (to disable automatic alignment, for example).

Series.array will always be an ExtensionArray. Briefly, an ExtensionArray is a thin wrapper around one or more concrete arrays like a numpy.ndarray. pandas knows how to take an ExtensionArray and store it in a Series or a column of a DataFrame. See dtypes for more.

While Series is ndarray-like, if you need an actual ndarray, then use Series.to_numpy().

Even if the Series is backed by a ExtensionArray, Series.to_numpy() will return a NumPy ndarray.

A Series is also like a fixed-size dict in that you can get and set values by index label:

If a label is not contained in the index, an exception is raised:

Using the Series.get() method, a missing label will return None or specified default:

These labels can also be accessed by attribute.

When working with raw NumPy arrays, looping through value-by-value is usually not necessary. The same is true when working with Series in pandas. Series can also be passed into most NumPy methods expecting an ndarray.

A key difference between Series and ndarray is that operations between Series automatically align the data based on label. Thus, you can write computations without giving consideration to whether the Series involved have the same labels.

The result of an operation between unaligned Series will have the union of the indexes involved. If a label is not found in one Series or the other, the result will be marked as missing NaN. Being able to write code without doing any explicit data alignment grants immense freedom and flexibility in interactive data analysis and research. The integrated data alignment features of the pandas data structures set pandas apart from the majority of related tools for working with labeled data.

In general, we chose to make the default result of operations between differently indexed objects yield the union of the indexes in order to avoid loss of information. Having an index label, though the data is missing, is typically important information as part of a computation. You of course have the option of dropping labels with missing data via the dropna function.

Series also has a name attribute:

The Series name can be assigned automatically in many cases, in particular, when selecting a single column from a DataFrame, the name will be assigned the column label.

You can rename a Series with the pandas.Series.rename() method.

Note that s and s2 refer to different objects.

DataFrame is a 2-dimensional labeled data structure with columns of potentially different types. You can think of it like a spreadsheet or SQL table, or a dict of Series objects. It is generally the most commonly used pandas object. Like Series, DataFrame accepts many different kinds of input:

Dict of 1D ndarrays, lists, dicts, or Series

Structured or record ndarray

Along with the data, you can optionally pass index (row labels) and columns (column labels) arguments. If you pass an index and / or columns, you are guaranteeing the index and / or columns of the resulting DataFrame. Thus, a dict of Series plus a specific index will discard all data not matching up to the passed index.

If axis labels are not passed, they will be constructed from the input data based on common sense rules.

The resulting index will be the union of the indexes of the various Series. If there are any nested dicts, these will first be converted to Series. If no columns are passed, the columns will be the ordered list of dict keys.

The row and column labels can be accessed respectively by accessing the index and columns attributes:

When a particular set of columns is passed along with a dict of data, the passed columns override the keys in the dict.

All ndarrays must share the same length. If an index is passed, it must also be the same length as the arrays. If no index is passed, the result will be range(n), where n is the array length.

This case is handled identically to a dict of arrays.

DataFrame is not intended to work exactly like a 2-dimensional NumPy ndarray.

You can automatically create a MultiIndexed frame by passing a tuples dictionary.

The result will be a DataFrame with the same index as the input Series, and with one column whose name is the original name of the Series (only if no other column name provided).

The field names of the first namedtuple in the list determine the columns of the DataFrame. The remaining namedtuples (or tuples) are simply unpacked and their values are fed into the rows of the DataFrame. If any of those tuples is shorter than the first namedtuple then the later columns in the corresponding row are marked as missing values. If any are longer than the first namedtuple, a ValueError is raised.

Data Classes as introduced in PEP557, can be passed into the DataFrame constructor. Passing a list of dataclasses is equivalent to passing a list of dictionaries.

Please be aware, that all values in the list should be dataclasses, mixing types in the list would result in a TypeError.

To construct a DataFrame with missing data, we use np.nan to represent missing values. Alternatively, you may pass a numpy.MaskedArray as the data argument to the DataFrame constructor, and its masked entries will be considered missing. See Missing data for more.

DataFrame.from_dict() takes a dict of dicts or a dict of array-like sequences and returns a DataFrame. It operates like the DataFrame constructor except for the orient parameter which is 'columns' by default, but which can be set to 'index' in order to use the dict keys as row labels.

If you pass orient='index', the keys will be the row labels. In this case, you can also pass the desired column names:

DataFrame.from_records

DataFrame.from_records() takes a list of tuples or an ndarray with structured dtype. It works analogously to the normal DataFrame constructor, except that the resulting DataFrame index may be a specific field of the structured dtype.

You can treat a DataFrame semantically like a dict of like-indexed Series objects. Getting, setting, and deleting columns works with the same syntax as the analogous dict operations:

Columns can be deleted or popped like with a dict:

When inserting a scalar value, it will naturally be propagated to fill the column:

When inserting a Series that does not have the same index as the DataFrame, it will be conformed to the DataFrame’s index:

You can insert raw ndarrays but their length must match the length of the DataFrame’s index.

By default, columns get inserted at the end. DataFrame.insert() inserts at a particular location in the columns:

Inspired by dplyr’s mutate verb, DataFrame has an assign() method that allows you to easily create new columns that are potentially derived from existing columns.

In the example above, we inserted a precomputed value. We can also pass in a function of one argument to be evaluated on the DataFrame being assigned to.

assign() always returns a copy of the data, leaving the original DataFrame untouched.

Passing a callable, as opposed to an actual value to be inserted, is useful when you don’t have a reference to the DataFrame at hand. This is common when using assign() in a chain of operations. For example, we can limit the DataFrame to just those observations with a Sepal Length greater than 5, calculate the ratio, and plot:

Since a function is passed in, the function is computed on the DataFrame being assigned to. Importantly, this is the DataFrame that’s been filtered to those rows with sepal length greater than 5. The filtering happens first, and then the ratio calculations. This is an example where we didn’t have a reference to the filtered DataFrame available.

The function signature for assign() is simply **kwargs. The keys are the column names for the new fields, and the values are either a value to be inserted (for example, a Series or NumPy array), or a function of one argument to be called on the DataFrame. A copy of the original DataFrame is returned, with the new values inserted.

The order of **kwargs is preserved. This allows for dependent assignment, where an expression later in **kwargs can refer to a column created earlier in the same assign().

In the second expression, x['C'] will refer to the newly created column, that’s equal to dfa['A'] + dfa['B'].

The basics of indexing are as follows:

Select row by integer location

Select rows by boolean vector

Row selection, for example, returns a Series whose index is the columns of the DataFrame:

For a more exhaustive treatment of sophisticated label-based indexing and slicing, see the section on indexing. We will address the fundamentals of reindexing / conforming to new sets of labels in the section on reindexing.

Data alignment between DataFrame objects automatically align on both the columns and the index (row labels). Again, the resulting object will have the union of the column and row labels.

When doing an operation between DataFrame and Series, the default behavior is to align the Series index on the DataFrame columns, thus broadcasting row-wise. For example:

For explicit control over the matching and broadcasting behavior, see the section on flexible binary operations.

Arithmetic operations with scalars operate element-wise:

Boolean operators operate element-wise as well:

To transpose, access the T attribute or DataFrame.transpose(), similar to an ndarray:

Most NumPy functions can be called directly on Series and DataFrame.

DataFrame is not intended to be a drop-in replacement for ndarray as its indexing semantics and data model are quite different in places from an n-dimensional array.

Series implements __array_ufunc__, which allows it to work with NumPy’s universal functions.

The ufunc is applied to the underlying array in a Series.

When multiple Series are passed to a ufunc, they are aligned before performing the operation.

Like other parts of the library, pandas will automatically align labeled inputs as part of a ufunc with multiple inputs. For example, using numpy.remainder() on two Series with differently ordered labels will align before the operation.

As usual, the union of the two indices is taken, and non-overlapping values are filled with missing values.

When a binary ufunc is applied to a Series and Index, the Series implementation takes precedence and a Series is returned.

NumPy ufuncs are safe to apply to Series backed by non-ndarray arrays, for example arrays.SparseArray (see Sparse calculation). If possible, the ufunc is applied without converting the underlying data to an ndarray.

A very large DataFrame will be truncated to display them in the console. You can also get a summary using info(). (The baseball dataset is from the plyr R package):

However, using DataFrame.to_string() will return a string representation of the DataFrame in tabular form, though it won’t always fit the console width:

Wide DataFrames will be printed across multiple rows by default:

You can change how much to print on a single row by setting the display.width option:

You can adjust the max width of the individual columns by setting display.max_colwidth

You can also disable this feature via the expand_frame_repr option. This will print the table in one block.

If a DataFrame column label is a valid Python variable name, the column can be accessed like an attribute:

The columns are also connected to the IPython completion mechanism so they can be tab-completed:

**Examples:**

Example 1 (typescript):
```typescript
In [1]: import numpy as np

In [2]: import pandas as pd
```

Example 2 (unknown):
```unknown
s = pd.Series(data, index=index)
```

Example 3 (typescript):
```typescript
In [3]: s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])

In [4]: s
Out[4]: 
a    0.469112
b   -0.282863
c   -1.509059
d   -1.135632
e    1.212112
dtype: float64

In [5]: s.index
Out[5]: Index(['a', 'b', 'c', 'd', 'e'], dtype='object')

In [6]: pd.Series(np.random.randn(5))
Out[6]: 
0   -0.173215
1    0.119209
2   -1.044236
3   -0.861849
4   -2.104569
dtype: float64
```

Example 4 (json):
```json
In [7]: d = {"b": 1, "a": 0, "c": 2}

In [8]: pd.Series(d)
Out[8]: 
b    1
a    0
c    2
dtype: int64
```

---

## Essential basic functionality#

**URL:** https://pandas.pydata.org/docs/user_guide/basics.html

**Contents:**
- Essential basic functionality#
- Head and tail#
- Attributes and underlying data#
- Accelerated operations#
- Flexible binary operations#
  - Matching / broadcasting behavior#
  - Missing data / operations with fill values#
  - Flexible comparisons#
  - Boolean reductions#
  - Comparing if objects are equivalent#

Here we discuss a lot of the essential functionality common to the pandas data structures. To begin, let’s create some example objects like we did in the 10 minutes to pandas section:

To view a small sample of a Series or DataFrame object, use the head() and tail() methods. The default number of elements to display is five, but you may pass a custom number.

pandas objects have a number of attributes enabling you to access the metadata

shape: gives the axis dimensions of the object, consistent with ndarray

Series: index (only axis)

DataFrame: index (rows) and columns

Note, these attributes can be safely assigned to!

pandas objects (Index, Series, DataFrame) can be thought of as containers for arrays, which hold the actual data and do the actual computation. For many types, the underlying array is a numpy.ndarray. However, pandas and 3rd party libraries may extend NumPy’s type system to add support for custom arrays (see dtypes).

To get the actual data inside a Index or Series, use the .array property

array will always be an ExtensionArray. The exact details of what an ExtensionArray is and why pandas uses them are a bit beyond the scope of this introduction. See dtypes for more.

If you know you need a NumPy array, use to_numpy() or numpy.asarray().

When the Series or Index is backed by an ExtensionArray, to_numpy() may involve copying data and coercing values. See dtypes for more.

to_numpy() gives some control over the dtype of the resulting numpy.ndarray. For example, consider datetimes with timezones. NumPy doesn’t have a dtype to represent timezone-aware datetimes, so there are two possibly useful representations:

An object-dtype numpy.ndarray with Timestamp objects, each with the correct tz

A datetime64[ns] -dtype numpy.ndarray, where the values have been converted to UTC and the timezone discarded

Timezones may be preserved with dtype=object

Or thrown away with dtype='datetime64[ns]'

Getting the “raw data” inside a DataFrame is possibly a bit more complex. When your DataFrame only has a single data type for all the columns, DataFrame.to_numpy() will return the underlying data:

If a DataFrame contains homogeneously-typed data, the ndarray can actually be modified in-place, and the changes will be reflected in the data structure. For heterogeneous data (e.g. some of the DataFrame’s columns are not all the same dtype), this will not be the case. The values attribute itself, unlike the axis labels, cannot be assigned to.

When working with heterogeneous data, the dtype of the resulting ndarray will be chosen to accommodate all of the data involved. For example, if strings are involved, the result will be of object dtype. If there are only floats and integers, the resulting array will be of float dtype.

In the past, pandas recommended Series.values or DataFrame.values for extracting the data from a Series or DataFrame. You’ll still find references to these in old code bases and online. Going forward, we recommend avoiding .values and using .array or .to_numpy(). .values has the following drawbacks:

When your Series contains an extension type, it’s unclear whether Series.values returns a NumPy array or the extension array. Series.array will always return an ExtensionArray, and will never copy data. Series.to_numpy() will always return a NumPy array, potentially at the cost of copying / coercing values.

When your DataFrame contains a mixture of data types, DataFrame.values may involve copying data and coercing values to a common dtype, a relatively expensive operation. DataFrame.to_numpy(), being a method, makes it clearer that the returned NumPy array may not be a view on the same data in the DataFrame.

pandas has support for accelerating certain types of binary numerical and boolean operations using the numexpr library and the bottleneck libraries.

These libraries are especially useful when dealing with large data sets, and provide large speedups. numexpr uses smart chunking, caching, and multiple cores. bottleneck is a set of specialized cython routines that are especially fast when dealing with arrays that have nans.

Here is a sample (using 100 column x 100,000 row DataFrames):

You are highly encouraged to install both libraries. See the section Recommended Dependencies for more installation info.

These are both enabled to be used by default, you can control this by setting the options:

With binary operations between pandas data structures, there are two key points of interest:

Broadcasting behavior between higher- (e.g. DataFrame) and lower-dimensional (e.g. Series) objects.

Missing data in computations.

We will demonstrate how to manage these issues independently, though they can be handled simultaneously.

DataFrame has the methods add(), sub(), mul(), div() and related functions radd(), rsub(), … for carrying out binary operations. For broadcasting behavior, Series input is of primary interest. Using these functions, you can use to either match on the index or columns via the axis keyword:

Furthermore you can align a level of a MultiIndexed DataFrame with a Series.

Series and Index also support the divmod() builtin. This function takes the floor division and modulo operation at the same time returning a two-tuple of the same type as the left hand side. For example:

We can also do elementwise divmod():

In Series and DataFrame, the arithmetic functions have the option of inputting a fill_value, namely a value to substitute when at most one of the values at a location are missing. For example, when adding two DataFrame objects, you may wish to treat NaN as 0 unless both DataFrames are missing that value, in which case the result will be NaN (you can later replace NaN with some other value using fillna if you wish).

Series and DataFrame have the binary comparison methods eq, ne, lt, gt, le, and ge whose behavior is analogous to the binary arithmetic operations described above:

These operations produce a pandas object of the same type as the left-hand-side input that is of dtype bool. These boolean objects can be used in indexing operations, see the section on Boolean indexing.

You can apply the reductions: empty, any(), all(), and bool() to provide a way to summarize a boolean result.

You can reduce to a final boolean value.

You can test if a pandas object is empty, via the empty property.

Asserting the truthiness of a pandas object will raise an error, as the testing of the emptiness or values is ambiguous.

See gotchas for a more detailed discussion.

Often you may find that there is more than one way to compute the same result. As a simple example, consider df + df and df * 2. To test that these two computations produce the same result, given the tools shown above, you might imagine using (df + df == df * 2).all(). But in fact, this expression is False:

Notice that the boolean DataFrame df + df == df * 2 contains some False values! This is because NaNs do not compare as equals:

So, NDFrames (such as Series and DataFrames) have an equals() method for testing equality, with NaNs in corresponding locations treated as equal.

Note that the Series or DataFrame index needs to be in the same order for equality to be True:

You can conveniently perform element-wise comparisons when comparing a pandas data structure with a scalar value:

pandas also handles element-wise comparisons between different array-like objects of the same length:

Trying to compare Index or Series objects of different lengths will raise a ValueError:

A problem occasionally arising is the combination of two similar data sets where values in one are preferred over the other. An example would be two data series representing a particular economic indicator where one is considered to be of “higher quality”. However, the lower quality series might extend further back in history or have more complete data coverage. As such, we would like to combine two DataFrame objects where missing values in one DataFrame are conditionally filled with like-labeled values from the other DataFrame. The function implementing this operation is combine_first(), which we illustrate:

The combine_first() method above calls the more general DataFrame.combine(). This method takes another DataFrame and a combiner function, aligns the input DataFrame and then passes the combiner function pairs of Series (i.e., columns whose names are the same).

So, for instance, to reproduce combine_first() as above:

There exists a large number of methods for computing descriptive statistics and other related operations on Series, DataFrame. Most of these are aggregations (hence producing a lower-dimensional result) like sum(), mean(), and quantile(), but some of them, like cumsum() and cumprod(), produce an object of the same size. Generally speaking, these methods take an axis argument, just like ndarray.{sum, std, …}, but the axis can be specified by name or integer:

Series: no axis argument needed

DataFrame: “index” (axis=0, default), “columns” (axis=1)

All such methods have a skipna option signaling whether to exclude missing data (True by default):

Combined with the broadcasting / arithmetic behavior, one can describe various statistical procedures, like standardization (rendering data zero mean and standard deviation of 1), very concisely:

Note that methods like cumsum() and cumprod() preserve the location of NaN values. This is somewhat different from expanding() and rolling() since NaN behavior is furthermore dictated by a min_periods parameter.

Here is a quick reference summary table of common functions. Each also takes an optional level parameter which applies only if the object has a hierarchical index.

Number of non-NA observations

Arithmetic median of values

Bessel-corrected sample standard deviation

Standard error of the mean

Sample skewness (3rd moment)

Sample kurtosis (4th moment)

Sample quantile (value at %)

Note that by chance some NumPy methods, like mean, std, and sum, will exclude NAs on Series input by default:

Series.nunique() will return the number of unique non-NA values in a Series:

There is a convenient describe() function which computes a variety of summary statistics about a Series or the columns of a DataFrame (excluding NAs of course):

You can select specific percentiles to include in the output:

By default, the median is always included.

For a non-numerical Series object, describe() will give a simple summary of the number of unique values and most frequently occurring values:

Note that on a mixed-type DataFrame object, describe() will restrict the summary to include only numerical columns or, if none are, only categorical columns:

This behavior can be controlled by providing a list of types as include/exclude arguments. The special value all can also be used:

That feature relies on select_dtypes. Refer to there for details about accepted inputs.

The idxmin() and idxmax() functions on Series and DataFrame compute the index labels with the minimum and maximum corresponding values:

When there are multiple rows (or columns) matching the minimum or maximum value, idxmin() and idxmax() return the first matching index:

idxmin and idxmax are called argmin and argmax in NumPy.

The value_counts() Series method computes a histogram of a 1D array of values. It can also be used as a function on regular arrays:

The value_counts() method can be used to count combinations across multiple columns. By default all columns are used but a subset can be selected using the subset argument.

Similarly, you can get the most frequently occurring value(s), i.e. the mode, of the values in a Series or DataFrame:

Continuous values can be discretized using the cut() (bins based on values) and qcut() (bins based on sample quantiles) functions:

qcut() computes sample quantiles. For example, we could slice up some normally distributed data into equal-size quartiles like so:

We can also pass infinite values to define the bins:

To apply your own or another library’s functions to pandas objects, you should be aware of the three methods below. The appropriate method to use depends on whether your function expects to operate on an entire DataFrame or Series, row- or column-wise, or elementwise.

Tablewise Function Application: pipe()

Row or Column-wise Function Application: apply()

Aggregation API: agg() and transform()

Applying Elementwise Functions: map()

DataFrames and Series can be passed into functions. However, if the function needs to be called in a chain, consider using the pipe() method.

extract_city_name and add_country_name are functions taking and returning DataFrames.

Now compare the following:

pandas encourages the second style, which is known as method chaining. pipe makes it easy to use your own or another library’s functions in method chains, alongside pandas’ methods.

In the example above, the functions extract_city_name and add_country_name each expected a DataFrame as the first positional argument. What if the function you wish to apply takes its data as, say, the second argument? In this case, provide pipe with a tuple of (callable, data_keyword). .pipe will route the DataFrame to the argument specified in the tuple.

For example, we can fit a regression using statsmodels. Their API expects a formula first and a DataFrame as the second argument, data. We pass in the function, keyword pair (sm.ols, 'data') to pipe:

The pipe method is inspired by unix pipes and more recently dplyr and magrittr, which have introduced the popular (%>%) (read pipe) operator for R. The implementation of pipe here is quite clean and feels right at home in Python. We encourage you to view the source code of pipe().

Arbitrary functions can be applied along the axes of a DataFrame using the apply() method, which, like the descriptive statistics methods, takes an optional axis argument:

The apply() method will also dispatch on a string method name.

The return type of the function passed to apply() affects the type of the final output from DataFrame.apply for the default behaviour:

If the applied function returns a Series, the final output is a DataFrame. The columns match the index of the Series returned by the applied function.

If the applied function returns any other type, the final output is a Series.

This default behaviour can be overridden using the result_type, which accepts three options: reduce, broadcast, and expand. These will determine how list-likes return values expand (or not) to a DataFrame.

apply() combined with some cleverness can be used to answer many questions about a data set. For example, suppose we wanted to extract the date where the maximum value for each column occurred:

You may also pass additional arguments and keyword arguments to the apply() method.

Another useful feature is the ability to pass Series methods to carry out some Series operation on each column or row:

Finally, apply() takes an argument raw which is False by default, which converts each row or column into a Series before applying the function. When set to True, the passed function will instead receive an ndarray object, which has positive performance implications if you do not need the indexing functionality.

The aggregation API allows one to express possibly multiple aggregation operations in a single concise way. This API is similar across pandas objects, see groupby API, the window API, and the resample API. The entry point for aggregation is DataFrame.aggregate(), or the alias DataFrame.agg().

We will use a similar starting frame from above:

Using a single function is equivalent to apply(). You can also pass named methods as strings. These will return a Series of the aggregated output:

Single aggregations on a Series this will return a scalar value:

You can pass multiple aggregation arguments as a list. The results of each of the passed functions will be a row in the resulting DataFrame. These are naturally named from the aggregation function.

Multiple functions yield multiple rows:

On a Series, multiple functions return a Series, indexed by the function names:

Passing a lambda function will yield a <lambda> named row:

Passing a named function will yield that name for the row:

Passing a dictionary of column names to a scalar or a list of scalars, to DataFrame.agg allows you to customize which functions are applied to which columns. Note that the results are not in any particular order, you can use an OrderedDict instead to guarantee ordering.

Passing a list-like will generate a DataFrame output. You will get a matrix-like output of all of the aggregators. The output will consist of all unique functions. Those that are not noted for a particular column will be NaN:

With .agg() it is possible to easily create a custom describe function, similar to the built in describe function.

The transform() method returns an object that is indexed the same (same size) as the original. This API allows you to provide multiple operations at the same time rather than one-by-one. Its API is quite similar to the .agg API.

We create a frame similar to the one used in the above sections.

Transform the entire frame. .transform() allows input functions as: a NumPy function, a string function name or a user defined function.

Here transform() received a single function; this is equivalent to a ufunc application.

Passing a single function to .transform() with a Series will yield a single Series in return.

Passing multiple functions will yield a column MultiIndexed DataFrame. The first level will be the original frame column names; the second level will be the names of the transforming functions.

Passing multiple functions to a Series will yield a DataFrame. The resulting column names will be the transforming functions.

Passing a dict of functions will allow selective transforming per column.

Passing a dict of lists will generate a MultiIndexed DataFrame with these selective transforms.

Since not all functions can be vectorized (accept NumPy arrays and return another array or value), the methods map() on DataFrame and analogously map() on Series accept any Python function taking a single value and returning a single value. For example:

Series.map() has an additional feature; it can be used to easily “link” or “map” values defined by a secondary series. This is closely related to merging/joining functionality:

reindex() is the fundamental data alignment method in pandas. It is used to implement nearly all other features relying on label-alignment functionality. To reindex means to conform the data to match a given set of labels along a particular axis. This accomplishes several things:

Reorders the existing data to match a new set of labels

Inserts missing value (NA) markers in label locations where no data for that label existed

If specified, fill data for missing labels using logic (highly relevant to working with time series data)

Here is a simple example:

Here, the f label was not contained in the Series and hence appears as NaN in the result.

With a DataFrame, you can simultaneously reindex the index and columns:

Note that the Index objects containing the actual axis labels can be shared between objects. So if we have a Series and a DataFrame, the following can be done:

This means that the reindexed Series’s index is the same Python object as the DataFrame’s index.

DataFrame.reindex() also supports an “axis-style” calling convention, where you specify a single labels argument and the axis it applies to.

MultiIndex / Advanced Indexing is an even more concise way of doing reindexing.

When writing performance-sensitive code, there is a good reason to spend some time becoming a reindexing ninja: many operations are faster on pre-aligned data. Adding two unaligned DataFrames internally triggers a reindexing step. For exploratory analysis you will hardly notice the difference (because reindex has been heavily optimized), but when CPU cycles matter sprinkling a few explicit reindex calls here and there can have an impact.

You may wish to take an object and reindex its axes to be labeled the same as another object. While the syntax for this is straightforward albeit verbose, it is a common enough operation that the reindex_like() method is available to make this simpler:

The align() method is the fastest way to simultaneously align two objects. It supports a join argument (related to joining and merging):

join='outer': take the union of the indexes (default)

join='left': use the calling object’s index

join='right': use the passed object’s index

join='inner': intersect the indexes

It returns a tuple with both of the reindexed Series:

For DataFrames, the join method will be applied to both the index and the columns by default:

You can also pass an axis option to only align on the specified axis:

If you pass a Series to DataFrame.align(), you can choose to align both objects either on the DataFrame’s index or columns using the axis argument:

reindex() takes an optional parameter method which is a filling method chosen from the following table:

Fill from the nearest index value

We illustrate these fill methods on a simple Series:

These methods require that the indexes are ordered increasing or decreasing.

Note that the same result could have been achieved using ffill (except for method='nearest') or interpolate:

reindex() will raise a ValueError if the index is not monotonically increasing or decreasing. fillna() and interpolate() will not perform any checks on the order of the index.

The limit and tolerance arguments provide additional control over filling while reindexing. Limit specifies the maximum count of consecutive matches:

In contrast, tolerance specifies the maximum distance between the index and indexer values:

Notice that when used on a DatetimeIndex, TimedeltaIndex or PeriodIndex, tolerance will coerced into a Timedelta if possible. This allows you to specify tolerance with appropriate strings.

A method closely related to reindex is the drop() function. It removes a set of labels from an axis:

Note that the following also works, but is a bit less obvious / clean:

The rename() method allows you to relabel an axis based on some mapping (a dict or Series) or an arbitrary function.

If you pass a function, it must return a value when called with any of the labels (and must produce a set of unique values). A dict or Series can also be used:

If the mapping doesn’t include a column/index label, it isn’t renamed. Note that extra labels in the mapping don’t throw an error.

DataFrame.rename() also supports an “axis-style” calling convention, where you specify a single mapper and the axis to apply that mapping to.

Finally, rename() also accepts a scalar or list-like for altering the Series.name attribute.

The methods DataFrame.rename_axis() and Series.rename_axis() allow specific names of a MultiIndex to be changed (as opposed to the labels).

The behavior of basic iteration over pandas objects depends on the type. When iterating over a Series, it is regarded as array-like, and basic iteration produces the values. DataFrames follow the dict-like convention of iterating over the “keys” of the objects.

In short, basic iteration (for i in object) produces:

DataFrame: column labels

Thus, for example, iterating over a DataFrame gives you the column names:

pandas objects also have the dict-like items() method to iterate over the (key, value) pairs.

To iterate over the rows of a DataFrame, you can use the following methods:

iterrows(): Iterate over the rows of a DataFrame as (index, Series) pairs. This converts the rows to Series objects, which can change the dtypes and has some performance implications.

itertuples(): Iterate over the rows of a DataFrame as namedtuples of the values. This is a lot faster than iterrows(), and is in most cases preferable to use to iterate over the values of a DataFrame.

Iterating through pandas objects is generally slow. In many cases, iterating manually over the rows is not needed and can be avoided with one of the following approaches:

Look for a vectorized solution: many operations can be performed using built-in methods or NumPy functions, (boolean) indexing, …

When you have a function that cannot work on the full DataFrame/Series at once, it is better to use apply() instead of iterating over the values. See the docs on function application.

If you need to do iterative manipulations on the values but performance is important, consider writing the inner loop with cython or numba. See the enhancing performance section for some examples of this approach.

You should never modify something you are iterating over. This is not guaranteed to work in all cases. Depending on the data types, the iterator returns a copy and not a view, and writing to it will have no effect!

For example, in the following case setting the value has no effect:

Consistent with the dict-like interface, items() iterates through key-value pairs:

Series: (index, scalar value) pairs

DataFrame: (column, Series) pairs

iterrows() allows you to iterate through the rows of a DataFrame as Series objects. It returns an iterator yielding each index value along with a Series containing the data in each row:

Because iterrows() returns a Series for each row, it does not preserve dtypes across the rows (dtypes are preserved across columns for DataFrames). For example,

All values in row, returned as a Series, are now upcasted to floats, also the original integer value in column x:

To preserve dtypes while iterating over the rows, it is better to use itertuples() which returns namedtuples of the values and which is generally much faster than iterrows().

For instance, a contrived way to transpose the DataFrame would be:

The itertuples() method will return an iterator yielding a namedtuple for each row in the DataFrame. The first element of the tuple will be the row’s corresponding index value, while the remaining values are the row values.

This method does not convert the row to a Series object; it merely returns the values inside a namedtuple. Therefore, itertuples() preserves the data type of the values and is generally faster as iterrows().

The column names will be renamed to positional names if they are invalid Python identifiers, repeated, or start with an underscore. With a large number of columns (>255), regular tuples are returned.

Series has an accessor to succinctly return datetime like properties for the values of the Series, if it is a datetime/period like Series. This will return a Series, indexed like the existing Series.

This enables nice expressions like this:

You can easily produces tz aware transformations:

You can also chain these types of operations:

You can also format datetime values as strings with Series.dt.strftime() which supports the same format as the standard strftime().

The .dt accessor works for period and timedelta dtypes.

Series.dt will raise a TypeError if you access with a non-datetime-like values.

Series is equipped with a set of string processing methods that make it easy to operate on each element of the array. Perhaps most importantly, these methods exclude missing/NA values automatically. These are accessed via the Series’s str attribute and generally have names matching the equivalent (scalar) built-in string methods. For example:

Powerful pattern-matching methods are provided as well, but note that pattern-matching generally uses regular expressions by default (and in some cases always uses them).

Prior to pandas 1.0, string methods were only available on object -dtype Series. pandas 1.0 added the StringDtype which is dedicated to strings. See Text data types for more.

Please see Vectorized String Methods for a complete description.

pandas supports three kinds of sorting: sorting by index labels, sorting by column values, and sorting by a combination of both.

The Series.sort_index() and DataFrame.sort_index() methods are used to sort a pandas object by its index levels.

Sorting by index also supports a key parameter that takes a callable function to apply to the index being sorted. For MultiIndex objects, the key is applied per-level to the levels specified by level.

For information on key sorting by value, see value sorting.

The Series.sort_values() method is used to sort a Series by its values. The DataFrame.sort_values() method is used to sort a DataFrame by its column or row values. The optional by parameter to DataFrame.sort_values() may used to specify one or more columns to use to determine the sorted order.

The by parameter can take a list of column names, e.g.:

These methods have special treatment of NA values via the na_position argument:

Sorting also supports a key parameter that takes a callable function to apply to the values being sorted.

key will be given the Series of values and should return a Series or array of the same shape with the transformed values. For DataFrame objects, the key is applied per column, so the key should still expect a Series and return a Series, e.g.

The name or type of each column can be used to apply different functions to different columns.

Strings passed as the by parameter to DataFrame.sort_values() may refer to either columns or index level names.

Sort by ‘second’ (index) and ‘A’ (column)

If a string matches both a column name and an index level name then a warning is issued and the column takes precedence. This will result in an ambiguity error in a future version.

Series has the searchsorted() method, which works similarly to numpy.ndarray.searchsorted().

Series has the nsmallest() and nlargest() methods which return the smallest or largest \(n\) values. For a large Series this can be much faster than sorting the entire Series and calling head(n) on the result.

DataFrame also has the nlargest and nsmallest methods.

You must be explicit about sorting when the column is a MultiIndex, and fully specify all levels to by.

The copy() method on pandas objects copies the underlying data (though not the axis indexes, since they are immutable) and returns a new object. Note that it is seldom necessary to copy objects. For example, there are only a handful of ways to alter a DataFrame in-place:

Inserting, deleting, or modifying a column.

Assigning to the index or columns attributes.

For homogeneous data, directly modifying the values via the values attribute or advanced indexing.

To be clear, no pandas method has the side effect of modifying your data; almost every method returns a new object, leaving the original object untouched. If the data is modified, it is because you did so explicitly.

For the most part, pandas uses NumPy arrays and dtypes for Series or individual columns of a DataFrame. NumPy provides support for float, int, bool, timedelta64[ns] and datetime64[ns] (note that NumPy does not support timezone-aware datetimes).

pandas and third-party libraries extend NumPy’s type system in a few places. This section describes the extensions pandas has made internally. See Extension types for how to write your own extension that works with pandas. See the ecosystem page for a list of third-party libraries that have implemented an extension.

The following table lists all of pandas extension types. For methods requiring dtype arguments, strings can be specified as indicated. See the respective documentation sections for more on each type.

'datetime64[ns, <tz>]'

arrays.PeriodArray 'Period[<freq>]'

'Sparse', 'Sparse[int]', 'Sparse[float]'

'interval', 'Interval', 'Interval[<numpy_dtype>]', 'Interval[datetime64[ns, <tz>]]', 'Interval[timedelta64[<freq>]]'

'Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64'

pandas has two ways to store strings.

object dtype, which can hold any Python object, including strings.

StringDtype, which is dedicated to strings.

Generally, we recommend using StringDtype. See Text data types for more.

Finally, arbitrary objects may be stored using the object dtype, but should be avoided to the extent possible (for performance and interoperability with other libraries and methods. See object conversion).

A convenient dtypes attribute for DataFrame returns a Series with the data type of each column.

On a Series object, use the dtype attribute.

If a pandas object contains data with multiple dtypes in a single column, the dtype of the column will be chosen to accommodate all of the data types (object is the most general).

The number of columns of each type in a DataFrame can be found by calling DataFrame.dtypes.value_counts().

Numeric dtypes will propagate and can coexist in DataFrames. If a dtype is passed (either directly via the dtype keyword, a passed ndarray, or a passed Series), then it will be preserved in DataFrame operations. Furthermore, different numeric dtypes will NOT be combined. The following example will give you a taste.

By default integer types are int64 and float types are float64, regardless of platform (32-bit or 64-bit). The following will all result in int64 dtypes.

Note that Numpy will choose platform-dependent types when creating arrays. The following WILL result in int32 on 32-bit platform.

Types can potentially be upcasted when combined with other types, meaning they are promoted from the current type (e.g. int to float).

DataFrame.to_numpy() will return the lower-common-denominator of the dtypes, meaning the dtype that can accommodate ALL of the types in the resulting homogeneous dtyped NumPy array. This can force some upcasting.

You can use the astype() method to explicitly convert dtypes from one to another. These will by default return a copy, even if the dtype was unchanged (pass copy=False to change this behavior). In addition, they will raise an exception if the astype operation is invalid.

Upcasting is always according to the NumPy rules. If two different dtypes are involved in an operation, then the more general one will be used as the result of the operation.

Convert a subset of columns to a specified type using astype().

Convert certain columns to a specific dtype by passing a dict to astype().

When trying to convert a subset of columns to a specified type using astype() and loc(), upcasting occurs.

loc() tries to fit in what we are assigning to the current dtypes, while [] will overwrite them taking the dtype from the right hand side. Therefore the following piece of code produces the unintended result.

pandas offers various functions to try to force conversion of types from the object dtype to other types. In cases where the data is already of the correct type, but stored in an object array, the DataFrame.infer_objects() and Series.infer_objects() methods can be used to soft convert to the correct type.

Because the data was transposed the original inference stored all columns as object, which infer_objects will correct.

The following functions are available for one dimensional object arrays or scalars to perform hard conversion of objects to a specified type:

to_numeric() (conversion to numeric dtypes)

to_datetime() (conversion to datetime objects)

to_timedelta() (conversion to timedelta objects)

To force a conversion, we can pass in an errors argument, which specifies how pandas should deal with elements that cannot be converted to desired dtype or object. By default, errors='raise', meaning that any errors encountered will be raised during the conversion process. However, if errors='coerce', these errors will be ignored and pandas will convert problematic elements to pd.NaT (for datetime and timedelta) or np.nan (for numeric). This might be useful if you are reading in data which is mostly of the desired dtype (e.g. numeric, datetime), but occasionally has non-conforming elements intermixed that you want to represent as missing:

In addition to object conversion, to_numeric() provides another argument downcast, which gives the option of downcasting the newly (or already) numeric data to a smaller dtype, which can conserve memory:

As these methods apply only to one-dimensional arrays, lists or scalars; they cannot be used directly on multi-dimensional objects such as DataFrames. However, with apply(), we can “apply” the function over each column efficiently:

Performing selection operations on integer type data can easily upcast the data to floating. The dtype of the input data will be preserved in cases where nans are not introduced. See also Support for integer NA.

While float dtypes are unchanged.

The select_dtypes() method implements subsetting of columns based on their dtype.

First, let’s create a DataFrame with a slew of different dtypes:

select_dtypes() has two parameters include and exclude that allow you to say “give me the columns with these dtypes” (include) and/or “give the columns without these dtypes” (exclude).

For example, to select bool columns:

You can also pass the name of a dtype in the NumPy dtype hierarchy:

select_dtypes() also works with generic dtypes as well.

For example, to select all numeric and boolean columns while excluding unsigned integers:

To select string columns you must use the object dtype:

To see all the child dtypes of a generic dtype like numpy.number you can define a function that returns a tree of child dtypes:

All NumPy dtypes are subclasses of numpy.generic:

pandas also defines the types category, and datetime64[ns, tz], which are not integrated into the normal NumPy hierarchy and won’t show up with the above function.

**Examples:**

Example 1 (typescript):
```typescript
In [1]: index = pd.date_range("1/1/2000", periods=8)

In [2]: s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])

In [3]: df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=["A", "B", "C"])
```

Example 2 (typescript):
```typescript
In [4]: long_series = pd.Series(np.random.randn(1000))

In [5]: long_series.head()
Out[5]: 
0   -1.157892
1   -1.344312
2    0.844885
3    1.075770
4   -0.109050
dtype: float64

In [6]: long_series.tail(3)
Out[6]: 
997   -0.289388
998   -1.020544
999    0.589993
dtype: float64
```

Example 3 (json):
```json
In [7]: df[:2]
Out[7]: 
                   A         B         C
2000-01-01 -0.173215  0.119209 -1.044236
2000-01-02 -0.861849 -2.104569 -0.494929

In [8]: df.columns = [x.lower() for x in df.columns]

In [9]: df
Out[9]: 
                   a         b         c
2000-01-01 -0.173215  0.119209 -1.044236
2000-01-02 -0.861849 -2.104569 -0.494929
2000-01-03  1.071804  0.721555 -0.706771
2000-01-04 -1.039575  0.271860 -0.424972
2000-01-05  0.567020  0.276232 -1.087401
2000-01-06 -0.673690  0.113648 -1.478427
2000-01-07  0.524988  0.404705  0.577046
2000-01-08 -1.715002 -1.039268 -0.370647
```

Example 4 (json):
```json
In [10]: s.array
Out[10]: 
<NumpyExtensionArray>
[ 0.4691122999071863, -0.2828633443286633, -1.5090585031735124,
 -1.1356323710171934,  1.2121120250208506]
Length: 5, dtype: float64

In [11]: s.index.array
Out[11]: 
<NumpyExtensionArray>
['a', 'b', 'c', 'd', 'e']
Length: 5, dtype: object
```

---

## PyArrow Functionality#

**URL:** https://pandas.pydata.org/docs/user_guide/pyarrow.html

**Contents:**
- PyArrow Functionality#
- Data Structure Integration#
- Operations#
- I/O Reading#

pandas can utilize PyArrow to extend functionality and improve the performance of various APIs. This includes:

More extensive data types compared to NumPy

Missing data support (NA) for all data types

Performant IO reader integration

Facilitate interoperability with other dataframe libraries based on the Apache Arrow specification (e.g. polars, cuDF)

To use this functionality, please ensure you have installed the minimum supported PyArrow version.

A Series, Index, or the columns of a DataFrame can be directly backed by a pyarrow.ChunkedArray which is similar to a NumPy array. To construct these from the main pandas data structures, you can pass in a string of the type followed by [pyarrow], e.g. "int64[pyarrow]"" into the dtype parameter

The string alias "string[pyarrow]" maps to pd.StringDtype("pyarrow") which is not equivalent to specifying dtype=pd.ArrowDtype(pa.string()). Generally, operations on the data will behave similarly except pd.StringDtype("pyarrow") can return NumPy-backed nullable types while pd.ArrowDtype(pa.string()) will return ArrowDtype.

For PyArrow types that accept parameters, you can pass in a PyArrow type with those parameters into ArrowDtype to use in the dtype parameter.

If you already have an pyarrow.Array or pyarrow.ChunkedArray, you can pass it into arrays.ArrowExtensionArray to construct the associated Series, Index or DataFrame object.

To retrieve a pyarrow pyarrow.ChunkedArray from a Series or Index, you can call the pyarrow array constructor on the Series or Index.

To convert a pyarrow.Table to a DataFrame, you can call the pyarrow.Table.to_pandas() method with types_mapper=pd.ArrowDtype.

PyArrow data structure integration is implemented through pandas’ ExtensionArray interface; therefore, supported functionality exists where this interface is integrated within the pandas API. Additionally, this functionality is accelerated with PyArrow compute functions where available. This includes:

Logical and comparison functions

Datetime functionality

The following are just some examples of operations that are accelerated by native PyArrow compute functions.

PyArrow also provides IO reading functionality that has been integrated into several pandas IO readers. The following functions provide an engine keyword that can dispatch to PyArrow to accelerate reading from an IO source.

By default, these functions and all other IO reader functions return NumPy-backed data. These readers can return PyArrow-backed data by specifying the parameter dtype_backend="pyarrow". A reader does not need to set engine="pyarrow" to necessarily return PyArrow-backed data.

Several non-IO reader functions can also use the dtype_backend argument to return PyArrow-backed data including:

DataFrame.convert_dtypes()

Series.convert_dtypes()

**Examples:**

Example 1 (typescript):
```typescript
In [1]: ser = pd.Series([-1.5, 0.2, None], dtype="float32[pyarrow]")

In [2]: ser
Out[2]: 
0    -1.5
1     0.2
2    <NA>
dtype: float[pyarrow]

In [3]: idx = pd.Index([True, None], dtype="bool[pyarrow]")

In [4]: idx
Out[4]: Index([True, <NA>], dtype='bool[pyarrow]')

In [5]: df = pd.DataFrame([[1, 2], [3, 4]], dtype="uint64[pyarrow]")

In [6]: df
Out[6]: 
   0  1
0  1  2
1  3  4
```

Example 2 (typescript):
```typescript
In [7]: import pyarrow as pa

In [8]: data = list("abc")

In [9]: ser_sd = pd.Series(data, dtype="string[pyarrow]")

In [10]: ser_ad = pd.Series(data, dtype=pd.ArrowDtype(pa.string()))

In [11]: ser_ad.dtype == ser_sd.dtype
Out[11]: False

In [12]: ser_sd.str.contains("a")
Out[12]: 
0     True
1    False
2    False
dtype: boolean

In [13]: ser_ad.str.contains("a")
Out[13]: 
0     True
1    False
2    False
dtype: bool[pyarrow]
```

Example 3 (typescript):
```typescript
In [14]: import pyarrow as pa

In [15]: list_str_type = pa.list_(pa.string())

In [16]: ser = pd.Series([["hello"], ["there"]], dtype=pd.ArrowDtype(list_str_type))

In [17]: ser
Out[17]: 
0    ['hello']
1    ['there']
dtype: list<item: string>[pyarrow]
```

Example 4 (typescript):
```typescript
In [18]: from datetime import time

In [19]: idx = pd.Index([time(12, 30), None], dtype=pd.ArrowDtype(pa.time64("us")))

In [20]: idx
Out[20]: Index([12:30:00, <NA>], dtype='time64[us][pyarrow]')
```

---

## Chart visualization#

**URL:** https://pandas.pydata.org/docs/user_guide/visualization.html

**Contents:**
- Chart visualization#
- Basic plotting: plot#
- Other plots#
  - Bar plots#
  - Histograms#
  - Box plots#
  - Area plot#
  - Scatter plot#
  - Hexagonal bin plot#
  - Pie plot#

The examples below assume that you’re using Jupyter.

This section demonstrates visualization through charting. For information on visualization of tabular data please see the section on Table Visualization.

We use the standard convention for referencing the matplotlib API:

We provide the basics in pandas to easily create decent looking plots. See the ecosystem page for visualization libraries that go beyond the basics documented here.

All calls to np.random are seeded with 123456.

We will demonstrate the basics, see the cookbook for some advanced strategies.

The plot method on Series and DataFrame is just a simple wrapper around plt.plot():

If the index consists of dates, it calls gcf().autofmt_xdate() to try to format the x-axis nicely as per above.

On DataFrame, plot() is a convenience to plot all of the columns with labels:

You can plot one column versus another using the x and y keywords in plot():

For more formatting and styling options, see formatting below.

Plotting methods allow for a handful of plot styles other than the default line plot. These methods can be provided as the kind keyword argument to plot(), and include:

‘bar’ or ‘barh’ for bar plots

‘kde’ or ‘density’ for density plots

‘area’ for area plots

‘scatter’ for scatter plots

‘hexbin’ for hexagonal bin plots

For example, a bar plot can be created the following way:

You can also create these other plots using the methods DataFrame.plot.<kind> instead of providing the kind keyword argument. This makes it easier to discover plot methods and the specific arguments they use:

In addition to these kind s, there are the DataFrame.hist(), and DataFrame.boxplot() methods, which use a separate interface.

Finally, there are several plotting functions in pandas.plotting that take a Series or DataFrame as an argument. These include:

Plots may also be adorned with errorbars or tables.

For labeled, non-time series data, you may wish to produce a bar plot:

Calling a DataFrame’s plot.bar() method produces a multiple bar plot:

To produce a stacked bar plot, pass stacked=True:

To get horizontal bar plots, use the barh method:

Histograms can be drawn by using the DataFrame.plot.hist() and Series.plot.hist() methods.

A histogram can be stacked using stacked=True. Bin size can be changed using the bins keyword.

You can pass other keywords supported by matplotlib hist. For example, horizontal and cumulative histograms can be drawn by orientation='horizontal' and cumulative=True.

See the hist method and the matplotlib hist documentation for more.

The existing interface DataFrame.hist to plot histogram still can be used.

DataFrame.hist() plots the histograms of the columns on multiple subplots:

The by keyword can be specified to plot grouped histograms:

In addition, the by keyword can also be specified in DataFrame.plot.hist().

Changed in version 1.4.0.

Boxplot can be drawn calling Series.plot.box() and DataFrame.plot.box(), or DataFrame.boxplot() to visualize the distribution of values within each column.

For instance, here is a boxplot representing five trials of 10 observations of a uniform random variable on [0,1).

Boxplot can be colorized by passing color keyword. You can pass a dict whose keys are boxes, whiskers, medians and caps. If some keys are missing in the dict, default colors are used for the corresponding artists. Also, boxplot has sym keyword to specify fliers style.

When you pass other type of arguments via color keyword, it will be directly passed to matplotlib for all the boxes, whiskers, medians and caps colorization.

The colors are applied to every boxes to be drawn. If you want more complicated colorization, you can get each drawn artists by passing return_type.

Also, you can pass other keywords supported by matplotlib boxplot. For example, horizontal and custom-positioned boxplot can be drawn by vert=False and positions keywords.

See the boxplot method and the matplotlib boxplot documentation for more.

The existing interface DataFrame.boxplot to plot boxplot still can be used.

You can create a stratified boxplot using the by keyword argument to create groupings. For instance,

You can also pass a subset of columns to plot, as well as group by multiple columns:

You could also create groupings with DataFrame.plot.box(), for instance:

Changed in version 1.4.0.

In boxplot, the return type can be controlled by the return_type, keyword. The valid choices are {"axes", "dict", "both", None}. Faceting, created by DataFrame.boxplot with the by keyword, will affect the output type as well:

Series of dicts of artists

Series of namedtuples

Groupby.boxplot always returns a Series of return_type.

The subplots above are split by the numeric columns first, then the value of the g column. Below the subplots are first split by the value of g, then by the numeric columns.

You can create area plots with Series.plot.area() and DataFrame.plot.area(). Area plots are stacked by default. To produce stacked area plot, each column must be either all positive or all negative values.

When input data contains NaN, it will be automatically filled by 0. If you want to drop or fill by different values, use dataframe.dropna() or dataframe.fillna() before calling plot.

To produce an unstacked plot, pass stacked=False. Alpha value is set to 0.5 unless otherwise specified:

Scatter plot can be drawn by using the DataFrame.plot.scatter() method. Scatter plot requires numeric columns for the x and y axes. These can be specified by the x and y keywords.

To plot multiple column groups in a single axes, repeat plot method specifying target ax. It is recommended to specify color and label keywords to distinguish each groups.

The keyword c may be given as the name of a column to provide colors for each point:

If a categorical column is passed to c, then a discrete colorbar will be produced:

Added in version 1.3.0.

You can pass other keywords supported by matplotlib scatter. The example below shows a bubble chart using a column of the DataFrame as the bubble size.

See the scatter method and the matplotlib scatter documentation for more.

You can create hexagonal bin plots with DataFrame.plot.hexbin(). Hexbin plots can be a useful alternative to scatter plots if your data are too dense to plot each point individually.

A useful keyword argument is gridsize; it controls the number of hexagons in the x-direction, and defaults to 100. A larger gridsize means more, smaller bins.

By default, a histogram of the counts around each (x, y) point is computed. You can specify alternative aggregations by passing values to the C and reduce_C_function arguments. C specifies the value at each (x, y) point and reduce_C_function is a function of one argument that reduces all the values in a bin to a single number (e.g. mean, max, sum, std). In this example the positions are given by columns a and b, while the value is given by column z. The bins are aggregated with NumPy’s max function.

See the hexbin method and the matplotlib hexbin documentation for more.

You can create a pie plot with DataFrame.plot.pie() or Series.plot.pie(). If your data includes any NaN, they will be automatically filled with 0. A ValueError will be raised if there are any negative values in your data.

For pie plots it’s best to use square figures, i.e. a figure aspect ratio 1. You can create the figure with equal width and height, or force the aspect ratio to be equal after plotting by calling ax.set_aspect('equal') on the returned axes object.

Note that pie plot with DataFrame requires that you either specify a target column by the y argument or subplots=True. When y is specified, pie plot of selected column will be drawn. If subplots=True is specified, pie plots for each column are drawn as subplots. A legend will be drawn in each pie plots by default; specify legend=False to hide it.

You can use the labels and colors keywords to specify the labels and colors of each wedge.

Most pandas plots use the label and color arguments (note the lack of “s” on those). To be consistent with matplotlib.pyplot.pie() you must use labels and colors.

If you want to hide wedge labels, specify labels=None. If fontsize is specified, the value will be applied to wedge labels. Also, other keywords supported by matplotlib.pyplot.pie() can be used.

If you pass values whose sum total is less than 1.0 they will be rescaled so that they sum to 1.

See the matplotlib pie documentation for more.

pandas tries to be pragmatic about plotting DataFrames or Series that contain missing data. Missing values are dropped, left out, or filled depending on the plot type.

Drop NaNs (column-wise)

Drop NaNs (column-wise)

Drop NaNs (column-wise)

If any of these defaults are not what you want, or if you want to be explicit about how missing values are handled, consider using fillna() or dropna() before plotting.

These functions can be imported from pandas.plotting and take a Series or DataFrame as an argument.

You can create a scatter plot matrix using the scatter_matrix method in pandas.plotting:

You can create density plots using the Series.plot.kde() and DataFrame.plot.kde() methods.

Andrews curves allow one to plot multivariate data as a large number of curves that are created using the attributes of samples as coefficients for Fourier series, see the Wikipedia entry for more information. By coloring these curves differently for each class it is possible to visualize data clustering. Curves belonging to samples of the same class will usually be closer together and form larger structures.

Note: The “Iris” dataset is available here.

Parallel coordinates is a plotting technique for plotting multivariate data, see the Wikipedia entry for an introduction. Parallel coordinates allows one to see clusters in data and to estimate other statistics visually. Using parallel coordinates points are represented as connected line segments. Each vertical line represents one attribute. One set of connected line segments represents one data point. Points that tend to cluster will appear closer together.

Lag plots are used to check if a data set or time series is random. Random data should not exhibit any structure in the lag plot. Non-random structure implies that the underlying data are not random. The lag argument may be passed, and when lag=1 the plot is essentially data[:-1] vs. data[1:].

Autocorrelation plots are often used for checking randomness in time series. This is done by computing autocorrelations for data values at varying time lags. If time series is random, such autocorrelations should be near zero for any and all time-lag separations. If time series is non-random then one or more of the autocorrelations will be significantly non-zero. The horizontal lines displayed in the plot correspond to 95% and 99% confidence bands. The dashed line is 99% confidence band. See the Wikipedia entry for more about autocorrelation plots.

Bootstrap plots are used to visually assess the uncertainty of a statistic, such as mean, median, midrange, etc. A random subset of a specified size is selected from a data set, the statistic in question is computed for this subset and the process is repeated a specified number of times. Resulting plots and histograms are what constitutes the bootstrap plot.

RadViz is a way of visualizing multi-variate data. It is based on a simple spring tension minimization algorithm. Basically you set up a bunch of points in a plane. In our case they are equally spaced on a unit circle. Each point represents a single attribute. You then pretend that each sample in the data set is attached to each of these points by a spring, the stiffness of which is proportional to the numerical value of that attribute (they are normalized to unit interval). The point in the plane, where our sample settles to (where the forces acting on our sample are at an equilibrium) is where a dot representing our sample will be drawn. Depending on which class that sample belongs it will be colored differently. See the R package Radviz for more information.

Note: The “Iris” dataset is available here.

From version 1.5 and up, matplotlib offers a range of pre-configured plotting styles. Setting the style can be used to easily give plots the general look that you want. Setting the style is as easy as calling matplotlib.style.use(my_plot_style) before creating your plot. For example you could write matplotlib.style.use('ggplot') for ggplot-style plots.

You can see the various available style names at matplotlib.style.available and it’s very easy to try them out.

Most plotting methods have a set of keyword arguments that control the layout and formatting of the returned plot:

For each kind of plot (e.g. line, bar, scatter) any additional arguments keywords are passed along to the corresponding matplotlib function (ax.plot(), ax.bar(), ax.scatter()). These can be used to control additional styling, beyond what pandas provides.

You may set the legend argument to False to hide the legend, which is shown by default.

You may set the xlabel and ylabel arguments to give the plot custom labels for x and y axis. By default, pandas will pick up index name as xlabel, while leaving it empty for ylabel.

You may pass logy to get a log-scale Y axis.

See also the logx and loglog keyword arguments.

To plot data on a secondary y-axis, use the secondary_y keyword:

To plot some columns in a DataFrame, give the column names to the secondary_y keyword:

Note that the columns plotted on the secondary y-axis is automatically marked with “(right)” in the legend. To turn off the automatic marking, use the mark_right=False keyword:

pandas provides custom formatters for timeseries plots. These change the formatting of the axis labels for dates and times. By default, the custom formatters are applied only to plots created by pandas with DataFrame.plot() or Series.plot(). To have them apply to all plots, including those made by matplotlib, set the option pd.options.plotting.matplotlib.register_converters = True or use pandas.plotting.register_matplotlib_converters().

pandas includes automatic tick resolution adjustment for regular frequency time-series data. For limited cases where pandas cannot infer the frequency information (e.g., in an externally created twinx), you can choose to suppress this behavior for alignment purposes.

Here is the default behavior, notice how the x-axis tick labeling is performed:

Using the x_compat parameter, you can suppress this behavior:

If you have more than one plot that needs to be suppressed, the use method in pandas.plotting.plot_params can be used in a with statement:

TimedeltaIndex now uses the native matplotlib tick locator methods, it is useful to call the automatic date tick adjustment from matplotlib for figures whose ticklabels overlap.

See the autofmt_xdate method and the matplotlib documentation for more.

Each Series in a DataFrame can be plotted on a different axis with the subplots keyword:

The layout of subplots can be specified by the layout keyword. It can accept (rows, columns). The layout keyword can be used in hist and boxplot also. If the input is invalid, a ValueError will be raised.

The number of axes which can be contained by rows x columns specified by layout must be larger than the number of required subplots. If layout can contain more axes than required, blank axes are not drawn. Similar to a NumPy array’s reshape method, you can use -1 for one dimension to automatically calculate the number of rows or columns needed, given the other.

The above example is identical to using:

The required number of columns (3) is inferred from the number of series to plot and the given number of rows (2).

You can pass multiple axes created beforehand as list-like via ax keyword. This allows more complicated layouts. The passed axes must be the same number as the subplots being drawn.

When multiple axes are passed via the ax keyword, layout, sharex and sharey keywords don’t affect to the output. You should explicitly pass sharex=False and sharey=False, otherwise you will see a warning.

Another option is passing an ax argument to Series.plot() to plot on a particular axis:

Plotting with error bars is supported in DataFrame.plot() and Series.plot().

Horizontal and vertical error bars can be supplied to the xerr and yerr keyword arguments to plot(). The error values can be specified using a variety of formats:

As a DataFrame or dict of errors with column names matching the columns attribute of the plotting DataFrame or matching the name attribute of the Series.

As a str indicating which of the columns of plotting DataFrame contain the error values.

As raw values (list, tuple, or np.ndarray). Must be the same length as the plotting DataFrame/Series.

Here is an example of one way to easily plot group means with standard deviations from the raw data.

Asymmetrical error bars are also supported, however raw error values must be provided in this case. For a N length Series, a 2xN array should be provided indicating lower and upper (or left and right) errors. For a MxN DataFrame, asymmetrical errors should be in a Mx2xN array.

Here is an example of one way to plot the min/max range using asymmetrical error bars.

Plotting with matplotlib table is now supported in DataFrame.plot() and Series.plot() with a table keyword. The table keyword can accept bool, DataFrame or Series. The simple way to draw a table is to specify table=True. Data will be transposed to meet matplotlib’s default layout.

Also, you can pass a different DataFrame or Series to the table keyword. The data will be drawn as displayed in print method (not transposed automatically). If required, it should be transposed manually as seen in the example below.

There also exists a helper function pandas.plotting.table, which creates a table from DataFrame or Series, and adds it to an matplotlib.Axes instance. This function can accept keywords which the matplotlib table has.

Note: You can get table instances on the axes using axes.tables property for further decorations. See the matplotlib table documentation for more.

A potential issue when plotting a large number of columns is that it can be difficult to distinguish some series due to repetition in the default colors. To remedy this, DataFrame plotting supports the use of the colormap argument, which accepts either a Matplotlib colormap or a string that is a name of a colormap registered with Matplotlib. A visualization of the default matplotlib colormaps is available here.

As matplotlib does not directly support colormaps for line-based plots, the colors are selected based on an even spacing determined by the number of columns in the DataFrame. There is no consideration made for background color, so some colormaps will produce lines that are not easily visible.

To use the cubehelix colormap, we can pass colormap='cubehelix'.

Alternatively, we can pass the colormap itself:

Colormaps can also be used other plot types, like bar charts:

Parallel coordinates charts:

Andrews curves charts:

In some situations it may still be preferable or necessary to prepare plots directly with matplotlib, for instance when a certain type of plot or customization is not (yet) supported by pandas. Series and DataFrame objects behave like arrays and can therefore be passed directly to matplotlib functions without explicit casts.

pandas also automatically registers formatters and locators that recognize date indices, thereby extending date and time support to practically all plot types available in matplotlib. Although this formatting does not provide the same level of refinement you would get when plotting via pandas, it can be faster when plotting a large number of points.

pandas can be extended with third-party plotting backends. The main idea is letting users select a plotting backend different than the provided one based on Matplotlib.

This can be done by passing ‘backend.module’ as the argument backend in plot function. For example:

Alternatively, you can also set this option globally, do you don’t need to specify the keyword in each plot call. For example:

This would be more or less equivalent to:

The backend module can then use other visualization tools (Bokeh, Altair, hvplot,…) to generate the plots. Some libraries implementing a backend for pandas are listed on the ecosystem page.

Developers guide can be found at https://pandas.pydata.org/docs/dev/development/extending.html#plotting-backends

**Examples:**

Example 1 (typescript):
```typescript
In [1]: import matplotlib.pyplot as plt

In [2]: plt.close("all")
```

Example 2 (typescript):
```typescript
In [3]: np.random.seed(123456)

In [4]: ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))

In [5]: ts = ts.cumsum()

In [6]: ts.plot();
```

Example 3 (typescript):
```typescript
In [7]: df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list("ABCD"))

In [8]: df = df.cumsum()

In [9]: plt.figure();

In [10]: df.plot();
```

Example 4 (typescript):
```typescript
In [11]: df3 = pd.DataFrame(np.random.randn(1000, 2), columns=["B", "C"]).cumsum()

In [12]: df3["A"] = pd.Series(list(range(len(df))))

In [13]: df3.plot(x="A", y="B");
```

---

## 

**URL:** https://pandas.pydata.org/docs/_sources/user_guide/10min.rst.txt

---

## 10 minutes to pandas#

**URL:** https://pandas.pydata.org/docs/user_guide/10min.html

**Contents:**
- 10 minutes to pandas#
- Basic data structures in pandas#
- Object creation#
- Viewing data#
- Selection#
  - Getitem ([])#
  - Selection by label#
  - Selection by position#
  - Boolean indexing#
  - Setting#

This is a short introduction to pandas, geared mainly for new users. You can see more complex recipes in the Cookbook.

Customarily, we import as follows:

Pandas provides two types of classes for handling data:

such as integers, strings, Python objects etc.

DataFrame: a two-dimensional data structure that holds data like a two-dimension array or a table with rows and columns.

See the Intro to data structures section.

Creating a Series by passing a list of values, letting pandas create a default RangeIndex.

Creating a DataFrame by passing a NumPy array with a datetime index using date_range() and labeled columns:

Creating a DataFrame by passing a dictionary of objects where the keys are the column labels and the values are the column values.

The columns of the resulting DataFrame have different dtypes:

If you’re using IPython, tab completion for column names (as well as public attributes) is automatically enabled. Here’s a subset of the attributes that will be completed:

As you can see, the columns A, B, C, and D are automatically tab completed. E and F are there as well; the rest of the attributes have been truncated for brevity.

See the Essentially basics functionality section.

Use DataFrame.head() and DataFrame.tail() to view the top and bottom rows of the frame respectively:

Display the DataFrame.index or DataFrame.columns:

Return a NumPy representation of the underlying data with DataFrame.to_numpy() without the index or column labels:

NumPy arrays have one dtype for the entire array while pandas DataFrames have one dtype per column. When you call DataFrame.to_numpy(), pandas will find the NumPy dtype that can hold all of the dtypes in the DataFrame. If the common data type is object, DataFrame.to_numpy() will require copying data.

describe() shows a quick statistic summary of your data:

Transposing your data:

DataFrame.sort_index() sorts by an axis:

DataFrame.sort_values() sorts by values:

While standard Python / NumPy expressions for selecting and setting are intuitive and come in handy for interactive work, for production code, we recommend the optimized pandas data access methods, DataFrame.at(), DataFrame.iat(), DataFrame.loc() and DataFrame.iloc().

See the indexing documentation Indexing and Selecting Data and MultiIndex / Advanced Indexing.

For a DataFrame, passing a single label selects a columns and yields a Series equivalent to df.A:

For a DataFrame, passing a slice : selects matching rows:

See more in Selection by Label using DataFrame.loc() or DataFrame.at().

Selecting a row matching a label:

Selecting all rows (:) with a select column labels:

For label slicing, both endpoints are included:

Selecting a single row and column label returns a scalar:

For getting fast access to a scalar (equivalent to the prior method):

See more in Selection by Position using DataFrame.iloc() or DataFrame.iat().

Select via the position of the passed integers:

Integer slices acts similar to NumPy/Python:

Lists of integer position locations:

For slicing rows explicitly:

For slicing columns explicitly:

For getting a value explicitly:

For getting fast access to a scalar (equivalent to the prior method):

Select rows where df.A is greater than 0.

Selecting values from a DataFrame where a boolean condition is met:

Using isin() method for filtering:

Setting a new column automatically aligns the data by the indexes:

Setting values by label:

Setting values by position:

Setting by assigning with a NumPy array:

The result of the prior setting operations:

A where operation with setting:

For NumPy data types, np.nan represents missing data. It is by default not included in computations. See the Missing Data section.

Reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of the data:

DataFrame.dropna() drops any rows that have missing data:

DataFrame.fillna() fills missing data:

isna() gets the boolean mask where values are nan:

See the Basic section on Binary Ops.

Operations in general exclude missing data.

Calculate the mean value for each column:

Calculate the mean value for each row:

Operating with another Series or DataFrame with a different index or column will align the result with the union of the index or column labels. In addition, pandas automatically broadcasts along the specified dimension and will fill unaligned labels with np.nan.

DataFrame.agg() and DataFrame.transform() applies a user defined function that reduces or broadcasts its result respectively.

See more at Histogramming and Discretization.

Series is equipped with a set of string processing methods in the str attribute that make it easy to operate on each element of the array, as in the code snippet below. See more at Vectorized String Methods.

pandas provides various facilities for easily combining together Series and DataFrame objects with various kinds of set logic for the indexes and relational algebra functionality in the case of join / merge-type operations.

See the Merging section.

Concatenating pandas objects together row-wise with concat():

Adding a column to a DataFrame is relatively fast. However, adding a row requires a copy, and may be expensive. We recommend passing a pre-built list of records to the DataFrame constructor instead of building a DataFrame by iteratively appending records to it.

merge() enables SQL style join types along specific columns. See the Database style joining section.

merge() on unique keys:

By “group by” we are referring to a process involving one or more of the following steps:

Splitting the data into groups based on some criteria

Applying a function to each group independently

Combining the results into a data structure

See the Grouping section.

Grouping by a column label, selecting column labels, and then applying the DataFrameGroupBy.sum() function to the resulting groups:

Grouping by multiple columns label forms MultiIndex.

See the sections on Hierarchical Indexing and Reshaping.

The stack() method “compresses” a level in the DataFrame’s columns:

With a “stacked” DataFrame or Series (having a MultiIndex as the index), the inverse operation of stack() is unstack(), which by default unstacks the last level:

See the section on Pivot Tables.

pivot_table() pivots a DataFrame specifying the values, index and columns

pandas has simple, powerful, and efficient functionality for performing resampling operations during frequency conversion (e.g., converting secondly data into 5-minutely data). This is extremely common in, but not limited to, financial applications. See the Time Series section.

Series.tz_localize() localizes a time series to a time zone:

Series.tz_convert() converts a timezones aware time series to another time zone:

Adding a non-fixed duration (BusinessDay) to a time series:

pandas can include categorical data in a DataFrame. For full docs, see the categorical introduction and the API documentation.

Converting the raw grades to a categorical data type:

Rename the categories to more meaningful names:

Reorder the categories and simultaneously add the missing categories (methods under Series.cat() return a new Series by default):

Sorting is per order in the categories, not lexical order:

Grouping by a categorical column with observed=False also shows empty categories:

See the Plotting docs.

We use the standard convention for referencing the matplotlib API:

The plt.close method is used to close a figure window:

When using Jupyter, the plot will appear using plot(). Otherwise use matplotlib.pyplot.show to show it or matplotlib.pyplot.savefig to write it to a file.

plot() plots all columns:

See the IO Tools section.

Writing to a csv file: using DataFrame.to_csv()

Reading from a csv file: using read_csv()

Writing to a Parquet file:

Reading from a Parquet file Store using read_parquet():

Reading and writing to Excel.

Writing to an excel file using DataFrame.to_excel():

Reading from an excel file using read_excel():

If you are attempting to perform a boolean operation on a Series or DataFrame you might see an exception like:

See Comparisons and Gotchas for an explanation and what to do.

**Examples:**

Example 1 (typescript):
```typescript
In [1]: import numpy as np

In [2]: import pandas as pd
```

Example 2 (typescript):
```typescript
In [3]: s = pd.Series([1, 3, 5, np.nan, 6, 8])

In [4]: s
Out[4]: 
0    1.0
1    3.0
2    5.0
3    NaN
4    6.0
5    8.0
dtype: float64
```

Example 3 (typescript):
```typescript
In [5]: dates = pd.date_range("20130101", periods=6)

In [6]: dates
Out[6]: 
DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype='datetime64[ns]', freq='D')

In [7]: df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))

In [8]: df
Out[8]: 
                   A         B         C         D
2013-01-01  0.469112 -0.282863 -1.509059 -1.135632
2013-01-02  1.212112 -0.173215  0.119209 -1.044236
2013-01-03 -0.861849 -2.104569 -0.494929  1.071804
2013-01-04  0.721555 -0.706771 -1.039575  0.271860
2013-01-05 -0.424972  0.567020  0.276232 -1.087401
2013-01-06 -0.673690  0.113648 -1.478427  0.524988
```

Example 4 (json):
```json
In [9]: df2 = pd.DataFrame(
   ...:     {
   ...:         "A": 1.0,
   ...:         "B": pd.Timestamp("20130102"),
   ...:         "C": pd.Series(1, index=list(range(4)), dtype="float32"),
   ...:         "D": np.array([3] * 4, dtype="int32"),
   ...:         "E": pd.Categorical(["test", "train", "test", "train"]),
   ...:         "F": "foo",
   ...:     }
   ...: )
   ...: 

In [10]: df2
Out[10]: 
     A          B    C  D      E    F
0  1.0 2013-01-02  1.0  3   test  foo
1  1.0 2013-01-02  1.0  3  train  foo
2  1.0 2013-01-02  1.0  3   test  foo
3  1.0 2013-01-02  1.0  3  train  foo
```

---

## Working with missing data#

**URL:** https://pandas.pydata.org/docs/user_guide/missing_data.html

**Contents:**
- Working with missing data#
- Values considered “missing”#
- NA semantics#
  - Propagation in arithmetic and comparison operations#
  - Logical operations#
  - NA in a boolean context#
  - NumPy ufuncs#
    - Conversion#
- Inserting missing data#
- Calculations with missing data#

pandas uses different sentinel values to represent a missing (also referred to as NA) depending on the data type.

numpy.nan for NumPy data types. The disadvantage of using NumPy data types is that the original data type will be coerced to np.float64 or object.

NaT for NumPy np.datetime64, np.timedelta64, and PeriodDtype. For typing applications, use api.types.NaTType.

NA for StringDtype, Int64Dtype (and other bit widths), Float64Dtype`(and other bit widths), :class:`BooleanDtype and ArrowDtype. These types will maintain the original data type of the data. For typing applications, use api.types.NAType.

To detect these missing value, use the isna() or notna() methods.

isna() or notna() will also consider None a missing value.

Equality compaisons between np.nan, NaT, and NA do not act like None

Therefore, an equality comparison between a DataFrame or Series with one of these missing values does not provide the same information as isna() or notna().

Experimental: the behaviour of NA` can still change without warning.

Starting from pandas 1.0, an experimental NA value (singleton) is available to represent scalar missing values. The goal of NA is provide a “missing” indicator that can be used consistently across data types (instead of np.nan, None or pd.NaT depending on the data type).

For example, when having missing values in a Series with the nullable integer dtype, it will use NA:

Currently, pandas does not yet use those data types using NA by default a DataFrame or Series, so you need to specify the dtype explicitly. An easy way to convert to those dtypes is explained in the conversion section.

In general, missing values propagate in operations involving NA. When one of the operands is unknown, the outcome of the operation is also unknown.

For example, NA propagates in arithmetic operations, similarly to np.nan:

There are a few special cases when the result is known, even when one of the operands is NA.

In equality and comparison operations, NA also propagates. This deviates from the behaviour of np.nan, where comparisons with np.nan always return False.

To check if a value is equal to NA, use isna()

An exception on this basic propagation rule are reductions (such as the mean or the minimum), where pandas defaults to skipping missing values. See the calculation section for more.

For logical operations, NA follows the rules of the three-valued logic (or Kleene logic, similarly to R, SQL and Julia). This logic means to only propagate missing values when it is logically required.

For example, for the logical “or” operation (|), if one of the operands is True, we already know the result will be True, regardless of the other value (so regardless the missing value would be True or False). In this case, NA does not propagate:

On the other hand, if one of the operands is False, the result depends on the value of the other operand. Therefore, in this case NA propagates:

The behaviour of the logical “and” operation (&) can be derived using similar logic (where now NA will not propagate if one of the operands is already False):

Since the actual value of an NA is unknown, it is ambiguous to convert NA to a boolean value.

This also means that NA cannot be used in a context where it is evaluated to a boolean, such as if condition: ... where condition can potentially be NA. In such cases, isna() can be used to check for NA or condition being NA can be avoided, for example by filling missing values beforehand.

A similar situation occurs when using Series or DataFrame objects in if statements, see Using if/truth statements with pandas.

pandas.NA implements NumPy’s __array_ufunc__ protocol. Most ufuncs work with NA, and generally return NA:

Currently, ufuncs involving an ndarray and NA will return an object-dtype filled with NA values.

The return type here may change to return a different array type in the future.

See DataFrame interoperability with NumPy functions for more on ufuncs.

If you have a DataFrame or Series using np.nan, Series.convert_dtypes() and DataFrame.convert_dtypes() in DataFrame that can convert data to use the data types that use NA such as Int64Dtype or ArrowDtype. This is especially helpful after reading in data sets from IO methods where data types were inferred.

In this example, while the dtypes of all columns are changed, we show the results for the first 10 columns.

You can insert missing values by simply assigning to a Series or DataFrame. The missing value sentinel used will be chosen based on the dtype.

For object types, pandas will use the value given:

Missing values propagate through arithmetic operations between pandas objects.

The descriptive statistics and computational methods discussed in the data structure overview (and listed here and here) are all account for missing data.

When summing data, NA values or empty data will be treated as zero.

When taking the product, NA values or empty data will be treated as 1.

Cumulative methods like cumsum() and cumprod() ignore NA values by default preserve them in the result. This behavior can be changed with skipna

Cumulative methods like cumsum() and cumprod() ignore NA values by default, but preserve them in the resulting arrays. To override this behaviour and include NA values, use skipna=False.

dropna() dropa rows or columns with missing data.

fillna() replaces NA values with non-NA data.

Replace NA with a scalar value

Fill gaps forward or backward

Limit the number of NA values filled

NA values can be replaced with corresponding value from a Series or DataFrame where the index and column aligns between the original object and the filled object.

DataFrame.where() can also be used to fill NA values.Same result as above.

DataFrame.interpolate() and Series.interpolate() fills NA values using various interpolation methods.

Interpolation relative to a Timestamp in the DatetimeIndex is available by setting method="time"

For a floating-point index, use method='values':

If you have scipy installed, you can pass the name of a 1-d interpolation routine to method. as specified in the scipy interpolation documentation and reference guide. The appropriate interpolation method will depend on the data type.

If you are dealing with a time series that is growing at an increasing rate, use method='barycentric'.

If you have values approximating a cumulative distribution function, use method='pchip'.

To fill missing values with goal of smooth plotting use method='akima'.

When interpolating via a polynomial or spline approximation, you must also specify the degree or order of the approximation:

Comparing several methods.

Interpolating new observations from expanding data with Series.reindex().

interpolate() accepts a limit keyword argument to limit the number of consecutive NaN values filled since the last valid observation

By default, NaN values are filled in a forward direction. Use limit_direction parameter to fill backward or from both directions.

By default, NaN values are filled whether they are surrounded by existing valid values or outside existing valid values. The limit_area parameter restricts filling to either inside or outside values.

Series.replace() and DataFrame.replace() can be used similar to Series.fillna() and DataFrame.fillna() to replace or insert missing values.

Replacing more than one value is possible by passing a list.

Replacing using a mapping dict.

Python strings prefixed with the r character such as r'hello world' are “raw” strings. They have different semantics regarding backslashes than strings without this prefix. Backslashes in raw strings will be interpreted as an escaped backslash, e.g., r'\' == '\\'.

Replace the ‘.’ with NaN

Replace the ‘.’ with NaN with regular expression that removes surrounding whitespace

Replace with a list of regexes.

Replace with a regex in a mapping dict.

Pass nested dictionaries of regular expressions that use the regex keyword.

Pass a list of regular expressions that will replace matches with a scalar.

All of the regular expression examples can also be passed with the to_replace argument as the regex argument. In this case the value argument must be passed explicitly by name or regex must be a nested dictionary.

A regular expression object from re.compile is a valid input as well.

**Examples:**

Example 1 (yaml):
```yaml
In [1]: pd.Series([1, 2], dtype=np.int64).reindex([0, 1, 2])
Out[1]: 
0    1.0
1    2.0
2    NaN
dtype: float64

In [2]: pd.Series([True, False], dtype=np.bool_).reindex([0, 1, 2])
Out[2]: 
0     True
1    False
2      NaN
dtype: object
```

Example 2 (yaml):
```yaml
In [3]: pd.Series([1, 2], dtype=np.dtype("timedelta64[ns]")).reindex([0, 1, 2])
Out[3]: 
0   0 days 00:00:00.000000001
1   0 days 00:00:00.000000002
2                         NaT
dtype: timedelta64[ns]

In [4]: pd.Series([1, 2], dtype=np.dtype("datetime64[ns]")).reindex([0, 1, 2])
Out[4]: 
0   1970-01-01 00:00:00.000000001
1   1970-01-01 00:00:00.000000002
2                             NaT
dtype: datetime64[ns]

In [5]: pd.Series(["2020", "2020"], dtype=pd.PeriodDtype("D")).reindex([0, 1, 2])
Out[5]: 
0    2020-01-01
1    2020-01-01
2           NaT
dtype: period[D]
```

Example 3 (yaml):
```yaml
In [6]: pd.Series([1, 2], dtype="Int64").reindex([0, 1, 2])
Out[6]: 
0       1
1       2
2    <NA>
dtype: Int64

In [7]: pd.Series([True, False], dtype="boolean[pyarrow]").reindex([0, 1, 2])
Out[7]: 
0     True
1    False
2     <NA>
dtype: bool[pyarrow]
```

Example 4 (typescript):
```typescript
In [8]: ser = pd.Series([pd.Timestamp("2020-01-01"), pd.NaT])

In [9]: ser
Out[9]: 
0   2020-01-01
1          NaT
dtype: datetime64[ns]

In [10]: pd.isna(ser)
Out[10]: 
0    False
1     True
dtype: bool
```

---

## Windowing operations#

**URL:** https://pandas.pydata.org/docs/user_guide/window.html

**Contents:**
- Windowing operations#
- Overview#
- Rolling window#
  - Centering windows#
  - Rolling window endpoints#
  - Custom window rolling#
  - Rolling apply#
  - Numba engine#
  - Binary window functions#
  - Computing rolling pairwise covariances and correlations#

pandas contains a compact set of APIs for performing windowing operations - an operation that performs an aggregation over a sliding partition of values. The API functions similarly to the groupby API in that Series and DataFrame call the windowing method with necessary parameters and then subsequently call the aggregation function.

The windows are comprised by looking back the length of the window from the current observation. The result above can be derived by taking the sum of the following windowed partitions of data:

pandas supports 4 types of windowing operations:

Rolling window: Generic fixed or variable sliding window over the values.

Weighted window: Weighted, non-rectangular window supplied by the scipy.signal library.

Expanding window: Accumulating window over the values.

Exponentially Weighted window: Accumulating and exponentially weighted window over the values.

Supports time-based windows

Supports chained groupby

Supports table method

Supports online operations

pandas.typing.api.Rolling

Yes (as of version 1.3)

pandas.typing.api.Window

pandas.typing.api.Expanding

Yes (as of version 1.3)

Exponentially Weighted window

pandas.typing.api.ExponentialMovingWindow

Yes (as of version 1.2)

Yes (as of version 1.3)

As noted above, some operations support specifying a window based on a time offset:

Additionally, some methods support chaining a groupby operation with a windowing operation which will first group the data by the specified keys and then perform a windowing operation per group.

Windowing operations currently only support numeric data (integer and float) and will always return float64 values.

Some windowing aggregation, mean, sum, var and std methods may suffer from numerical imprecision due to the underlying windowing algorithms accumulating sums. When values differ with magnitude \(1/np.finfo(np.double).eps\) this results in truncation. It must be noted, that large values may have an impact on windows, which do not include these values. Kahan summation is used to compute the rolling sums to preserve accuracy as much as possible.

Added in version 1.3.0.

Some windowing operations also support the method='table' option in the constructor which performs the windowing operation over an entire DataFrame instead of a single column or row at a time. This can provide a useful performance benefit for a DataFrame with many columns or rows (with the corresponding axis argument) or the ability to utilize other columns during the windowing operation. The method='table' option can only be used if engine='numba' is specified in the corresponding method call.

For example, a weighted mean calculation can be calculated with apply() by specifying a separate column of weights.

Added in version 1.3.

Some windowing operations also support an online method after constructing a windowing object which returns a new object that supports passing in new DataFrame or Series objects to continue the windowing calculation with the new values (i.e. online calculations).

The methods on this new windowing objects must call the aggregation method first to “prime” the initial state of the online calculation. Then, new DataFrame or Series objects can be passed in the update argument to continue the windowing calculation.

All windowing operations support a min_periods argument that dictates the minimum amount of non-np.nan values a window must have; otherwise, the resulting value is np.nan. min_periods defaults to 1 for time-based windows and window for fixed windows

Additionally, all windowing operations supports the aggregate method for returning a result of multiple aggregations applied to a window.

Generic rolling windows support specifying windows as a fixed number of observations or variable number of observations based on an offset. If a time based offset is provided, the corresponding time based index must be monotonic.

For all supported aggregation functions, see Rolling window functions.

By default the labels are set to the right edge of the window, but a center keyword is available so the labels can be set at the center.

This can also be applied to datetime-like indices.

Added in version 1.3.0.

The inclusion of the interval endpoints in rolling window calculations can be specified with the closed parameter:

For example, having the right endpoint open is useful in many problems that require that there is no contamination from present information back to past information. This allows the rolling window to compute statistics “up to that point in time”, but not including that point in time.

In addition to accepting an integer or offset as a window argument, rolling also accepts a BaseIndexer subclass that allows a user to define a custom method for calculating window bounds. The BaseIndexer subclass will need to define a get_window_bounds method that returns a tuple of two arrays, the first being the starting indices of the windows and second being the ending indices of the windows. Additionally, num_values, min_periods, center, closed and step will automatically be passed to get_window_bounds and the defined method must always accept these arguments.

For example, if we have the following DataFrame

and we want to use an expanding window where use_expanding is True otherwise a window of size 1, we can create the following BaseIndexer subclass:

You can view other examples of BaseIndexer subclasses here

One subclass of note within those examples is the VariableOffsetWindowIndexer that allows rolling operations over a non-fixed offset like a BusinessDay.

For some problems knowledge of the future is available for analysis. For example, this occurs when each data point is a full time series read from an experiment, and the task is to extract underlying conditions. In these cases it can be useful to perform forward-looking rolling window computations. FixedForwardWindowIndexer class is available for this purpose. This BaseIndexer subclass implements a closed fixed-width forward-looking rolling window, and we can use it as follows:

We can also achieve this by using slicing, applying rolling aggregation, and then flipping the result as shown in example below:

The apply() function takes an extra func argument and performs generic rolling computations. The func argument should be a single function that produces a single value from an ndarray input. raw specifies whether the windows are cast as Series objects (raw=False) or ndarray objects (raw=True).

Additionally, apply() can leverage Numba if installed as an optional dependency. The apply aggregation can be executed using Numba by specifying engine='numba' and engine_kwargs arguments (raw must also be set to True). See enhancing performance with Numba for general usage of the arguments and performance considerations.

Numba will be applied in potentially two routines:

If func is a standard Python function, the engine will JIT the passed function. func can also be a JITed function in which case the engine will not JIT the function again.

The engine will JIT the for loop where the apply function is applied to each window.

The engine_kwargs argument is a dictionary of keyword arguments that will be passed into the numba.jit decorator. These keyword arguments will be applied to both the passed function (if a standard Python function) and the apply for loop over each window.

Added in version 1.3.0.

mean, median, max, min, and sum also support the engine and engine_kwargs arguments.

cov() and corr() can compute moving window statistics about two Series or any combination of DataFrame/Series or DataFrame/DataFrame. Here is the behavior in each case:

two Series: compute the statistic for the pairing.

DataFrame/Series: compute the statistics for each column of the DataFrame with the passed Series, thus returning a DataFrame.

DataFrame/DataFrame: by default compute the statistic for matching column names, returning a DataFrame. If the keyword argument pairwise=True is passed then computes the statistic for each pair of columns, returning a DataFrame with a MultiIndex whose values are the dates in question (see the next section).

In financial data analysis and other fields it’s common to compute covariance and correlation matrices for a collection of time series. Often one is also interested in moving-window covariance and correlation matrices. This can be done by passing the pairwise keyword argument, which in the case of DataFrame inputs will yield a MultiIndexed DataFrame whose index are the dates in question. In the case of a single DataFrame argument the pairwise argument can even be omitted:

Missing values are ignored and each entry is computed using the pairwise complete observations.

Assuming the missing data are missing at random this results in an estimate for the covariance matrix which is unbiased. However, for many applications this estimate may not be acceptable because the estimated covariance matrix is not guaranteed to be positive semi-definite. This could lead to estimated correlations having absolute values which are greater than one, and/or a non-invertible covariance matrix. See Estimation of covariance matrices for more details.

The win_type argument in .rolling generates a weighted windows that are commonly used in filtering and spectral estimation. win_type must be string that corresponds to a scipy.signal window function. Scipy must be installed in order to use these windows, and supplementary arguments that the Scipy window methods take must be specified in the aggregation function.

For all supported aggregation functions, see Weighted window functions.

An expanding window yields the value of an aggregation statistic with all the data available up to that point in time. Since these calculations are a special case of rolling statistics, they are implemented in pandas such that the following two calls are equivalent:

For all supported aggregation functions, see Expanding window functions.

An exponentially weighted window is similar to an expanding window but with each prior point being exponentially weighted down relative to the current point.

In general, a weighted moving average is calculated as

where \(x_t\) is the input, \(y_t\) is the result and the \(w_i\) are the weights.

For all supported aggregation functions, see Exponentially-weighted window functions.

The EW functions support two variants of exponential weights. The default, adjust=True, uses the weights \(w_i = (1 - \alpha)^i\) which gives

When adjust=False is specified, moving averages are calculated as

which is equivalent to using weights

These equations are sometimes written in terms of \(\alpha' = 1 - \alpha\), e.g.

The difference between the above two variants arises because we are dealing with series which have finite history. Consider a series of infinite history, with adjust=True:

Noting that the denominator is a geometric series with initial term equal to 1 and a ratio of \(1 - \alpha\) we have

which is the same expression as adjust=False above and therefore shows the equivalence of the two variants for infinite series. When adjust=False, we have \(y_0 = x_0\) and \(y_t = \alpha x_t + (1 - \alpha) y_{t-1}\). Therefore, there is an assumption that \(x_0\) is not an ordinary value but rather an exponentially weighted moment of the infinite series up to that point.

One must have \(0 < \alpha \leq 1\), and while it is possible to pass \(\alpha\) directly, it’s often easier to think about either the span, center of mass (com) or half-life of an EW moment:

One must specify precisely one of span, center of mass, half-life and alpha to the EW functions:

Span corresponds to what is commonly called an “N-day EW moving average”.

Center of mass has a more physical interpretation and can be thought of in terms of span: \(c = (s - 1) / 2\).

Half-life is the period of time for the exponential weight to reduce to one half.

Alpha specifies the smoothing factor directly.

You can also specify halflife in terms of a timedelta convertible unit to specify the amount of time it takes for an observation to decay to half its value when also specifying a sequence of times.

The following formula is used to compute exponentially weighted mean with an input vector of times:

ExponentialMovingWindow also has an ignore_na argument, which determines how intermediate null values affect the calculation of the weights. When ignore_na=False (the default), weights are calculated based on absolute positions, so that intermediate null values affect the result. When ignore_na=True, weights are calculated by ignoring intermediate null values. For example, assuming adjust=True, if ignore_na=False, the weighted average of 3, NaN, 5 would be calculated as

Whereas if ignore_na=True, the weighted average would be calculated as

The var(), std(), and cov() functions have a bias argument, specifying whether the result should contain biased or unbiased statistics. For example, if bias=True, ewmvar(x) is calculated as ewmvar(x) = ewma(x**2) - ewma(x)**2; whereas if bias=False (the default), the biased variance statistics are scaled by debiasing factors

(For \(w_i = 1\), this reduces to the usual \(N / (N - 1)\) factor, with \(N = t + 1\).) See Weighted Sample Variance on Wikipedia for further details.

**Examples:**

Example 1 (typescript):
```typescript
In [1]: s = pd.Series(range(5))

In [2]: s.rolling(window=2).sum()
Out[2]: 
0    NaN
1    1.0
2    3.0
3    5.0
4    7.0
dtype: float64
```

Example 2 (yaml):
```yaml
In [3]: for window in s.rolling(window=2):
   ...:     print(window)
   ...: 
0    0
dtype: int64
0    0
1    1
dtype: int64
1    1
2    2
dtype: int64
2    2
3    3
dtype: int64
3    3
4    4
dtype: int64
```

Example 3 (typescript):
```typescript
In [4]: s = pd.Series(range(5), index=pd.date_range('2020-01-01', periods=5, freq='1D'))

In [5]: s.rolling(window='2D').sum()
Out[5]: 
2020-01-01    0.0
2020-01-02    1.0
2020-01-03    3.0
2020-01-04    5.0
2020-01-05    7.0
Freq: D, dtype: float64
```

Example 4 (typescript):
```typescript
In [6]: df = pd.DataFrame({'A': ['a', 'b', 'a', 'b', 'a'], 'B': range(5)})

In [7]: df.groupby('A').expanding().sum()
Out[7]: 
       B
A       
a 0  0.0
  2  2.0
  4  6.0
b 1  1.0
  3  4.0
```

---

## 

**URL:** https://pandas.pydata.org/docs/_sources/user_guide/io.rst.txt

---


# Linear Regression with Auto MPG

For the following exercise, we will use the well know Auto MPG dataset, which you can read about [here](https://archive.ics.uci.edu/ml/datasets/Auto+MPG).

The task for this exercise will be to build two models, using Sklearn and Statsmodels, to predict miles per gallon for each car record.  To do so, we have a set of predictive features.  The list, `column_names`, contains the names of both the dependent and independent variables.  


The dataset has been loaded into a dataframe for you in the cell below.

Now, using the `columns` attribute of `df`, add column names to the dataframe.  


```python
df.columns = column_names

```

# Data Prep

As always, we need to check for missing values.


```python
df.isna().sum()
```

Let's also inspec the column datatypes.


```python
df.info()
```

Oddly enough, the `horsepower` column is encoded as a string.   Let's convert the `horsepower` column to `float`. 

* Hint: your first attempt to convert the column may through an error. The last line of the error message should indicate the value that is gumming up the works.  Use df.replace(), and replace value with np.nan, then try to change the dtype once more*



```python
df['horsepower'] = df['horsepower'].replace({'?':np.nan}).astype(float)
```

Now we have some NA values. Drop the records with NA's in the `horsepower` column.


```python
df.dropna(subset=['horsepower'], inplace=True)
```

The goal of this exercise is to become familiar with using our regression tools. Before doing so, we will pause for the briefest of EDA, and run a pairplot in the cell below (word of caution: EDA is always important. A pairplot is a first step. It does not represent a complete EDA process).

There is much you can gather from the pairplot above, but for now, just notice that the plots for cylinders, model year, and origin have a different type of pattern than the rest. Looking at the first row of the pairplot, we see that the x-values of those three columns correspond to discrete values on the X-axis, resulting in horizontal lines.  These descrete outcomes are possible candidates for one hot encoding or binarizing.

Two other important takeaways from the plot are: collinearity between features (evident in points grouped along the diagonal); and curvature (which might suggest a polynomial transformation could be beneficial).  We will leave that aside for now, and practice model fitting.

# Model building with sklearn

Use the mask below to isolate the 4 continuous features and the target from our `df`.  


```python
df_continuous = df[continuous_mask]
df_continuous.shape

```

Split the target off from the dataset, and assign it the variable `y` below. 


```python
y = df['mpg']
```

Drop the target from df_continous, and assign the resulting dataframe to the variable `X` below.


```python
X = df_continuous.drop('mpg', axis = 1)
```

The data is now ready to be fed into sklearn's LinearRegression class, which is imported for you in the next cell.

To build the model, create an instance of the LinearRegression class: assign `LinearRegression()` to the variable `lr` below.


```python
lr = LinearRegression()
```

Next, pass our `X` and `y` variables, in that order, as arguments into the fit() method, called off the end of our `lr` variable.


```python
lr.fit(X,y)
```

Now that the model has been fit, the `lr` variable has been filled with information learned from the data. Look at the `.coef_` attribute, which describes the calculated betas associated with each independent variable.


The coefficient associated with horsepower is roughly -.0438.

#### Interepret the meaning of that coefficient. How does a 1-Unit increase in horsepower affect mpg?

> Your written answer here.


```python
"""
A unit increase in horsepower results in reduction -.0438 mpg for any given car.
"""
```

Lastly, feed in `X` and `y` to the `score` method chained off the end of our `lr` variable. That method gives us an R^2, which we will compare to the Statsmodel output.


```python
r_2 = lr.score(X,y)
```

# Statsmodels

Let's now compare Statsmodel's output.  
**Spoiler Alert: it will be the same.**

Statsmodels takes a **formula string** as an argument, which looks like what you might expect from the R language.

$target \sim column\_name\_1 \ + column\_name\_2 + \ ...\ + column\_name\_n$

To do so, we can join the list of columns of X with a `+`


```python
columns = '+'.join(X.columns)
```

Using the string of column names joined in the cell above, we can now construct the formula by running the cells below.

Lastly, pass `formula` and the original `df` into `ols()` as parameters.  

Then, chain the methods .fit() and .summary().  

Note: You need to feed in the original `df`, because `ols` requires the target to be present in the data parameter that you pass.


```python
ols(formula,df).fit().summary()
```

The `summary` gives a lot of good information, but for now, just confirm that the R_squared is the same (rounded) as the sklearn R_squared found nearer the top of the notebook.

Also, look at the coef column and confirm that the beta coefficients are consistent across both sklearn and statsmodels.

If they are the same, pat yourself on the back.

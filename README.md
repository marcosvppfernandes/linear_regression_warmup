
# Linear Regression with Auto MPG

For the following exercise, we will use the well know Auto MPG dataset, which you can read about [here](https://archive.ics.uci.edu/ml/datasets/Auto+MPG).

The task for this exercise will be to build a series of models, using both sklearn and statsmodels, to predict miles per gallon for each car record.  To do so, we have a set of predictive features.  The list, `column_names`, contains the names of both both the dependent and independent variables.  


The dataset has been loaded into a dataframe for you in the cell below.

Now, using the `columns` attribute, add column names to the dataframe.  


```python
df.columns = column_names

```

# Data Prep


```python
df.isna().sum()
```




    mpg             0
    cylinders       0
    displacement    0
    horsepower      0
    weight          0
    acceleration    0
    modelyear       0
    origin          0
    carname         0
    dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 391 entries, 0 to 396
    Data columns (total 9 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   mpg           391 non-null    float64
     1   cylinders     391 non-null    int64  
     2   displacement  391 non-null    float64
     3   horsepower    391 non-null    float64
     4   weight        391 non-null    float64
     5   acceleration  391 non-null    float64
     6   modelyear     391 non-null    int64  
     7   origin        391 non-null    int64  
     8   carname       391 non-null    object 
    dtypes: float64(5), int64(3), object(1)
    memory usage: 30.5+ KB


Oddly enough, the `horsepower` column is encoded as a string.   Let's convert the `horsepower` column to a float. 

* Hint: your first attempt to convert the column may through an error. The last line of the error message should indicate the value messing things up.  Replace that value with np.nan and try again*



```python
df['horsepower'] = df['horsepower'].replace({'?':np.nan}).astype(float)
```

Now we have some NA values. Drop the records with NA's in the `horsepower` column.


```python
df.dropna(subset=['horsepower'], inplace=True)
```

The goal of this exercise is to become familiar with using our regression tools. We will pause for the briefest of EDA, and run a pairplot in the cell below:

There is much you gan gather from the pairplot above, but for now, just notice that the plots for cylinders, model year, and origin have a different pattern than the rest. Looking at the first row of the pairplot, we see that the x-values correspond to discrete values on the X-axis, and resulting in horizontal lines.  These descrete outcomes are possible candidates for One Hot Encoding or turning into binary variables.

Two other important takeaways from the plot are: collinearity between features (evident in points grouped along the diagonal); and curvature (which might suggest a polynomial transformation could be beneficial).  We will leave that aside for now, and practice model fitting.

# Model building with sklearn

Use the mask below to isolate the 4 continuous features and the target.  


```python
df_continuous = df[continuous_mask]
df_continuous.shape

```




    (391, 5)



Split the target off from the dataset, and assign it the variable `y` below. 


```python
y = df['mpg']
```

Drop the target from df_continous, and assign the resulting dataframe to the variable `X` below.


```python
X = df_continous.drop('mpg', axis = 1)
```

The data is now ready to be fed into sklearn's LinearRegression class, which is imported for you in the next cell.

To build the model, instantiate an instance of the LinearRegression class. Assign `LinearRegression()` to the variable `lr` below.


```python
lr = LinearRegression()
```

Next, pass our `X` and `y` variables, in that order, as arguments into the fit() method, called off the end of our `lr` variable.


```python
lr.fit(X,y)
```




    LinearRegression()



Now that the model has been fit, the `lr` variable has been filled with information learned from the data.


The coefficient associated with horsepower is roughly -.0438.

Interepret the meaning of that coefficient. How does a 1-Unit increase in horsepower affect mpg?

> Your written answer here.


```python
"""
A unit increase in horsepower results in reduction -.0438 mpg for any given car.
"""
```




    '\nA unit increase in horsepower results in reduction -.0438 mpg for any given car.\n'



Lastly, feed in `X` and `y` to the `score` method chained off the end of our `lr` variable. That method gives us an R^2, which we will compare to the Statsmodel output.


```python
r_2 = lr.score(X,y)
```

# Statsmodels

Let's now compare Statsmodel's output.  
**Spoiler Alert: it will be the same.**

Statsmodels takes a formula string as an argument, which looks like what you might expect from the R language.

$target \sim column\_name\_1 \ + column\_name\_2 + \ ...\ + column\_name\_n$

To do so, we can join the list of columns of X with a `+`


```python
columns = '+'.join(X.columns)
```

Using the string of column names joined in the cell above, we can now construct the formula by running the cells below.

Lastly, pass the formula and the original `df` into `ols()` as parameters.  Then, chain the methods .fit() and .summary().  You need to feed in the original df, because `ols` requires the target to be present in the data parameter that you pass.

The `summary` gives a lot of good information, but for now, just confirm that the r_squared is the same (rounded) as the sklearn method performed at the top of the notebook.

Also, look at the coef column and confirm that the beta coefficients are consistent across both sklearn and statsmodels.

If they are the same, pat yourself on the back.

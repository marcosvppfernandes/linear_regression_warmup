
# Linear Regression with Auto MPG

For the following exercise, we will use the well know Auto MPG dataset, which you can read about [here](https://archive.ics.uci.edu/ml/datasets/Auto+MPG).

The task for this exercise will be to build a series of models, using both sklearn and statsmodels, to predict miles per gallon for each car record.  To do so, we have a set of predictive features.  The list, `column_names`, contains the names of both the dependent and independent variables.  


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



Let's also inspec the column datatypes.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 397 entries, 0 to 396
    Data columns (total 9 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   mpg           397 non-null    float64
     1   cylinders     397 non-null    int64  
     2   displacement  397 non-null    float64
     3   horsepower    397 non-null    object 
     4   weight        397 non-null    float64
     5   acceleration  397 non-null    float64
     6   modelyear     397 non-null    int64  
     7   origin        397 non-null    int64  
     8   carname       397 non-null    object 
    dtypes: float64(4), int64(3), object(2)
    memory usage: 28.0+ KB


Oddly enough, the `horsepower` column is encoded as a string.   Let's convert the `horsepower` column to `float`. 

* Hint: your first attempt to convert the column may through an error. The last line of the error message should indicate the value that gumming up the works messing.  Use df.replace(), and replace value with np.nan, then try to change the dtype once more*



```python
df['horsepower'] = df['horsepower'].replace({'?':np.nan}).astype(float)
```

Now we have some NA values. Drop the records with NA's in the `horsepower` column.


```python
df.dropna(subset=['horsepower'], inplace=True)
```

The goal of this exercise is to become familiar with using our regression tools. Before doing so, we will pause for the briefest of EDA, and run a pairplot in the cell below (word of caution: EDA is always important. A pairplot is a first step. It does not represent a complete EDA process).

There is much you gan gather from the pairplot above, but for now, just notice that the plots for cylinders, model year, and origin have a different type of pattern than the rest. Looking at the first row of the pairplot, we see that the x-values of those three columns correspond to discrete values on the X-axis, resulting in horizontal lines.  These descrete outcomes are possible candidates for one hot encoding or binarizing.

Two other important takeaways from the plot are: collinearity between features (evident in points grouped along the diagonal); and curvature (which might suggest a polynomial transformation could be beneficial).  We will leave that aside for now, and practice model fitting.

# Model building with sklearn

Use the mask below to isolate the 4 continuous features and the target from our `df`.  


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




    LinearRegression()



Now that the model has been fit, the `lr` variable has been filled with information learned from the data. Look at the `.coef_` attribute, which describes the calculated betas associated with each independent variable.


The coefficient associated with horsepower is roughly -.0438.

#### Interepret the meaning of that coefficient. How does a 1-Unit increase in horsepower affect mpg?

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




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.707</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.704</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   232.5</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 09 Feb 2021</td> <th>  Prob (F-statistic):</th> <td>2.20e-101</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:40:48</td>     <th>  Log-Likelihood:    </th> <td> -1118.2</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   391</td>      <th>  AIC:               </th> <td>   2246.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   386</td>      <th>  BIC:               </th> <td>   2266.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>   45.2912</td> <td>    2.465</td> <td>   18.371</td> <td> 0.000</td> <td>   40.444</td> <td>   50.138</td>
</tr>
<tr>
  <th>displacement</th> <td>   -0.0059</td> <td>    0.007</td> <td>   -0.878</td> <td> 0.381</td> <td>   -0.019</td> <td>    0.007</td>
</tr>
<tr>
  <th>horsepower</th>   <td>   -0.0438</td> <td>    0.017</td> <td>   -2.637</td> <td> 0.009</td> <td>   -0.076</td> <td>   -0.011</td>
</tr>
<tr>
  <th>weight</th>       <td>   -0.0053</td> <td>    0.001</td> <td>   -6.507</td> <td> 0.000</td> <td>   -0.007</td> <td>   -0.004</td>
</tr>
<tr>
  <th>acceleration</th> <td>   -0.0248</td> <td>    0.126</td> <td>   -0.197</td> <td> 0.844</td> <td>   -0.272</td> <td>    0.223</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>38.029</td> <th>  Durbin-Watson:     </th> <td>   0.861</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  50.709</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.713</td> <th>  Prob(JB):          </th> <td>9.74e-12</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.039</td> <th>  Cond. No.          </th> <td>3.56e+04</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 3.56e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



The `summary` gives a lot of good information, but for now, just confirm that the R_squared is the same (rounded) as the sklearn r_squared found nearer the top of the notebook.

Also, look at the coef column and confirm that the beta coefficients are consistent across both sklearn and statsmodels.

If they are the same, pat yourself on the back.

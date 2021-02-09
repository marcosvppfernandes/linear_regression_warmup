
# Linear Regression with Auto MPG

For the following exercise, we will use the well know Auto MPG dataset, which you can read about [here](https://archive.ics.uci.edu/ml/datasets/Auto+MPG).

The task for this exercise will be to build a series of models, using both sklearn and statsmodels, to predict miles per gallon for each car record.  To do so, we have a set of predictive features.  The list, `column_names`, contains the names of both the dependent and independent variables.  



```python
column_names = ["mpg","cylinders","displacement","horsepower",
                "weight","acceleration","modelyear","origin",
                "carname"]
```

The dataset has been loaded into a dataframe for you in the cell below.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv('data/auto-mpg.data', delim_whitespace=' ')
```

Now, using the `columns` attribute of `df`, add column names to the dataframe.  


```python
# Your code here
```


```python
#__SOLUTION__
df.columns = column_names

```


```python
assert df.columns[0]=='mpg'
print("Nice job!")
```

    Nice job!



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>modelyear</th>
      <th>origin</th>
      <th>carname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449.0</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.0</td>
      <td>8</td>
      <td>429.0</td>
      <td>198.0</td>
      <td>4341.0</td>
      <td>10.0</td>
      <td>70</td>
      <td>1</td>
      <td>ford galaxie 500</td>
    </tr>
  </tbody>
</table>
</div>



# Data Prep

As always, we need to check for missing values.


```python
# Code to inspect if there are missing values.
```


```python
#__SOLUTION__
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
# code to inspect the datatypes of the columns

```


```python
#__SOLUTION__
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
# your code here
```


```python
#__SOLUTION__
df['horsepower'] = df['horsepower'].replace({'?':np.nan}).astype(float)
```


```python
assert df['horsepower'].dtype == 'float64'
print('You got it.')
```

    You got it.


Now we have some NA values. Drop the records with NA's in the `horsepower` column.


```python
# Your code here
```


```python
#__SOLUTION__
df.dropna(subset=['horsepower'], inplace=True)
```


```python
assert df['horsepower'].isna().sum() == 0
assert df.shape[0] == 391
print("Dropping those NA's should result in 391 records. Good job.")
```

    Dropping those NA's should result in 391 records. Good job.


The goal of this exercise is to become familiar with using our regression tools. Before doing so, we will pause for the briefest of EDA, and run a pairplot in the cell below (word of caution: EDA is always important. A pairplot is a first step. It does not represent a complete EDA process).


```python
sns.pairplot(df)
```




    <seaborn.axisgrid.PairGrid at 0x11b16c160>




![png](index_files/index_27_1.png)


There is much you gan gather from the pairplot above, but for now, just notice that the plots for cylinders, model year, and origin have a different type of pattern than the rest. Looking at the first row of the pairplot, we see that the x-values of those three columns correspond to discrete values on the X-axis, resulting in horizontal lines.  These descrete outcomes are possible candidates for one hot encoding or binarizing.

Two other important takeaways from the plot are: collinearity between features (evident in points grouped along the diagonal); and curvature (which might suggest a polynomial transformation could be beneficial).  We will leave that aside for now, and practice model fitting.

# Model building with sklearn

Use the mask below to isolate the 4 continuous features and the target from our `df`.  


```python
continuous_mask = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
```


```python
# Replace None with your code
df_continuous = None
```


```python
#__SOLUTION__
df_continuous = df[continuous_mask]
df_continuous.shape

```




    (391, 5)




```python
assert df_continuous.shape[1] == 5
assert list(df_continuous.columns) == continuous_mask
```

Split the target off from the dataset, and assign it the variable `y` below. 


```python
# Replace None with your code
y = None
```


```python
#__SOLUTION__
y = df['mpg']
```


```python
assert y[0] == 15.0
print('Nice work')
```

    Nice work


Drop the target from df_continous, and assign the resulting dataframe to the variable `X` below.


```python
# Replace None with your code
X = None
```


```python
#__SOLUTION__ 
X = df_continuous.drop('mpg', axis = 1)
```


```python
assert X.shape[1] == 4 
```

The data is now ready to be fed into sklearn's LinearRegression class, which is imported for you in the next cell.


```python
from sklearn.linear_model import LinearRegression
```

To build the model, create an instance of the LinearRegression class: assign `LinearRegression()` to the variable `lr` below.


```python
# Replace None with your code
lr = None
```


```python
#__SOLUTION__
lr = LinearRegression()
```

Next, pass our `X` and `y` variables, in that order, as arguments into the fit() method, called off the end of our `lr` variable.


```python
# your code here
```


```python
#__SOLUTION__
lr.fit(X,y)
```




    LinearRegression()




```python
assert np.isclose(lr.coef_[1], -0.04381764059543403 )
print('Noice')
```

    Noice


Now that the model has been fit, the `lr` variable has been filled with information learned from the data. Look at the `.coef_` attribute, which describes the calculated betas associated with each independent variable.



```python
for column_name, coefficient in zip(lr.coef_, X.columns):
    print(column_name, coefficient)
```

    -0.005906430624810988 displacement
    -0.04381764059543403 horsepower
    -0.005283508472229711 weight
    -0.024759046676832288 acceleration


The coefficient associated with horsepower is roughly -.0438.

#### Interepret the meaning of that coefficient. How does a 1-Unit increase in horsepower affect mpg?

> Your written answer here.


```python
#__SOLUTION__
"""
A unit increase in horsepower results in reduction -.0438 mpg for any given car.
"""
```




    '\nA unit increase in horsepower results in reduction -.0438 mpg for any given car.\n'



Lastly, feed in `X` and `y` to the `score` method chained off the end of our `lr` variable. That method gives us an R^2, which we will compare to the Statsmodel output.


```python
# Replace None with your code

r_2 = None
```


```python
#__SOLUTION__
r_2 = lr.score(X,y)
```


```python
assert np.isclose(r_2, 0.70665)
print('Great work!')
```

    Great work!


# Statsmodels

Let's now compare Statsmodel's output.  
**Spoiler Alert: it will be the same.**


```python
from statsmodels.formula.api import ols
```

Statsmodels takes a **formula string** as an argument, which looks like what you might expect from the R language.

$target \sim column\_name\_1 \ + column\_name\_2 + \ ...\ + column\_name\_n$

To do so, we can join the list of columns of X with a `+`


```python
# join the columns by feeding X.columns into "+".join().
columns = None
```


```python
#__SOLUTION__
columns = '+'.join(X.columns)
```

Using the string of column names joined in the cell above, we can now construct the formula by running the cells below.


```python
target = 'mpg'
```


```python
formula = target + '~'+columns
```


```python
assert formula == 'mpg~displacement+horsepower+weight+acceleration'
```

Lastly, pass `formula` and the original `df` into `ols()` as parameters.  

Then, chain the methods .fit() and .summary().  

Note: You need to feed in the original `df`, because `ols` requires the target to be present in the data parameter that you pass.


```python
# Feed formula and df to the line of code below. 
ols().fit().summary()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-106-0d67883d2125> in <module>
          1 # Feed formula and df to the line of code below.
    ----> 2 ols().fit().summary()
    

    TypeError: from_formula() missing 2 required positional arguments: 'formula' and 'data'



```python
#__SOLUTION__
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

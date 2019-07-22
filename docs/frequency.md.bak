
# Frequency

Frequency analysis in sci-analysis is similar to Distribution analysis, but provides summary statistics and a [bar chart](https://en.wikipedia.org/wiki/Bar_chart) of categorical data instead of numeric data. It provides the count, percent, and rank of the occurrence of each category in a given sequence.

## Interpreting the Graphs

The only graph shown by the frequency analysis is a bar chart where each bar is a unique category in the data set. By default the bar chart displays the frequency (counts) of each category in the bar chart, but can be configured to display the percent of each category instead.


```python
import numpy as np
import scipy.stats as st
from sci_analysis import analyze

%matplotlib inline
```


```python
np.random.seed(987654321)
pets = ['cat', 'dog', 'hamster', 'rabbit', 'bird']
sequence = [pets[np.random.randint(5)] for _ in range(200)]
```


```python
analyze(sequence)
```


![png](img/frequency_6_0.png)


    
    
    Overall Statistics
    ------------------
    
    Total            =  200
    Number of Groups =  5
    
    
    Statistics
    ----------
    
    Rank          Frequency     Percent       Category      
    --------------------------------------------------------
    1             46             23.0000      bird          
    2             43             21.5000      hamster       
    3             41             20.5000      cat           
    4             36             18.0000      dog           
    5             34             17.0000      rabbit        


## Interpreting the Statistics

* **Total** - The total number of data points in the data set.
* **Number of Groups** - The number of unique categories in the data set.
* **Rank** - The ranking of largest category to smallest.
* **Frequency** - The number occurrences of each categorical value in the data set.
* **Percent** - The percent each category makes up of the entire data set.
* **Category** - The unique categorical values in the data set.

## Usage
.. py:function:: analyze(sequence[, percent=False, vertical=True, grid=True, labels=True, dropna=False, order=None, title='Frequency', name='Categories', xname='Categories', yname=None, save_to=None])

    Perform a Frequency analysis on *sequence*.
    
    :param array-like sequence: The array-like object to analyze. It can be a list, tuple, numpy array or pandas Series of string values.
    :param bool percent: Display the percent of each category on the bar chart if **True**, otherwise will display the count of each category.
    :param bool vertical: Display the bar chart with a vertical orientation if **True**.
    :param bool grid: Add grid lines to the bar chart if **True**.
    :param bool labels: Display count or percent labels on the bar chart for each group if **True**.
    :param bool dropna: If **False**, missing values in sequence are grouped together as their own category on the bar chart.
    :param array-like order: Sets the order of the categories displayed on the bar chart according to the order of values in *order*.
    :param str title: The title of the graph.
    :param str name: The name of the data to show on the graph.
    :param str xname: Alias for name.
    :param str yname: The label of the y-axis of the bar chart. The default is "Percent" if percent is **True**, otherwise the default is "Count."
    :param str save_to: If a string value, the path to save the graph to.
## Argument Examples

### sequence

A sequence of categorical values to be analyzed.


```python
analyze(sequence)
```


![png](img/frequency_14_0.png)


    
    
    Overall Statistics
    ------------------
    
    Total            =  200
    Number of Groups =  5
    
    
    Statistics
    ----------
    
    Rank          Frequency     Percent       Category      
    --------------------------------------------------------
    1             46             23.0000      bird          
    2             43             21.5000      hamster       
    3             41             20.5000      cat           
    4             36             18.0000      dog           
    5             34             17.0000      rabbit        


### percent

Controls whether percents are displayed instead of counts on the bar chart. The default is **False**.


```python
analyze(
    sequence, 
    percent=True,
)
```


![png](img/frequency_17_0.png)


    
    
    Overall Statistics
    ------------------
    
    Total            =  200
    Number of Groups =  5
    
    
    Statistics
    ----------
    
    Rank          Frequency     Percent       Category      
    --------------------------------------------------------
    1             46             23.0000      bird          
    2             43             21.5000      hamster       
    3             41             20.5000      cat           
    4             36             18.0000      dog           
    5             34             17.0000      rabbit        


### vertical

Controls whether the bar chart is displayed in a vertical orientation or not. The default is **True**.


```python
analyze(
    sequence, 
    vertical=False,
)
```


![png](img/frequency_20_0.png)


    
    
    Overall Statistics
    ------------------
    
    Total            =  200
    Number of Groups =  5
    
    
    Statistics
    ----------
    
    Rank          Frequency     Percent       Category      
    --------------------------------------------------------
    1             46             23.0000      bird          
    2             43             21.5000      hamster       
    3             41             20.5000      cat           
    4             36             18.0000      dog           
    5             34             17.0000      rabbit        


### grid

Controls whether the grid is displayed on the bar chart or not. The default is **False**.


```python
analyze(
    sequence, 
    grid=True,
)
```


![png](img/frequency_23_0.png)


    
    
    Overall Statistics
    ------------------
    
    Total            =  200
    Number of Groups =  5
    
    
    Statistics
    ----------
    
    Rank          Frequency     Percent       Category      
    --------------------------------------------------------
    1             46             23.0000      bird          
    2             43             21.5000      hamster       
    3             41             20.5000      cat           
    4             36             18.0000      dog           
    5             34             17.0000      rabbit        


### labels

Controls whether the count or percent labels are displayed or not. The default is **True**.


```python
analyze(
    sequence, 
    labels=False,
)
```


![png](img/frequency_26_0.png)


    
    
    Overall Statistics
    ------------------
    
    Total            =  200
    Number of Groups =  5
    
    
    Statistics
    ----------
    
    Rank          Frequency     Percent       Category      
    --------------------------------------------------------
    1             46             23.0000      bird          
    2             43             21.5000      hamster       
    3             41             20.5000      cat           
    4             36             18.0000      dog           
    5             34             17.0000      rabbit        


### dropna

Removes missing values from the bar chart if **True**, otherwise, missing values are grouped together into a category called "nan". The default is **False**.


```python
# Convert 10 random values in sequence to NaN.
for _ in range(10):
    sequence[np.random.randint(200)] = np.nan
```


```python
analyze(sequence)
```


![png](img/frequency_30_0.png)


    
    
    Overall Statistics
    ------------------
    
    Total            =  200
    Number of Groups =  6
    
    
    Statistics
    ----------
    
    Rank          Frequency     Percent       Category      
    --------------------------------------------------------
    1             43             21.5000      bird          
    2             42             21.0000      hamster       
    3             39             19.5000      cat           
    4             33             16.5000      dog           
    4             33             16.5000      rabbit        
    5             10             5.0000        nan          



```python
analyze(
    sequence, 
    dropna=True,
)
```


![png](img/frequency_31_0.png)


    
    
    Overall Statistics
    ------------------
    
    Total            =  200
    Number of Groups =  6
    
    
    Statistics
    ----------
    
    Rank          Frequency     Percent       Category      
    --------------------------------------------------------
    1             43             21.5000      bird          
    2             42             21.0000      hamster       
    3             39             19.5000      cat           
    4             33             16.5000      dog           
    4             33             16.5000      rabbit        
    5             10             5.0000        nan          


### order

A list of category names that sets the order for how categories are displayed on the bar chart. If sequence contains missing values, the category "nan" is shown first.


```python
analyze(
    sequence, 
    order=['rabbit', 'hamster', 'dog', 'cat', 'bird'],
)
```


![png](img/frequency_34_0.png)


    
    
    Overall Statistics
    ------------------
    
    Total            =  200
    Number of Groups =  6
    
    
    Statistics
    ----------
    
    Rank          Frequency     Percent       Category      
    --------------------------------------------------------
    1             43             21.5000      bird          
    2             42             21.0000      hamster       
    3             39             19.5000      cat           
    4             33             16.5000      dog           
    4             33             16.5000      rabbit        
    5             10             5.0000        nan          


If there are categories in *sequence* that aren't listed in *order*, they are reported as "nan" on the bar chart.


```python
analyze(
    sequence, 
    order=['bird', 'cat', 'dog'],
)
```


![png](img/frequency_36_0.png)


    
    
    Overall Statistics
    ------------------
    
    Total            =  200
    Number of Groups =  6
    
    
    Statistics
    ----------
    
    Rank          Frequency     Percent       Category      
    --------------------------------------------------------
    1             43             21.5000      bird          
    2             42             21.0000      hamster       
    3             39             19.5000      cat           
    4             33             16.5000      dog           
    4             33             16.5000      rabbit        
    5             10             5.0000        nan          


Missing values can be dropped from the bar chart with *dropna*=**True**. 


```python
analyze(
    sequence, 
    order=['bird', 'cat', 'dog'], 
    dropna=True,
)
```


![png](img/frequency_38_0.png)


    
    
    Overall Statistics
    ------------------
    
    Total            =  200
    Number of Groups =  6
    
    
    Statistics
    ----------
    
    Rank          Frequency     Percent       Category      
    --------------------------------------------------------
    1             43             21.5000      bird          
    2             42             21.0000      hamster       
    3             39             19.5000      cat           
    4             33             16.5000      dog           
    4             33             16.5000      rabbit        
    5             10             5.0000        nan          


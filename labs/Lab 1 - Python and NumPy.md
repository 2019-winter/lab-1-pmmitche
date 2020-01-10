---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
**Parker Mitchell**


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


<ol>
    <li>More details about editing shortcuts and the editing modes of the J notebook (even though some were outlined</li>
    <li>Why notebooks are often chosen to be used mainly in the data science field</li>
    <li>Other notebook shortcuts and general tips for improving productivity/efficiency</li>
</ol>


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.


## Exercise 1

```python
# YOUR SOLUTION HERE
import numpy as np

a = np.full((6, 4), 2)
print(a)
```

## Exercise 2

```python
# YOUR SOLUTION HERE
b = np.ones((6, 4), dtype=int)
np.fill_diagonal(b, 3)
print(b)
```

## Exercise 3

```python
# YOUR SOLUTION HERE
print('a * b (element wise multiplication):\n')
print(a * b)
print('\n np.dot(a, b) (dot product)\n')
print(np.dot(a, b))
```

a*b is element by element, not the dot product. The number of columns in the first matrix (columns = 4) is not equal to the number of rows in the second matrix (rows = 6). The inners should match in order for you to be able to multiply matrices.


## Exercise 4

```python
# YOUR SOLUTION HERE
print(np.dot(a.transpose(), b))
print()
print(np.dot(a, b.transpose()))
```

The results are different shapes due to the dimensions of the matrices that were multiplied. The first time we do the dot product, matrix 'a' is a 6x4 transposed to a 4x6. Matrix 'b' is a 6x4 still. The outers are the dimensions that will make up the dot product of these two matrices, so it will be a 4x4 matrix.

For the second dot product, 'a' is a 6x4 matrix and 'b' is transposed to a 4x6 matrix. Looking at the outers, we can see that the dot product should produce a 6x6 matrix, which it does as shown by the second matrix.


## Exercise 5 ##

```python
# YOUR SOLUTION HERE
def print_stuff(string):
    print(string)

print_stuff("hello world")
```

## Exercise 6 ##

```python
# YOUR SOLUTION HERE
def do_array_stuff():
    c = np.arange(5)
    d = np.arange(0, 100, 3)
    e = np.zeros((3, 4), dtype=int)
    f = np.arange(10).reshape(5,2)
    print(c)
    print()
    print(d)
    print()
    print(e)
    print()
    print(f)
    
    print("\nArray c dimensions: %d" % np.ndim(c))
    print("\nArray d size: %d" % np.size(d))
    print("\nArray f sum: %d" % np.sum(f))
    print("\nArray f size: %d" % np.size(f))
    print("\nArray f average: %.2f" % (np.sum(f) / np.size(f)))
    
    
do_array_stuff()
```

## Exercise 7

```python
# YOUR SOLUTION HERE
e = np.arange(9).reshape(3, 3)
e[1][0] = 1
e[2][2] = 1

print(e)
print()

def thing1(e):
    count = 0
    for row in e.shape[0]:
        for col in e.shape[1]:
            if e[row][col] == 1:
                count += 1

print('Manual count of number of ones:', count)
print()
x = np.where(e == 1)
print(x)
print("Count of 1's using 'where':", len(x[0]))

```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
# YOUR SOLUTION HERE
import pandas as pd
import numpy as np

a = pd.DataFrame(np.full((6, 4), 2))
a
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python
# YOUR SOLUTION HERE
b = pd.DataFrame(np.full((6, 4), 1))
np.fill_diagonal(b.values, 3)
b
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.

```python
# YOUR SOLUTION HERE

print('a * b (element wise multiplication):\n')
print(a * b)
print('\n a.dot(b) (dot product)\n')
print(a.dot(b))
```

The number of columns in the first matrix (columns = 4) is not equal to the number of rows in the second matrix (rows = 6). The inners should match in order for you to be able to multiply matrices.


## Exercise 11
Repeat exercise A.7 using a dataframe.

```python
# YOUR SOLUTION HERE
e = np.arange(9).reshape(3, 3)
f = pd.DataFrame(e)
e[1][0] = 1
e[2][2] = 1

print(e)
print()
def thing2(e):
    count = 0
    for row in e.shape[0]:
        for col in e.shape[1]:
            if e.iloc[row, col] == 1:
                count += 1

print('Manual count of number of ones:', count)
print()
x = np.where(e == 1)
print(x)
print("Count of 1's using 'where':", len(x[0]))
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
## YOUR SOLUTION HERE
titanic_df['name']
```

## Exercise 13 ##
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
## YOUR SOLUTION HERE

# NOTE: .set_index was giving me a weird error saying that column 'sex' didn't exist, so I reset titanic_df
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df.set_index('sex',inplace=True)
titanic_df.loc['female']
```

## Exercise 14
How do you reset the index?

```python
## YOUR SOLUTION HERE
titanic_df.reset_index()
```

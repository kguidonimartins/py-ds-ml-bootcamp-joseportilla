
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___
# Python Crash Course Exercises - Solutions

This is an optional exercise to test your understanding of Python Basics. If you find this extremely challenging, then you probably are not ready for the rest of this course yet and don't have enough programming experience to continue. I would suggest you take another course more geared towards complete beginners, such as [Complete Python Bootcamp](https://www.udemy.com/complete-python-bootcamp/?couponCode=PY20)

## Exercises

Answer the questions or complete the tasks outlined in bold below, use the specific method described if applicable.

** What is 7 to the power of 4?**


```python
7**4
```




    2401



** Split this string:**

    s = "Hi there Sam!"
    
**into a list. **


```python
s = 'Hi there Sam!'
```


```python
s.split()
```




    ['Hi', 'there', 'dad!']



** Given the variables:**

    planet = "Earth"
    diameter = 12742

** Use .format() to print the following string: **

    The diameter of Earth is 12742 kilometers.


```python
planet = "Earth"
diameter = 12742
```


```python
print("The diameter of {} is {} kilometers.".format(planet,diameter))
```

    The diameter of Earth is 12742 kilometers.


** Given this nested list, use indexing to grab the word "hello" **


```python
lst = [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]
```


```python
lst[3][1][2][0]
```




    'hello'



** Given this nest dictionary grab the word "hello". Be prepared, this will be annoying/tricky **


```python
d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}
```


```python
d['k1'][3]['tricky'][3]['target'][3]
```




    'hello'



** What is the main difference between a tuple and a list? **


```python
# Tuple is immutable
```

** Create a function that grabs the email website domain from a string in the form: **

    user@domain.com
    
**So for example, passing "user@domain.com" would return: domain.com**


```python
def domainGet(email):
    return email.split('@')[-1]
```


```python
domainGet('user@domain.com')
```




    'domain.com'



** Create a basic function that returns True if the word 'dog' is contained in the input string. Don't worry about edge cases like a punctuation being attached to the word dog, but do account for capitalization. **


```python
def findDog(st):
    return 'dog' in st.lower().split()
```


```python
findDog('Is there a dog here?')
```




    True



** Create a function that counts the number of times the word "dog" occurs in a string. Again ignore edge cases. **


```python
def countDog(st):
    count = 0
    for word in st.lower().split():
        if word == 'dog':
            count += 1
    return count
```


```python
countDog('This dog runs faster than the other dog dude!')
```




    2



** Use lambda expressions and the filter() function to filter out words from a list that don't start with the letter 's'. For example:**

    seq = ['soup','dog','salad','cat','great']

**should be filtered down to:**

    ['soup','salad']


```python
seq = ['soup','dog','salad','cat','great']
```


```python
list(filter(lambda word: word[0]=='s',seq))
```




    ['soup', 'salad']



### Final Problem
**You are driving a little too fast, and a police officer stops you. Write a function
  to return one of 3 possible results: "No ticket", "Small ticket", or "Big Ticket". 
  If your speed is 60 or less, the result is "No Ticket". If speed is between 61 
  and 80 inclusive, the result is "Small Ticket". If speed is 81 or more, the result is "Big    Ticket". Unless it is your birthday (encoded as a boolean value in the parameters of the function) -- on your birthday, your speed can be 5 higher in all 
  cases. **


```python
def caught_speeding(speed, is_birthday):
    
    if is_birthday:
        speeding = speed - 5
    else:
        speeding = speed
    
    if speeding > 80:
        return 'Big Ticket'
    elif speeding > 60:
        return 'Small Ticket'
    else:
        return 'No Ticket'
```


```python
caught_speeding(81,True)
```




    'Small Ticket'




```python
caught_speeding(81,False)
```




    'Big Ticket'



# Great job!

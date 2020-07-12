<img align="right"  height="100" src="../images/lesson03.png">

# Decision Tree - Artificial Intelligence Course
3rd lesson of the Artificial Intelligence Crash Course for all<BR/>
By Diogo Cortiz (TIDD/PUC-SP)<BR/>

Theoretical class<BR/>

The video shows how to calculate partial derivatives, power rule and chain rule.<BR/> 
These are the fundamentals necessary to understand the next class of the course, on Gradient Descending.<BR/>


[Video at YouTube](https://www.youtube.com/watch?v=ecYpXd4WREk&t=3376s) <BR/>


## Theory
Entropy is the measurement of uncentainty:
as bigger the entropy as bigger the mess and uncentainty<BR/>
target: reduce entropy<BR/>

_Book: The information, Claude Shannon, James Gleick_<BR/>

When we binary values, the maximun entropy is 1.<BR/> In multiclasses system, the entropy can be higher.<BR/>

### Gain of information

Calculate the feature with better gain of information, for each feature.<BR/>
We choose the feature with higher information gain.<BR/><BR/>
height(son)= n of samples son/n of samples parent<BR/><BR/>
gain= entropy(parent)-sum heigh (sons)*entropy(sons)<BR/><BR/>

Information Gain Calculator: https://planetcalc.com/8421/<BR/>

### Recursivity

Overfit: too large tree.<BR/>

<p align="center">
<img src = "../images/overfitting.png"  width=600> 
<p> <br/> 

### Trade-off of bias and variance

<p align="center">
<img src = "../images/bias-variance.png" width=600>  
<p><br/> 

Variance is how the model adapts to different datasets. <BR/>

Overfitting causes high variance because the model behaves well in training and baddly in testing.<BR/>

The model fits the data. Many errors = high variable<BR/>

## Code
Analysis of a dataset on COVID-19 from the Hospital Israelita Albert Einstein, Brazil.<BR/>
About SARS-CoV-2 for almost 6000 patients.<BR/>
The dataset was redcucted to learning, removing the less important variables.<BR/>

## Other definitions
linear regression is a supervised learning algorithm that predicts an outcome based on continuous features. Linear regression is versatile in the sense that it has the ability to be run on a single variable (simple linear regression) or on many features (multiple linear regression). The way it works is by assigning optimal weights to the variables in order to create a line (ax + b) that will be used to predict output. Check out the video below for a more thorough explanation.
[source](https://www.linkedin.com/pulse/hitchhikers-guide-machine-learning-python-conor-dewey/)




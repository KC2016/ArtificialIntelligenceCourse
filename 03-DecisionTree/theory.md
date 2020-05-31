## Theory:
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
<img src = "images/overfitting.png"  width=600> 
<p> <br/> 

### Trade-off of bias and variance

<p align="center">
<img src = "images/bias-variance.png" width=600>  
<p><br/> 

Variance is how the model adapts to different datasets. <BR/>

Overfitting causes high variance because the model behaves well in training and baddly in testing.<BR/>

The model fits the data. Many errors = high variable<BR/>

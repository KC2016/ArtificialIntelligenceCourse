<img align="right"  height="100" src="../images/lesson09.png">

#  Neural Networks, Perceptron - Artificial Intelligence Course
9th lesson of the Artificial Intelligence Crash Course for all<BR/>
By Diogo Cortiz (TIDD/PUC-SP)<BR/>

- Theoretical class<BR/>

The video shows a little of the history of neural networks and explains how to model Perceptron, one of the first artificial neuron projects that paved the way so that now, decades later, we can have extremely complex models of neural networks.<BR/>

[video_at_youtube](https://www.youtube.com/watch?v=fEukSrpDPH0&t=2s)

<BR/>
Perceptron:<BR/>
features (input) * their weights, sum features with weight, apply a activation fuction, have the prediction (output).<BR/>
<BR/>
step = the most simple activation function<BR/>
if > 0 then 1<BR/>
if <= 0 then 0<BR/>
<BR/>
y = real value<BR/>
y_hat = prediciton of the model<BR/>
<BR/>
Recipe:<BR/>
1- start with a supervised data set (with lables)<BR/>
2- feed forward (predictions)<BR/>
3- calculate the loss and update the weights<BR/>
4- repeat 2 and 3 for each feature (datapoint)(= perceptron rule)<BR/>

Concepts:
- Epoch: repetitions.<BR/>
- Dimension of the batch: number of examples in a training in a batch (before the model update the weights).<BR/>
- Interactions: number of bacht processment by the algorithm.<BR/>
- Bias: aditional weight.<BR/><BR/>

<img align="left" src="../images/calculate_loss.png"><BR/>

<img align="left"   src="../images/bias_nn.png"><BR/>

Limitations:<BR/>
Does not work well when they are not linearly separated.<BR/>

BOOK: Perceptrons, Marwon L. Minsky<BR/>
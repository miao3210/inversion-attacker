# inversion-attacker
A package aims at gradient inversion for general deep learning algorithms, including deep learning and reinforcement learning.

Three gradient inversion attacker classes are provided. 
  
  The GradientAttackerBase class provides the basic gradient inversion attack. 

  The Balanced class utilizes the gradient balance method for the gradient reconstruction loss and the prior loss. More information of loss balancing can be found at MetaBalance: Improving Multi-Task Recommendations via Adapting Gradient Magnitudes of Auxiliary Tasks (https://arxiv.org/abs/2203.06801). 
  
  BiPart class provides a two-stage gradient inversion attack framework. typically, the first stage conducts gradient inversion for linear layers and the second stage focuses on the convolution layers. This allows auxiliary supervision signal for the convolution layers, namely using the input of linear layers to supervise the output of convolution layers.

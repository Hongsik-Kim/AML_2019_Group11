# AML_2019_Group11 : Gradient Descent

## 1. Explain concisely, in layman’s terms without using any formulae

In machine learning, we often have to minimize a loss function (say, MSE) of given model. However, it is extremely hard to analytically minimize MSE as the number of parameters in the model goes up. Hence, we use numerical minimizing technique, gradient descent, as an alternative. Gradient is the direction which the function value can increase in the fastest speed. The concept of gradient descent (hereafter, GD) is quite straightforward. Given an initial point (θ_0), we simple update θ by subtracting gradient evaluated at previous θ multiplied by certain parameter called learning-rate (η). By updating θ, we get closer to the certain point, local (global) minimum of the function. We keep updating θ until there is no significant difference between previous gradient and the latest gradient. However, plain GD method has some drawbacks. For example, it can oscillate depending on the η or stuck in the saddle-point. Hence, two advanced methods have been devised. First, momentum is a method that can reduce oscillation and enhance convergence. Second, Nesterov’s Accelerated Gradient (NAG) is extended version of momentum method. It has better convergence than momentum method.

# Function Description
![THC_plot](https://user-images.githubusercontent.com/52567223/60716640-dbab1f80-9f17-11e9-95d8-78ada3cdb5c8.png)


Plain vanilla Gradient Descent: initial value is set at [2,2]

 	 
[Figure 1] Minimized function values with respect to (w.r.t) different learning-rates. As shown in above figure, if the learning-rate is too small, the minimized function values get stuck in a local minimum value of 0.3

[Figure 2] The number of iteration w.r.t different learning-rates. The smaller the learning-rate, the larger the iteration number and longer computation time.

 	 
[Figure 3] Converging pattern of function values w.r.t iteration numbers. As iteration number goes up, the function gets stuck in a local minimum (0.3) and converges to it.	

[Figure 4] Converging pattern of x w.r.t iteration numbers. As iteration number goes up, the x gradually converges to a local maximum point of [1.75, -0.87].

Two alternative GD: initial value is set at [2,2]

[Figure 5] Momentum method: Minimized function values w.r.t different learning-rates. Regardless of learning-rates, the momentum method shows stable minimization result (at global minimum).

[Figure 6] NAG method: Minimized function values w.r.t different learning-rates. Regardless of learning-rates, the NAG method shows stable minimization result (at global minimum).

 	 
[Figure 7] Momentum method: Converging pattern of function values w.r.t iterations. As iteration number goes up, the function gradually converges to global minimum value of 0 even if the learning-rate (η) is very small (0.001).

[Figure 8] NAG method: Converging pattern of function values w.r.t iterations. As iteration number goes up, the function gradually converges to global minimum value of 0 even if the learning-rate (η) is very small (0.001).


Conclusion

Plain vanilla GD is simple and powerful way to find a minimum value of a function. However, if the function has many local minima (e.g., Three-hump-camel function above), it’s performance could be bad. On the other hand, momentum method and NAG method are not only faster than plain vanilla GD, but provide much better convergence result.

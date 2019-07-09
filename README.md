# AML_2019_Group11 : Gradient Descent

## 1. Explain concisely, in layman’s terms without using any formulae

In machine learning, we often have to minimize a loss function of given model. However, it is extremely hard to analytically minimize a loss function as the number of parameters in the model increases. Hence, we use numerical minimizing technique, gradient descent, as an alternative. Gradient denotes the direction  in which the function value can increase in the fastest speed. The concept of gradient descent (hereafter, GD) is quite straightforward. Given an initial point (θ_0), we simply update θ by subtracting the gradient evaluated at previous θ multiplied by a certain parameter called learning-rate (η). By updating θ, we get closer to the certain point, local (global) minimum of the function. We keep updating θ until there is no significant difference between the previous gradient and the latest gradient. However, plain GD method has some drawbacks. For example, it can oscillate depending on the η or can be stuck in the saddle-point. Hence, two advanced methods have been devised. First, momentum is a method that can reduce oscillation and enhance convergence. Second, Nesterov’s Accelerated Gradient (NAG) is an extended version of the momentum method and it is known better convergence than the momentum method.

## 2. Function Description : Three-Hump-Camel function

![THC_plot](https://user-images.githubusercontent.com/52567223/60716640-dbab1f80-9f17-11e9-95d8-78ada3cdb5c8.png)

Three-Hump-Camel function is definded on (ℝxℝ), has three local minima, and has global minimum at x=(0,0) with the function value of 0

## 3. Plain vanilla Gradient Descent: initial value is set at [2,2]

![Figure1](https://user-images.githubusercontent.com/52567223/60723143-cd1a3380-9f2a-11e9-8709-74b28a98cbe5.png)

[Figure 1] Minimized function values with respect to (w.r.t) different learning-rates. As shown in above figure, if the learning-rate is too small, the minimized function values get stuck in a local minimum value of 0.3

![Figure2](https://user-images.githubusercontent.com/52567223/60723144-cd1a3380-9f2a-11e9-8147-0d6ffb8818eb.png)

[Figure 2] The number of iteration w.r.t different learning-rates. The smaller the learning-rate, the larger the iteration number and longer computation time.

![Figure3](https://user-images.githubusercontent.com/52567223/60723145-cd1a3380-9f2a-11e9-95d5-475213322315.png)

[Figure 3] Converging pattern of function values w.r.t iteration numbers. As iteration number goes up, the function gets stuck in a local minimum (0.3) and converges to it.	

![Figure4](https://user-images.githubusercontent.com/52567223/60723146-cd1a3380-9f2a-11e9-9e90-054149792e8c.png)

[Figure 4] Converging pattern of x w.r.t iteration numbers. As iteration number goes up, the x gradually converges to a local maximum point of [1.75, -0.87].

## 4. Two alternative Gradient Descent: initial value is set at [2,2]

![Figure5](https://user-images.githubusercontent.com/52567223/60723147-cdb2ca00-9f2a-11e9-8f57-f83bf3f3a469.png)

[Figure 5] Momentum method: Minimized function values w.r.t different learning-rates. Regardless of learning-rates, the momentum method shows stable minimization result (at global minimum).

![Figure6](https://user-images.githubusercontent.com/52567223/60723148-cdb2ca00-9f2a-11e9-9278-2d1e4bca840b.png)

[Figure 6] NAG method: Minimized function values w.r.t different learning-rates. Regardless of learning-rates, the NAG method shows stable minimization result (at global minimum).

![Figure7](https://user-images.githubusercontent.com/52567223/60723149-cdb2ca00-9f2a-11e9-97cb-9c614d756db9.png)

[Figure 7] Momentum method: Converging pattern of function values w.r.t iterations. As iteration number goes up, the function gradually converges to global minimum value of 0 even if the learning-rate (η) is very small (0.001).

![Figure8](https://user-images.githubusercontent.com/52567223/60723151-ce4b6080-9f2a-11e9-875f-083f39f665df.png)

[Figure 8] NAG method: Converging pattern of function values w.r.t iterations. As iteration number goes up, the function gradually converges to global minimum value of 0 even if the learning-rate (η) is very small (0.001).


## 5. Conclusion

Plain vanilla GD is a simple and powerful way to find a minimum value of a function. However, if the function has many local minima (e.g., Three-hump-camel function above), GD could show a poor performance. On the other hand, momentum method and NAG method are not only faster than the plain vanilla GD, but also provide much better convergence result.

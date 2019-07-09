""" 
Introduction to Machine Learning

Course work Part 1

Group 11

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
os.chdir('PUT YOUR WORKING DIRECTORY IN HERE')

#=========================================================================================================================================================================================================================

def Three_Hump_Camel(x):
    x1, x2 = x
    return 2*(x1**2) - 1.05*(x1**4) + (x1**6)/6 + (x1*x2) + (x2**2)

def Three_Hump_Camel_gradient(x):
    x1,x2=x
    f1 = 4*x1 - 4.2*(x1**3) + (x1**5) + x2
    f2 = x1 + 2*x2
    return np.array([f1,f2])

#=========================================================================================================================================================================================================================


# Plotting the THC function
x1 = np.linspace(-2,2,100)
x2 = np.linspace(-2,2,100)
x1grid, x2grid = np.meshgrid(x1, x2)
xx = np.stack([x1grid, x2grid])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1grid, x2grid, Three_Hump_Camel(xx),cmap=cm.gist_rainbow)
ax.set_title('Three-Hump Camel function')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

plt.show()


#=========================================================================================================================================================================================================================

# Importing 2-dimensional gradient descent module
from gd_2d import gd_2d_plain, gd_2d_momentum, gd_2d_NAG

# Initial value
x0 = [2,2]

#=========================================================================================================================================================================================================================

# Plain vanilla gd
object1 = gd_2d_plain(Three_Hump_Camel,Three_Hump_Camel_gradient)
res1 = object1.minimize(x0,0.001,tol=1e-8)
print('function value of plain method : ',np.around(res1.fval,5))

#=========================================================================================================================================================================================================================

# Momentum gd
object2 = gd_2d_momentum(Three_Hump_Camel,Three_Hump_Camel_gradient)
res2 = object2.minimize(x0,0.001,tol=1e-8)
print('function value of momentum method : ',np.around(res2.fval,5))

#=========================================================================================================================================================================================================================

# NAG
object3 = gd_2d_NAG(Three_Hump_Camel,Three_Hump_Camel_gradient)
res3 = object3.minimize(x0,0.001,tol=1e-8)
print('function value of NAG method : ',np.around(res3.fval,5))


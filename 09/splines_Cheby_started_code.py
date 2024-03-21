import numpy as np
import scipy.interpolate as intp  # SciPy has more interpolation functions than NumPy.
import matplotlib.pyplot as plt

def f(x):                      # Test function
    return np.arctan(5.0*x)
# Domain
left = -2
right = 2

#Grid for plotting
Nplot = 1000
xs = np.linspace(-2,2,Nplot) # Grid for plotting

N0 = 2
N1 = 7
Ntest = N1 - N0 + 1
errs = np.zeros((Ntest,3))

# Spline interpolation:
for n in range(N0,N1+1):
    N = 2**n                                                      # Fix nr. of knots.
    
    x_interpolate = np.linspace(left, right, n)                   # The locations of the knots. Equivalent Nodes.
    y_interpolate = f(x_interpolate)                              # The y-values at the knots.
    
    # want to interpolate using scipy interpolate library
   
    Function_cubic = intp.CubicSpline(x_interpolate,y_interpolate,bc_type = 'natural')    # Using the spline function from SciPy.interpolate.
    
    y_plot_interpolate = Function_cubic.__call__(x_plot)                                                      # This SciPy function returns a PPoly object, this is how to evaluate the interpolant.

    #Plot function and interpolant
    plt.plot(x_plot, y_plot,'-k',label='f(x)')
    plt.plot(x_plot,y_plot_interpolate,'-r',label='cubic spline on '+str(N)+' knots') # Plot with legend.
    plt.ylim([-2,2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    e = abs(y_plot-y_plot_interpolate)                                                 # Estimate the error on the fine grid.
    plt.title('|f(x)-S_N(x)| for N='+str(N))
    plt.xlabel('x')
    plt.ylabel('error')
    plt.plot(x_plot,e,'-r')
    plt.show()
    errs[n-N0,0] = float(N)                                        # Remember the error to compare the methods.
    errs[n-N0,1] = np.max(e)

# That still doesn't look nice, the error increases near the boundary. Try differnt nodes..

for n in range(N0,N1+1):
    N = 2**n
   
    x = 2.0*np.cos(np.linspace(0,N,N+1) * np.pi/float(N))           # Set the interpolation nodes.
    
    
    y_interpolate = f(x_interpolate)                                                           # Compute the corresponding y-values.
    ys =  intp.barycentric_interpolate(x_interpolate,y_interpolate,x_plot)                                                          # Use an in-built function or our own..

    # Plot Function and Interpolant
    plt.plot(x_plot,y_plot,'-k',x_plot_interpolate,y_plot_interpolate,'-r')
    plt.ylim([-2,2])
    plt.show()
    
    
    e = abs(y_plot-y_plot_interpolate)
    plt.plot(x_plot,e,'-r')
    plt.show()
    errs[n-N0,2] = np.max(e)

plt.loglog(errs[:,0],errs[:,1],label='error of spline interpolation')
plt.loglog(errs[:,0],errs[:,2],label='error of nonuniform interpolation')
plt.xlabel('nr. of knots/nodes')
plt.ylabel('error')
plt.legend()
plt.show()

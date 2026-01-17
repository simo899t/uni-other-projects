"""linreg.py

This assignment is based on the Linear Regression - Least Squares
theory.  The present script is associated with the text document
available from the course web page.

Do not touch the imports, and specifically, do not import matplotlib
in this file! Use the provided file draw.py for visualization. You can
run it by executing the script from command line:

python3 draw.py


The imports listed below should be enough to accomplish all tasks.

The functions /docstring/s contain some real examples and usage of the
functions. You can run these examples by executing the script from
command line:

python3 linreg.py

Note that the unit tests for the final grading may contain different
tests, and that certain requirements given below are not tested in the
testing before the final testing.
"""

import numpy as np
from scipy import linalg as la

np.set_printoptions(precision=3)

def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b.

    Parameters:
        A ((m,n) ndarray): A matrix 
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.

    Examples
    --------
    >>> A = np.array([[4, 1],[1, 1],[8, 9],[6, 9],[5, 2],[7, 7],[7, 1],[5, 1]])
    >>> b = np.array([[8],[4],[1],[8],[4],[6],[7],[8]])
    >>> np.array(least_squares(A,b))
    array([[ 1.236],
           [-0.419]])
    """
    # QR decomposition of matrix A
    Q, R = la.qr(A, mode = "economic")
    
    # Transform the right-hand side: Q^T * b
    Qb = Q.T @ b
    
    # Solve the upper triangular system: R * x = Q^T * b
    x = la.solve(R, Qb)
    
    return x 


def linear_model(x,y):
    """Find the a and b coefficients of the least squares line y = ax + b.

    Parameters
    ----------
    x       : np.ndarray : a numpy array of floats for the input (predictor variables) 
    y       : np.ndarray : a numpy array of floats for the output (response variable)
    
    Returns
    -------
    (a,b)   : a tuple containing the coefficients of the line y = ax + b.

    Examples
    --------
    >>> x = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    >>> y = np.array([1.75, 1.91, 2.03, 2.13, 2.22, 2.30, 2.37, 2.43])
    >>> np.array(linear_model(x,y))
    array([0.095, 1.621])
    """
    # Create the design matrix A for y = ax + b
    # A has two columns: [x values, ones for the constant term]
    A = np.column_stack([x, np.ones(len(x))])
    
    # Reshape y to be a column vector
    y_col = y.reshape(-1, 1)
    
    # Use QR decomposition via least_squares to solve A * [a, b]^T = y
    coefficients = least_squares(A, y_col)
    
    # Extract slope (a) and intercept (b)
    a = coefficients[0, 0]  # slope
    b = coefficients[1, 0]  # intercept
    
    return [a, b]



def exponential_model(x,y):
    """Find the a and b coefficients of the best fitting curve y = ae^(bx).

    Parameters
    ----------
    x       : np.ndarray : a numpy array of floats for the input (predictor variables) 
    y       : np.ndarray : a numpy array of floats for the output (response variable)
    
    Returns
    -------
    (a,b)   : a tuple containing the coefficients of the model  y = ae^(bx).

    Examples
    --------
    >>> x = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    >>> y = np.array([1.75, 1.91, 2.03, 2.13, 2.22, 2.30, 2.37, 2.43])        
    >>> np.array(exponential_model(x,y))
    array([1.662, 0.045])
    """
    # Transform y = ae^(bx) to linear form: ln(y) = ln(a) + bx
    # Take natural logarithm of y values
    ln_y = np.log(y)
    
    # Use linear model on (x, ln(y)) to get [b, ln(a)]
    b, ln_a = linear_model(x, ln_y)
    
    # Convert back: a = e^(ln(a))
    a = np.exp(ln_a)
    
    return [a, b]



def power_model(x,y):
    """Find the a and b coefficients of the best fitting curve y = a x^b.

    Parameters
    ----------
    x       : np.ndarray : a numpy array of floats for the input (predictor variables) 
    y       : np.ndarray : a numpy array of floats for the output (response variable)
    
    Returns
    -------
    (a,b)   : a tuple containing the coefficients of the model  y = a x^b.

    Examples
    --------
    >>> x = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    >>> y = np.array([1.75, 1.91, 2.03, 2.13, 2.22, 2.30, 2.37, 2.43])     
    >>> np.array(power_model(x,y))
    array([1.501, 0.219])
    """
    # Transform y = ax^b to linear form: ln(y) = ln(a) + b*ln(x)
    # Take natural logarithm of both x and y values
    ln_x = np.log(x)
    ln_y = np.log(y)
    
    # Use linear model on (ln(x), ln(y)) to get [b, ln(a)]
    b, ln_a = linear_model(ln_x, ln_y)
    
    # Convert back: a = e^(ln(a))
    a = np.exp(ln_a)
    
    return [a, b]



def logarithmic_model(x,y):
    """Find the a and b coefficients of the best fitting curve y = a + b ln x.

    Parameters
    ----------
    x       : np.ndarray : a numpy array of floats for the input (predictor variables) 
    y       : np.ndarray : a numpy array of floats for the output (response variable)
    
    Returns
    -------
    (a,b)   : a tuple containing the coefficients of the model y = a + b ln x.

    Examples
    --------
    >>> x = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    >>> y = np.array([1.75, 1.91, 2.03, 2.13, 2.22, 2.30, 2.37, 2.43])     
    >>> np.array(logarithmic_model(x,y))
    array([1.415, 0.455])
    """
    # For y = a + b*ln(x), this is already linear in ln(x)
    # Transform only x: ln(x)
    ln_x = np.log(x)
    
    # Use linear model on (ln(x), y) to get [b, a]
    b, a = linear_model(ln_x, y)
    
    return [a, b]



def training_error(f,xx,yy):
    """Find the sum of squared errors of the model f on the data xx and yy used to
    determine the parameters of f.

    Parameters
    ----------
    f        : a lambda function containing the fitted parameters 
               implementing one models under study
    xx       : np.ndarray : a numpy array of floats for the input (predictor variables) 
    yy       : np.ndarray : a numpy array of floats for the output (response variable)
    
    Returns
    -------
    err      : a float representing the training error.
    >>> x = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    >>> y = np.array([1.75, 1.91, 2.03, 2.13, 2.22, 2.30, 2.37, 2.43])     
    >>> a,b = power_model(x,y)
    >>> f_pow = lambda xx: a*(xx**b)
    >>> np.array(training_error(f_pow, x, y))
    array(0.008)
    """
    # Calculate predictions using the fitted model
    y_pred = f(xx)
    
    # Calculate residuals (errors)
    residuals = yy - y_pred
    
    # Calculate sum of squared errors
    sse = np.sum(residuals**2)
    
    return np.sqrt(sse)
    





if __name__ == "__main__":
    import doctest
    doctest.testmod()

```python
#########################################################################
# Author: Dan Betea                                                     #
# (C) December 2022                                                     #
# License: CC BY-SA 4.0                                                 #
# License description: https://creativecommons.org/licenses/by-sa/4.0/  #
#########################################################################
```

# LUE patterns 1: linear regression at the hard edge and beyond

Consider the Laguerre-$\alpha$ Unitary Ensemble (LUE-$\alpha$) distribution on ordered tuples of $N$ positive real numbers $(\lambda_1 < \dots < \lambda_N).$ That is, consider the probability measure

$$P(\lambda_1, \dots, \lambda_N)d \lambda_1 \dots d \lambda_N \propto \prod_{1 \leq i < j \leq N} (\lambda_i - \lambda_j)^2 \prod_{1 \leq i \leq N} \lambda_i^{\alpha-1} e^{-\lambda_i} d \lambda_i$$

where $\alpha > 0$ (notice the somewhat less standard $\alpha$ convention we use for LUE). See [this Wikipedia article](https://en.wikipedia.org/wiki/Complex_Wishart_distribution) for the motivation behind this distribution and how it comes about when studying covariance matrices (see in particular the Eigenvalues section).

We do linear regression (in Python, using Scikit-learn) on 

$$(\log i, \log E \lambda_i^s)$$

for $i$ in a certain interval like $1, \dots, 10; 1, \dots, \log N$ or more generally $m_0, \dots, m_0 + M - 1$. Here $\lambda_i$ the $i$-th lowest eigenvalue of the LUE-$\alpha$ ensemble and $s$ is a real number, taken negative for convergence. We consider several cases:

- $m_0 = 1, M = 15$, $\alpha = 4, s = -2$, and $N \in \{10000, 5000, 1000\}$
- $m_0 = 1, M \in \{ 100, 1000, 5000, 7500, 9000, 10000\}$, $\alpha = 4, s = -2$, and $N=10000$

**Some remarks:**

- array indexing starts at 0 by default in Python
- the data files, containing iid samples $(\lambda_1 < \dots < \lambda_N)$, are assumed to be in the same directory as the notebook, and be in the correct format
- steps below can be automated; for this exploratory notebook they are not


```python
# importing the necessary packages
import numpy as np                 # for linear algebra and loading from files
from sklearn import linear_model   # for linear regression
```

## First case: 1000 samples for $\alpha=4, s=-2, N \in \{1000, 5000, 10000 \}$

We set up the parameters, as well as the $x$ and $y$ vectors in the model $y = \beta_0 + \beta_1 \cdot x$. We print out some of the raw data for illustration for this initial step. Python dictionaries are used whenever possible for their convenience. 


```python
Ns = [1000, 5000, 10000]   # matrix sizes to be considered
s = -2.0                   # point where to compute E \lambda_i^s
alpha_str = "4.00"
alpha = float(alpha_str)   # = 4.0, Laguerre parameter alpha-1 
m0 = 0                     # regression starts at \lambda_{m_0+1}
M = 15                     # regression ends with \lambda_{m_0+M} 
range_ms = range(m0, m0+M) # index range for regression

# building up dictionary of filenames (data is read from these files)
filename_dict = {}
for N in Ns:
    filename = f"LUE_N_{N}_alpha_{alpha_str}.txt"
    filename_dict[str(N)] = filename
    
# build the x variables for regression
# x = log i for m0+1 <= i <= m0+M (+1 because of Python 0-indexed arrays) 
# reshape into column vectors needed
log_ms = np.log(np.arange(m0+1, m0+M+1)).reshape(-1, 1)
print("x = log i = ")
print(log_ms.flatten())
print("\n"+73*"-")

# build up a dictionary of the data for each N and
# load up the corresponding matrix A, only M columns: m0, ..., m0+M-1 and
# build the y variables for regression into a dictionary, one vector per N
# (same reshape needed as above)
A_dict = {}
log_expectations_dict = {}
for N in Ns:
    A = np.loadtxt(filename_dict[str(N)], usecols=range_ms)
    A = A[:1000, :]
    A_dict[str(N)] = A
    print()
    print(f"N = {N}\n")
    print("A = eigenvalue sample matrix (iid / line) = \n")
    print(A)
    print()
    print(f"shape of A is: {A.shape}\n")
    # below: axis = 0 means we sum over rows, i.e. take the average of each column (iid eigenval sample)
    log_expectations = np.log(np.mean(A**s, axis=0)).reshape(-1, 1) 
    log_expectations_dict[str(N)] = log_expectations
    print("log E \lambda_i^s = \n")
    print(log_expectations.flatten())
    print("\n"+73*"-")
```

    x = log i = 
    [0.         0.69314718 1.09861229 1.38629436 1.60943791 1.79175947
     1.94591015 2.07944154 2.19722458 2.30258509 2.39789527 2.48490665
     2.56494936 2.63905733 2.7080502 ]
    
    -------------------------------------------------------------------------
    
    N = 1000
    
    A = eigenvalue sample matrix (iid / line) = 
    
    [[0.00914148 0.01421346 0.02985021 ... 0.42833932 0.45033342 0.52411581]
     [0.01065387 0.016576   0.02827056 ... 0.42273216 0.51435102 0.55804161]
     [0.00897166 0.01592701 0.0399726  ... 0.42665176 0.51484552 0.53870744]
     ...
     [0.0104201  0.01963166 0.03846435 ... 0.43218855 0.51115892 0.55671524]
     [0.00582622 0.00984355 0.03220857 ... 0.43013025 0.51010797 0.61515227]
     [0.01404306 0.02702059 0.04454848 ... 0.43967195 0.58155765 0.72504212]]
    
    shape of A is: (1000, 15)
    
    log E \lambda_i^s = 
    
    [10.58149645  8.08605446  6.71897571  5.74520463  4.9633212   4.32960924
      3.76935676  3.28454521  2.84860863  2.454915    2.10854766  1.78467947
      1.47591149  1.20055577  0.94139837]
    
    -------------------------------------------------------------------------
    
    N = 5000
    
    A = eigenvalue sample matrix (iid / line) = 
    
    [[0.00114    0.00280814 0.00431431 ... 0.0926519  0.11608572 0.11824966]
     [0.00111132 0.00349919 0.0055273  ... 0.10164782 0.11596621 0.13266211]
     [0.00043231 0.00354626 0.00623526 ... 0.10506654 0.11092772 0.12206301]
     ...
     [0.00045616 0.00319769 0.0064798  ... 0.09133178 0.11651967 0.12998098]
     [0.00226186 0.0043195  0.00913852 ... 0.1156643  0.13076585 0.14402008]
     [0.00060436 0.00350177 0.00719529 ... 0.08875277 0.10719097 0.10839797]]
    
    shape of A is: (1000, 15)
    
    log E \lambda_i^s = 
    
    [13.67337088 11.30585338  9.94404935  8.93789822  8.17152669  7.5218516
      6.96950095  6.48499387  6.06356981  5.67542682  5.3242507   5.00303445
      4.69810113  4.41511591  4.15786195]
    
    -------------------------------------------------------------------------
    
    N = 10000
    
    A = eigenvalue sample matrix (iid / line) = 
    
    [[0.00033733 0.00259883 0.0047951  ... 0.0440713  0.05207287 0.06427409]
     [0.00077967 0.00120705 0.00350779 ... 0.04596401 0.05579818 0.06630394]
     [0.00040075 0.00085289 0.00203974 ... 0.04768492 0.05348894 0.06787949]
     ...
     [0.00138265 0.00230368 0.00389185 ... 0.04685326 0.05335075 0.06536883]
     [0.00107706 0.00250075 0.00369636 ... 0.04957134 0.0542272  0.06526369]
     [0.0012579  0.00231975 0.00431389 ... 0.04635724 0.05579029 0.06257022]]
    
    shape of A is: (1000, 15)
    
    log E \lambda_i^s = 
    
    [15.1033871  12.6871983  11.33888625 10.33274879  9.5524451   8.91856004
      8.35558169  7.87351933  7.45123803  7.05925234  6.70438788  6.37730676
      6.079491    5.80433728  5.54284714]
    
    -------------------------------------------------------------------------


Next we do the actual regression. We finally record the coefficients $\beta_0, \beta_1$ in the linear model $y = \beta_0 + \beta_1 \cdot x$, as well as the $R^2$ coefficient, in the dictionary ```coeff_dict```. We print out a summary for each value of $N$.


```python
# doing the actual regression
reg_dict = {} # dictionary to hold the regressors
# below: dictionary to hold the coefficients of the regression in the form
# str(N): {"b0": ..., "b1": ..., "r2": ...}
# with b0 = intercept, b1 = slope, r2 = R squared
coeff_dict = {str(N) : {} for N in Ns} 
for N in Ns:
    print(f"\n N = {N}\n")
    log_expectations = log_expectations_dict[str(N)]
    reg_dict[str(N)] = linear_model.LinearRegression()
    reg_dict[str(N)].fit(log_ms, log_expectations) # linear fit
    reg = reg_dict[str(N)]
    print(f" slope:     {reg.coef_.item()}\n intercept: {reg.intercept_.item()}\n R squared: \
 {reg.score(log_ms, log_expectations)}")
    coeff_dict[str(N)]["b0"] = reg.coef_.item()
    coeff_dict[str(N)]["b1"] = reg.intercept_.item()
    coeff_dict[str(N)]["r2"] = reg.score(log_ms, log_expectations)
    print("\n"+73*"-")
```

    
     N = 1000
    
     slope:     -3.548172881151952
     intercept: 10.618974545782708
     R squared:  0.9997037656394379
    
    -------------------------------------------------------------------------
    
     N = 5000
    
     slope:     -3.5206102893835896
     intercept: 13.771257840234028
     R squared:  0.9996302367539628
    
    -------------------------------------------------------------------------
    
     N = 10000
    
     slope:     -3.532143922054429
     intercept: 15.18169525800185
     R squared:  0.9996715306706269
    
    -------------------------------------------------------------------------


## Second case: $\alpha=4, s = -2, N = 10000$, varying number of points

We do the same as before except here $N=10000$ is fixed (as are $\alpha, s$), and we vary the number of points $M$ in our data set: 

$$M \in \{100, 1000, 5000, 7500, 9000, 10000\}.$$

Indeed for the last value we linearly regress *all the available data*. Finally we print out a summary as we go along.


```python
# the same as before, except N=10000 is fixed, and we regress for 
# 1 <= i <= M for M in [100, 1000, 5000, 7500, 9000, 10000]

N = 10000                                   # matrix sizes to be considered
s = -2.0                                    # point where to compute E \lambda_i^s
alpha_str = "4.00"
alpha = float(alpha_str)                    # = 4.0, Laguerre parameter alpha-1 
m0 = 0                                      # regression starts at \lambda_{m_0+1}
Ms = [100, 1000, 5000, 7500, 9000, 10000]   # regression ends with \lambda_{m_0+M} 

# build dict of index ranges
range_ms_dict = {}
for M in Ms:
    range_ms = range(m0, m0+M) # index range for regression
    range_ms_dict[str(M)] = range_ms
    
filename = f"LUE_N_{N}_alpha_{alpha_str}.txt"
# load the whole file into the matrix A_full
A_full = np.loadtxt(filename)
    
# build the x and y variables for regression, using full data
# in each case we then select the first M entries of this vector
log_ms_full = np.log(np.arange(1, N+1)).reshape(-1, 1)
log_expectations_full = np.log(np.mean(A_full**s, axis=0)).reshape(-1, 1) 

reg_dict_2 = {} # dictionary to hold the regressors
# below: dictionary to hold the coefficients of the regression in the form
# str(M): {"b0": ..., "b1": ..., "r2": ...}
# with b0 = intercept, b1 = slope, r2 = R squared
coeff_dict_2 = {str(M) : {} for M in Ms} 

# loop over different values of M and do the regression
for M in Ms:
    print(f"\n N = {N}, M = {M}\n")
    range_ms = range_ms_dict[str(M)]                   
    A = A_full[:, range_ms]                                # select matrix cols to be averaged
    log_expectations = log_expectations_full[range_ms]     # build the y vector
    log_ms = log_ms_full[range_ms]                         # build the x vector
    reg_dict_2[str(M)] = linear_model.LinearRegression()
    reg_dict_2[str(M)].fit(log_ms, log_expectations)       # linear fit
    reg = reg_dict_2[str(M)]
    print(f" slope:     {reg.coef_.item()}\n intercept: {reg.intercept_.item()}\n R squared: \
 {reg.score(log_ms, log_expectations)}")
    coeff_dict_2[str(M)]["b0"] = reg.coef_.item()
    coeff_dict_2[str(M)]["b1"] = reg.intercept_.item()
    coeff_dict_2[str(M)]["r2"] = reg.score(log_ms, log_expectations)
    print("\n"+73*"-")
```

    
     N = 10000, M = 100
    
     slope:     -3.780744870879914
     intercept: 15.66091040785845
     R squared:  0.9992499318603848
    
    -------------------------------------------------------------------------
    
     N = 10000, M = 1000
    
     slope:     -3.943678885839752
     intercept: 16.25556139424464
     R squared:  0.9997446406326693
    
    -------------------------------------------------------------------------
    
     N = 10000, M = 5000
    
     slope:     -4.005669845341229
     intercept: 16.614773378617755
     R squared:  0.9998622769202816
    
    -------------------------------------------------------------------------
    
     N = 10000, M = 7500
    
     slope:     -4.0472824715801385
     intercept: 16.89594031977959
     R squared:  0.9996702668461551
    
    -------------------------------------------------------------------------
    
     N = 10000, M = 9000
    
     slope:     -4.0864595889817
     intercept: 17.17131304217444
     R squared:  0.9992497462934117
    
    -------------------------------------------------------------------------
    
     N = 10000, M = 10000
    
     slope:     -4.128533480066506
     intercept: 17.472940530864584
     R squared:  0.9983577899394526
    
    -------------------------------------------------------------------------


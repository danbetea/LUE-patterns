```python
#########################################################################
# Author: Dan Betea                                                     #
# (C) December 2022                                                     #
# License: CC BY-SA 4.0                                                 #
# License description: https://creativecommons.org/licenses/by-sa/4.0/  #
#########################################################################
```

# LUE patterns 2: linear regression with naive train/test split

Consider the Laguerre-$\alpha$ Unitary Ensemble (LUE-$\alpha$) distribution on ordered tuples of $N$ positive real numbers $(\lambda_1 < \dots < \lambda_N).$ That is, consider the probability measure

$$P(\lambda_1, \dots, \lambda_N)d \lambda_1 \dots d \lambda_N \propto \prod_{1 \leq i < j \leq N} (\lambda_i - \lambda_j)^2 \prod_{1 \leq i \leq N} \lambda_i^{\alpha-1} e^{-\lambda_i} d \lambda_i$$

where $\alpha > 0$ (notice the somewhat less standard $\alpha$ convention we use for LUE). See [this Wikipedia article](https://en.wikipedia.org/wiki/Complex_Wishart_distribution) for the motivation behind this distribution and how it comes about when studying covariance matrices (see in particular the Eigenvalues section).

We do linear regression (in Python, using Scikit-learn) on 

$$(\log i, \log E \lambda_i^s)$$

for $i$ in a certain interval like $1, \dots, 10; 1, \dots, \log N$ or more generally $m_0, \dots, m_0 + M - 1$. Here $\lambda_i$ the $i$-th lowest eigenvalue of the LUE-$\alpha$ ensemble and $s$ is a real number, taken negative for convergence. 

We consider several cases:

- $m_0 = 1, M = 15$, $\alpha = 4, s = -2$, $N \in \{1000, 5000, 10000\}$, and 1000 total samples

and we additionally consider two cases for splitting the data into training and testing: 

- first we try 60%-40%, and
- then we try 80%-20%.

**Important remark:** The dependent variable $x = (\log i)_{m_0 \leq i \leq m_0+M}$ is deterministic.

**Some other remarks:**

- array indexing starts at 0 by default in Python
- the data files, containing iid samples $(\lambda_1 < \dots < \lambda_N)$, are assumed to be in the same directory as the notebook, and be in the correct format
- steps below can be automated; for this exploratory notebook they are not


```python
# importing the necessary packages
import numpy as np                 # for linear algebra and loading from files
from sklearn import linear_model   # for linear regression
```

First we set up the parameters independent of the data split: the values of $N$, $\alpha$, $s$, $m_0, M$, and the $x$ vector.


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
# print("\n"+73*"-")
```

    x = log i = 
    [0.         0.69314718 1.09861229 1.38629436 1.60943791 1.79175947
     1.94591015 2.07944154 2.19722458 2.30258509 2.39789527 2.48490665
     2.56494936 2.63905733 2.7080502 ]


### First case: 60-40 split

Next we set up the train and test data, namely the $y$ vectors corresponding to train and test sets.


```python
# build up a dictionary of the data for each N and
# load up the corresponding matrix A, only M columns: m0, ..., m0+M-1 and
# build the y variables for regression into a dictionary, one vector per N
# (same reshape needed as above)
log_expectations_train_dict = {}
log_expectations_test_dict = {}
for N in Ns:
    A = np.loadtxt(filename_dict[str(N)], usecols=range_ms)
    A = A[:1000, :]              # take only 1000 samples, for consistency
    A_train = A[:600, :]         # 80 % for training
    A_test = A[600:, :]          # 20 % test
    print()
    print(f"N = {N}\n")
    print("A_train = eigenvalue sample matrix (iid / line) = \n")
    print(A_train)
    print()
    print(f"shape of A_train is: {A.shape}\n")
    # below: axis = 0 means we sum over rows, i.e. take the average of each column (iid eigenval sample)
    # build the y variable, for training
    log_expectations_train = np.log(np.mean(A_train**s, axis=0)).reshape(-1, 1) 
    log_expectations_train_dict[str(N)] = log_expectations_train
    # build the y variable, for testing
    log_expectations_test = np.log(np.mean(A_test**s, axis=0)).reshape(-1, 1) 
    log_expectations_test_dict[str(N)] = log_expectations_test
    print("log E \lambda_i^s (train) = \n")
    print(log_expectations_train.flatten())
    print("\n"+73*"-")
```

    
    N = 1000
    
    A_train = eigenvalue sample matrix (iid / line) = 
    
    [[0.00914148 0.01421346 0.02985021 ... 0.42833932 0.45033342 0.52411581]
     [0.01065387 0.016576   0.02827056 ... 0.42273216 0.51435102 0.55804161]
     [0.00897166 0.01592701 0.0399726  ... 0.42665176 0.51484552 0.53870744]
     ...
     [0.0043258  0.01206655 0.03115058 ... 0.45655002 0.57092822 0.67062551]
     [0.00438205 0.03887275 0.05732078 ... 0.50094824 0.55490721 0.66749578]
     [0.0058209  0.01021484 0.035881   ... 0.46072031 0.55426236 0.60527526]]
    
    shape of A_train is: (1000, 15)
    
    log E \lambda_i^s (train) = 
    
    [10.60487598  8.10004521  6.73538035  5.75510236  4.9694163   4.32724725
      3.77248462  3.28860453  2.84765712  2.45641598  2.11183603  1.7882116
      1.47478295  1.201755    0.94275976]
    
    -------------------------------------------------------------------------
    
    N = 5000
    
    A_train = eigenvalue sample matrix (iid / line) = 
    
    [[0.00114    0.00280814 0.00431431 ... 0.0926519  0.11608572 0.11824966]
     [0.00111132 0.00349919 0.0055273  ... 0.10164782 0.11596621 0.13266211]
     [0.00043231 0.00354626 0.00623526 ... 0.10506654 0.11092772 0.12206301]
     ...
     [0.00144574 0.00526298 0.01049865 ... 0.09467053 0.10503609 0.12177519]
     [0.00066836 0.00682101 0.01005465 ... 0.100075   0.13061468 0.13990584]
     [0.0026046  0.00520637 0.0079645  ... 0.08808057 0.100557   0.11083758]]
    
    shape of A_train is: (1000, 15)
    
    log E \lambda_i^s (train) = 
    
    [13.69455166 11.29388867  9.96698935  8.92929003  8.16342951  7.51796829
      6.97376252  6.48424481  6.06317906  5.67491726  5.32283372  5.0050972
      4.70035352  4.41509341  4.15448704]
    
    -------------------------------------------------------------------------
    
    N = 10000
    
    A_train = eigenvalue sample matrix (iid / line) = 
    
    [[0.00033733 0.00259883 0.0047951  ... 0.0440713  0.05207287 0.06427409]
     [0.00077967 0.00120705 0.00350779 ... 0.04596401 0.05579818 0.06630394]
     [0.00040075 0.00085289 0.00203974 ... 0.04768492 0.05348894 0.06787949]
     ...
     [0.00100661 0.00165194 0.00482695 ... 0.05863087 0.06472924 0.06977831]
     [0.00095886 0.00224867 0.00521771 ... 0.04886442 0.05580353 0.0567986 ]
     [0.0017652  0.00242253 0.00349969 ... 0.04788582 0.05601745 0.0645648 ]]
    
    shape of A_train is: (1000, 15)
    
    log E \lambda_i^s (train) = 
    
    [15.21639962 12.69603267 11.33922138 10.32955918  9.55388277  8.93154929
      8.35822421  7.8634264   7.44821947  7.05668318  6.69737211  6.37099209
      6.07400957  5.7990274   5.53662213]
    
    -------------------------------------------------------------------------


#### Training phase

The actual regression goes down below. First the training phase to get the regression line.

**Note:** The actual $R^2$ coefficient displayed below is obviously computed on the training set, and expected to be big. It is still nevertheless *quite high*.


```python
# doing the actual regression
reg_dict = {} # dictionary to hold the regressors
# below: dictionary to hold the coefficients of the regression in the form
# str(N): {"b0": ..., "b1": ..., "r2": ...}
# with b0 = intercept, b1 = slope, r2 = R squared
coeff_dict = {str(N) : {} for N in Ns} 
print("\n Training phase ")
print("\n----------------")
for N in Ns:
    print(f"\n N = {N}\n")
    log_expectations_train = log_expectations_train_dict[str(N)]
    reg_dict[str(N)] = linear_model.LinearRegression()
    reg_dict[str(N)].fit(log_ms, log_expectations_train) # linear fit
    reg = reg_dict[str(N)]
    print(f" slope:     {reg.coef_.item()}\n intercept: {reg.intercept_.item()}\n R squared: \
 {reg.score(log_ms, log_expectations_train)}")
    coeff_dict[str(N)]["b0"] = reg.coef_.item()
    coeff_dict[str(N)]["b1"] = reg.intercept_.item()
    coeff_dict[str(N)]["r2"] = reg.score(log_ms, log_expectations_train)
    print("\n"+73*"-")
```

    
     Training phase 
    
    ----------------
    
     N = 1000
    
     slope:     -3.5565342748568156
     intercept: 10.640085997010413
     R squared:  0.9997252840928547
    
    -------------------------------------------------------------------------
    
     N = 5000
    
     slope:     -3.524984873534735
     intercept: 13.780306377117752
     R squared:  0.9996525781321135
    
    -------------------------------------------------------------------------
    
     N = 10000
    
     slope:     -3.560837784875215
     intercept: 15.241066745513617
     R squared:  0.9997641013695571
    
    -------------------------------------------------------------------------


#### Testing phase

Now we do the predicting phase. Note the $R^2$ coefficient continues to be high.


```python
print("\n Predicting phase ")
print("\n------------------")
for N in Ns:
    print(f"\n N = {N}\n")
    reg = reg_dict[str(N)]
    log_expectations_test = log_expectations_test_dict[str(N)]
    print(" R squared on the test set: ", reg.score(log_ms, log_expectations_test))
    print("\n"+73*"-")
```

    
     Predicting phase 
    
    ------------------
    
     N = 1000
    
     R squared on the test set:  0.9996010133806793
    
    -------------------------------------------------------------------------
    
     N = 5000
    
     R squared on the test set:  0.9995437008969466
    
    -------------------------------------------------------------------------
    
     N = 10000
    
     R squared on the test set:  0.9986867445554635
    
    -------------------------------------------------------------------------


### Second case: 80-20 split


```python
# build up a dictionary of the data for each N and
# load up the corresponding matrix A, only M columns: m0, ..., m0+M-1 and
# build the y variables for regression into a dictionary, one vector per N
# (same reshape needed as above)
log_expectations_train_dict_2 = {}
log_expectations_test_dict_2 = {}
for N in Ns:
    A = np.loadtxt(filename_dict[str(N)], usecols=range_ms)
    A = A[:1000, :]              # take only 1000 samples, for consistency
    A_train = A[200:, :]         # 80 % for training
    A_test = A[:200, :]          # 20 % test
    print()
    print(f"N = {N}\n")
    print("A_train = eigenvalue sample matrix (iid / line) = \n")
    print(A_train)
    print()
    print(f"shape of A_train is: {A.shape}\n")
    # below: axis = 0 means we sum over rows, i.e. take the average of each column (iid eigenval sample)
    # build the y variable, for training
    log_expectations_train = np.log(np.mean(A_train**s, axis=0)).reshape(-1, 1) 
    log_expectations_train_dict_2[str(N)] = log_expectations_train
    # build the y variable, for testing
    log_expectations_test = np.log(np.mean(A_test**s, axis=0)).reshape(-1, 1) 
    log_expectations_test_dict_2[str(N)] = log_expectations_test
    print("log E \lambda_i^s (train) = \n")
    print(log_expectations_train.flatten())
    print("\n"+73*"-")
```

    
    N = 1000
    
    A_train = eigenvalue sample matrix (iid / line) = 
    
    [[0.00405837 0.01576241 0.03433577 ... 0.51706077 0.57475493 0.6476627 ]
     [0.00648518 0.01320577 0.03985386 ... 0.49188188 0.5384115  0.63590799]
     [0.0109845  0.03155448 0.04758467 ... 0.47631312 0.51299513 0.60330846]
     ...
     [0.0104201  0.01963166 0.03846435 ... 0.43218855 0.51115892 0.55671524]
     [0.00582622 0.00984355 0.03220857 ... 0.43013025 0.51010797 0.61515227]
     [0.01404306 0.02702059 0.04454848 ... 0.43967195 0.58155765 0.72504212]]
    
    shape of A_train is: (1000, 15)
    
    log E \lambda_i^s (train) = 
    
    [10.6138792   8.09176491  6.72208319  5.73931204  4.96804947  4.33075151
      3.76904102  3.28213565  2.84679669  2.45226633  2.10730741  1.78289361
      1.47532656  1.19744723  0.94050505]
    
    -------------------------------------------------------------------------
    
    N = 5000
    
    A_train = eigenvalue sample matrix (iid / line) = 
    
    [[0.00250476 0.00369888 0.00814249 ... 0.09167625 0.11054006 0.12709592]
     [0.00116709 0.00259737 0.00807019 ... 0.09373403 0.10646009 0.1226777 ]
     [0.00039302 0.00395358 0.00529761 ... 0.10341038 0.11384548 0.13902808]
     ...
     [0.00045616 0.00319769 0.0064798  ... 0.09133178 0.11651967 0.12998098]
     [0.00226186 0.0043195  0.00913852 ... 0.1156643  0.13076585 0.14402008]
     [0.00060436 0.00350177 0.00719529 ... 0.08875277 0.10719097 0.10839797]]
    
    shape of A_train is: (1000, 15)
    
    log E \lambda_i^s (train) = 
    
    [13.66246821 11.30848859  9.92924621  8.93900948  8.1749269   7.52319692
      6.96911512  6.48355978  6.06256024  5.67470078  5.320529    5.00174115
      4.6969904   4.41269979  4.15538788]
    
    -------------------------------------------------------------------------
    
    N = 10000
    
    A_train = eigenvalue sample matrix (iid / line) = 
    
    [[0.00122284 0.00334067 0.0046821  ... 0.05742837 0.05956978 0.07414231]
     [0.00075535 0.00152887 0.00260423 ... 0.05078005 0.05502777 0.06151355]
     [0.00068556 0.00139475 0.00317724 ... 0.04797801 0.05471206 0.05984219]
     ...
     [0.00138265 0.00230368 0.00389185 ... 0.04685326 0.05335075 0.06536883]
     [0.00107706 0.00250075 0.00369636 ... 0.04957134 0.0542272  0.06526369]
     [0.0012579  0.00231975 0.00431389 ... 0.04635724 0.05579029 0.06257022]]
    
    shape of A_train is: (1000, 15)
    
    log E \lambda_i^s (train) = 
    
    [15.08636083 12.66940511 11.34628246 10.33403811  9.54937016  8.91703858
      8.35667485  7.87483933  7.44832389  7.06140227  6.70579201  6.37958384
      6.0840979   5.80731564  5.54662223]
    
    -------------------------------------------------------------------------


#### Training phase


```python
# doing the actual regression
reg_dict_2 = {} # dictionary to hold the regressors
# below: dictionary to hold the coefficients of the regression in the form
# str(N): {"b0": ..., "b1": ..., "r2": ...}
# with b0 = intercept, b1 = slope, r2 = R squared
coeff_dict_2 = {str(N) : {} for N in Ns} 
print("\n Training phase ")
print("\n----------------")
for N in Ns:
    print(f"\n N = {N}\n")
    log_expectations_train = log_expectations_train_dict_2[str(N)]
    reg_dict_2[str(N)] = linear_model.LinearRegression()
    reg_dict_2[str(N)].fit(log_ms, log_expectations_train) # linear fit
    reg = reg_dict_2[str(N)]
    print(f" slope:     {reg.coef_.item()}\n intercept: {reg.intercept_.item()}\n R squared: \
 {reg.score(log_ms, log_expectations_train)}")
    coeff_dict_2[str(N)]["b0"] = reg.coef_.item()
    coeff_dict_2[str(N)]["b1"] = reg.intercept_.item()
    coeff_dict_2[str(N)]["r2"] = reg.score(log_ms, log_expectations_train)
    print("\n"+73*"-")
```

    
     Training phase 
    
    ----------------
    
     N = 1000
    
     slope:     -3.556985804860991
     intercept: 10.637124810312272
     R squared:  0.9997278908057867
    
    -------------------------------------------------------------------------
    
     N = 5000
    
     slope:     -3.5184585909075796
     intercept: 13.765136769385732
     R squared:  0.9996063762935679
    
    -------------------------------------------------------------------------
    
     N = 10000
    
     slope:     -3.525326475682311
     intercept: 15.168079151939518
     R squared:  0.9996381045595585
    
    -------------------------------------------------------------------------


#### Testing phase


```python
print("\n Predicting phase ")
print("\n------------------")
for N in Ns:
    print(f"\n N = {N}\n")
    reg = reg_dict_2[str(N)]
    log_expectations_test = log_expectations_test_dict_2[str(N)]
    print(" R squared on the test set: ", reg.score(log_ms, log_expectations_test))
    print("\n"+73*"-")
```

    
     Predicting phase 
    
    ------------------
    
     N = 1000
    
     R squared on the test set:  0.9993046075393165
    
    -------------------------------------------------------------------------
    
     N = 5000
    
     R squared on the test set:  0.9996495122867071
    
    -------------------------------------------------------------------------
    
     N = 10000
    
     R squared on the test set:  0.9996484001983278
    
    -------------------------------------------------------------------------


## Miniconclusion

Even with a train-test split of 60-40 or 80-20, the relation looks solidly linear. Note one thing though: our variable $x$ vector never changes, it remains constant.

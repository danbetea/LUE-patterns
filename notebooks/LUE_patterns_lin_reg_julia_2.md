```julia
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

We do linear regression (in Julia) on 

$$(\log i, \log E \lambda_i^s)$$

for $i$ in a certain interval like $1, \dots, 10; 1, \dots, \log N$ or more generally $m_0, \dots, m_0 + M - 1$. Here $\lambda_i$ the $i$-th lowest eigenvalue of the LUE-$\alpha$ ensemble and $s$ is a real number, taken negative for convergence. 

We consider several cases:

- $m_0 = 1, M = 15$, $\alpha = 4, s = -2$, $N \in \{1000, 5000, 10000\}$, and 1000 total samples

and we additionally consider two cases for splitting the data into training and testing: 

- first we try 60%-40%, and
- then we try 80%-20%.

**Important remark:** The dependent variable $x = (\log i)_{m_0 \leq i \leq m_0+M}$ is deterministic.

**Some other remarks:**

- array indexing starts at 1 by default in Julia
- the data files, containing iid samples $(\lambda_1 < \dots < \lambda_N)$, are assumed to be in the same directory as the notebook, and be in the correct format
- steps below can be automated; for this exploratory notebook they are not


```julia
# some packages we need
using DataFrames          # for dataframes, surely not needed
using GLM                 # for linear regression
using DelimitedFiles      # for reading a big matrix out of a file easily
using Plots               # for plotting 
using LaTeXStrings        # for LaTeX symbols inside plots
using Statistics          # for the obvious reason
```

### First case: 60-40 split


```julia
alpha_str = "4.00"
alpha = parse(Float64, alpha_str)            # alpha-1 is the Laguerre parameter, alpha > 0
Ns = [1000, 5000, 10000]                     # matrix sizes
num_samples = 1000                           # number of samples, each sample is a line
range_train = 1:600                          # range (samples to include) for training
range_test = 601:1000                        # range (samples to include) for testing
s = -2.0

m0 = 1                                       # where to start with the linear regression
M = 15                                       # we stop at m0 + M

range_ms = m0 : m0 + M - 1                   # the range of data points used
log_ms = [log(m) for m in range_ms]          # the x variable for regression

reg_dict = Dict()
coeff_dict = Dict()

for N in Ns
    println("N = $(N)\n")
    A = readdlm(string("LUE_N_", N,"_alpha_", alpha_str, ".txt"), '\t', Float64, '\n')[1:num_samples, range_ms]
    A_train = A[range_train, :]
    A_test = A[range_test, :]
    log_expectations_train = [log(mean(A_train[:, m - m0 + 1] .^ s)) for m in range_ms]
    log_expectations_test = [log(mean(A_test[:, m - m0 + 1] .^ s)) for m in range_ms]
    
    data = DataFrame(X=log_ms, Y=log_expectations_train)
    reg = lm(@formula(Y ~ X), data)
    reg_dict[string(N)] = reg

    display(reg)
    
    println()
    println("R squared on the training set is: ", r2(reg))
    println()
    
    # compute R^2 on the test set (see e.g. Wikipedia article)
    SS_tot = sum((log_expectations_test .- mean(log_expectations_test)).^2)
    SS_res = sum((log_expectations_test .- (coef(reg)[1] .+ coef(reg)[2] .* log_ms)).^2)
    r2_test = 1 - SS_res/SS_tot
    
    println("R squared on the test set is: ", r2_test)
    println("\n--------------------------------------------------------------------------\n")
    
    coeff_dict[string(N)] = Dict([("b0", coef(reg)[1]), ("b1", coef(reg)[2]), ("r2", r2(reg))])
end
```

    N = 1000
    



    StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}
    
    Y ~ 1 + X
    
    Coefficients:
    ──────────────────────────────────────────────────────────────────────────
                    Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
    ──────────────────────────────────────────────────────────────────────────
    (Intercept)  10.6401    0.0328259   324.14    <1e-26   10.5692    10.711
    X            -3.55653   0.0163515  -217.51    <1e-23   -3.59186   -3.52121
    ──────────────────────────────────────────────────────────────────────────


    
    R squared on the training set is: 0.9997252840928547
    
    R squared on the test set is: 0.9996010133806793
    
    --------------------------------------------------------------------------
    
    N = 5000
    



    StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}
    
    Y ~ 1 + X
    
    Coefficients:
    ──────────────────────────────────────────────────────────────────────────
                    Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
    ──────────────────────────────────────────────────────────────────────────
    (Intercept)  13.7803    0.0365889   376.63    <1e-26   13.7013    13.8594
    X            -3.52498   0.0182259  -193.40    <1e-23   -3.56436   -3.48561
    ──────────────────────────────────────────────────────────────────────────


    
    R squared on the training set is: 0.9996525781321135
    
    R squared on the test set is: 0.9995437008969466
    
    --------------------------------------------------------------------------
    
    N = 10000
    



    StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}
    
    Y ~ 1 + X
    
    Coefficients:
    ──────────────────────────────────────────────────────────────────────────
                    Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
    ──────────────────────────────────────────────────────────────────────────
    (Intercept)  15.2411    0.0304547   500.45    <1e-28   15.1753    15.3069
    X            -3.56084   0.0151703  -234.72    <1e-24   -3.59361   -3.52806
    ──────────────────────────────────────────────────────────────────────────


    
    R squared on the training set is: 0.9997641013695571
    
    R squared on the test set is: 0.9986867445554634
    
    --------------------------------------------------------------------------
    


### Second case: 60-40 split


```julia
alpha_str = "4.00"
alpha = parse(Float64, alpha_str)            # alpha-1 is the Laguerre parameter, alpha > 0
Ns = [1000, 5000, 10000]                     # matrix sizes
num_samples = 1000                           # number of samples, each sample is a line
range_train = 201:1000                       # range (samples to include) for training
range_test = 1:200                           # range (samples to include) for testing
s = -2.0

m0 = 1                                       # where to start with the linear regression
M = 15                                       # we stop at m0 + M

range_ms = m0 : m0 + M - 1                   # the range of data points used
log_ms = [log(m) for m in range_ms]          # the x variable for regression

reg_dict_2 = Dict()
coeff_dict_2 = Dict()

for N in Ns
    println("N = $(N)\n")
    A = readdlm(string("LUE_N_", N,"_alpha_", alpha_str, ".txt"), '\t', Float64, '\n')[1:num_samples, range_ms]
    A_train = A[range_train, :]
    A_test = A[range_test, :]
    log_expectations_train = [log(mean(A_train[:, m - m0 + 1] .^ s)) for m in range_ms]
    log_expectations_test = [log(mean(A_test[:, m - m0 + 1] .^ s)) for m in range_ms]
    
    data = DataFrame(X=log_ms, Y=log_expectations_train)
    reg = lm(@formula(Y ~ X), data)
    reg_dict_2[string(N)] = reg

    display(reg)
    
    println()
    println("R squared on the training set is: ", r2(reg))
    println()
    
    # compute R^2 on the test set (see e.g. Wikipedia article)
    SS_tot = sum((log_expectations_test .- mean(log_expectations_test)).^2)
    SS_res = sum((log_expectations_test .- (coef(reg)[1] .+ coef(reg)[2] .* log_ms)).^2)
    r2_test = 1 - SS_res/SS_tot
    
    println("R squared on the test set is: ", r2_test)
    println("\n--------------------------------------------------------------------------\n")
    
    coeff_dict_2[string(N)] = Dict([("b0", coef(reg)[1]), ("b1", coef(reg)[2]), ("r2", r2(reg))])
end
```

    N = 1000
    



    StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}
    
    Y ~ 1 + X
    
    Coefficients:
    ──────────────────────────────────────────────────────────────────────────
                    Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
    ──────────────────────────────────────────────────────────────────────────
    (Intercept)  10.6371    0.0326739   325.55    <1e-26   10.5665    10.7077
    X            -3.55699   0.0162758  -218.55    <1e-23   -3.59215   -3.52182
    ──────────────────────────────────────────────────────────────────────────


    
    R squared on the training set is: 0.9997278908057867
    
    R squared on the test set is: 0.9993046075393165
    
    --------------------------------------------------------------------------
    
    N = 5000
    



    StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}
    
    Y ~ 1 + X
    
    Coefficients:
    ──────────────────────────────────────────────────────────────────────────
                    Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
    ──────────────────────────────────────────────────────────────────────────
    (Intercept)  13.7651    0.0388747   354.09    <1e-26   13.6812    13.8491
    X            -3.51846   0.0193645  -181.70    <1e-22   -3.56029   -3.47662
    ──────────────────────────────────────────────────────────────────────────


    
    R squared on the training set is: 0.9996063762935677
    
    R squared on the test set is: 0.9996495122867071
    
    --------------------------------------------------------------------------
    
    N = 10000
    



    StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}
    
    Y ~ 1 + X
    
    Coefficients:
    ──────────────────────────────────────────────────────────────────────────
                    Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
    ──────────────────────────────────────────────────────────────────────────
    (Intercept)  15.1681    0.0373472   406.14    <1e-27   15.0874    15.2488
    X            -3.52533   0.0186036  -189.50    <1e-23   -3.56552   -3.48514
    ──────────────────────────────────────────────────────────────────────────


    
    R squared on the training set is: 0.9996381045595585
    
    R squared on the test set is: 0.9996484001983278
    
    --------------------------------------------------------------------------
    


## Miniconclusion

Even with a train-test split of 60-40 or 80-20, the relation looks solidly linear. Note one thing though: our variable $x$ vector never changes, it remains constant.

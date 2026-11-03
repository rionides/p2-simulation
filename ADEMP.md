# ADEMP

#### A: Aims

Reproduce the simulations in Section 4 of Wainwright's *Sharp thresholds for high-dimensional and noisy recovery of sparsity*. Investigate how the results of those simulations change with weaker effects.

The original paper (and so also this simulation study) examines the probability of recoverying the signs of the support of a regression problem as the sample-size parameter θ increases.We also aim to extend (or perhaps challenge) the results of the paper by simulating the same problem with a weaker signal.

We examine these patterns through recovery curves and error counts, without formal significance tests. Finite-sample departures from the predicted curves do not by themselves contradict the asymptotic results.

#### D: Data-generating mechanisms

For each trial, generate a Gaussian linear regression dataset $Y_i=x_i^T\beta^*+\varepsilon_i$, $x_i\overset{\mathrm{iid}}{\sim}N_p(0,\Sigma)$, $\varepsilon_i\overset{\mathrm{iid}}{\sim}N(0,0.5^2)$ where predictors / coefficients / noise are generated independently. Following the paper, use either independent predictors with $\Sigma=I_p$ or Toeplitz predictors with 
$\Sigma_{j\ell}=0.1^{|j-\ell|}$.

Draw the support uniformly from all subsets of size k. Active coefficients have independent, equally likely positive or negative signs. In the reproduction study, all active magnitudes are 1. In the weak-signal study, five randomly selected active coefficients have magnitude 0.25; the rest remain at magnitude 1. Coefficients outside the support are zero.

Use $\alpha$ = 0.4 and $\gamma$ = 0.75, with three sparsity regimes:

| Regime | Active coefficients |
| --- | --- |
| Linear | \(k=\lfloor\alpha p\rfloor\) |
| Sublinear | \(k=\lfloor\alpha p/\log(\alpha p)\rfloor\) |
| Fractional power | \(k=\lfloor\alpha p^\gamma\rfloor\) |

Calculate the sample size, $n=\lfloor2\theta k\log(p-k)\rfloor+k+1$, where we vary $\theta\in\{0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3\}.$ 

**Design Matrix**

Each row below is run in both studies (i.e. the paper simulation and the extension to a weak signal), with both types of covariance matrices and all 12 $\theta$ values. This gives 48 conditions per row and 432 conditions overall. Each condition has 200 trials, so there are 43,200 per study.

| p | Sparsity regime | k |
| ---: | --- | ---: |
| 128 | Linear | 51 |
| 128 | Sublinear | 13 |
| 128 | Fractional power | 15 |
| 256 | Linear | 102 |
| 256 | Sublinear | 22 |
| 256 | Fractional power | 25 |
| 512 | Linear | 204 |
| 512 | Sublinear | 38 |
| 512 | Fractional power | 43 |

Within each study, trials are ordered by covariance (identity, Toeplitz), regime (linear, sublinear, fractional power), increasing p, increasing θ, and repetition. Each trial uses `np.random.default_rng(607 + zero_based_row_number)` and generates its own data (the nature of this study is not such that one could use one simulated dataset as detailed in the spec.)

Both the reproduction and the and weak-signal trials share their predictors, support, signs, and noise draws. The existing random support order determines which five effects are weakened without additional random draws.

#### E : Estimands and targets

For each condition, estimate the probability of exact signed-support recovery:

\[
q=\Pr\{\operatorname{sign}(\widehat\beta)
=\operatorname{sign}(\beta^*)\}.
\]

i.e. a trial succeeds if Lasso selects exactly the true active variables and gives each coefficient the correct sign. The probability of success in each condition is averaged over the results from 200 independently generated datasets. 


#### M : Methods

Fit the same Lasso estimator in every condition:  $\widehat\beta=\arg\min_b\left\{\frac{\|Y-Xb\|_2^2}{2n}+\rho_n\|b\|_1\right\}$,
  $\rho_n=\sqrt{\frac{\log(p-k)\log(k)}{n}}$.

 There is no cross-validation or comparison with other estimators. Computationally, this is done with scikit-learn's `Lasso` with `alpha=rho_n`, no intercept, cyclic coordinate descent, and tolerance `1e-7`. Predictors are neither centered nor standardized. 

For evaluation, coefficients with absolute magnitude at most `1e-7` count as zero. A convergence warning stops the run, and incomplete runs are not summarized or counted as failures.



#### P : Performance measures
As stated above, for R = 200 trials per condition, estimate recovery probability and its Monte Carlo standard error as

\[
\widehat q=\frac{1}{R}\sum_{r=1}^R I_r,\qquad
\widehat{\mathrm{MCSE}}(\widehat q)
=\sqrt{\frac{\widehat q(1-\widehat q)}{R}},
\]

where $I_r$ is 1 for exact signed-support recovery, 0 else. The figures also show 95% pointwise Wilson intervals. 

Also, record false positives (selected inactive variables), false negatives (active variables estimated as zero), and sign errors (incorrect signs among selected active variables).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)
from scipy.optimize import fsolve
from scipy.special import erf
from scipy.integrate import quad
from functools import partial

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from statsmodels.stats.multitest import multipletests


from crepes import ConformalRegressor, ConformalPredictiveSystem

from crepes.fillings import (sigma_variance, 
                            sigma_variance_oob,
                            sigma_knn,
                            binning)

""" 
In this document we will show how to use the conformal regressor to detect distribution shifts
    in the data. We will use the house sales dataset from OpenML. We will first train a random forest
    regressor on the data, and then use the conformal regressor to detect distribution shifts.
"""

# ----------------------------- PART 1: Creating the conformal regressor ---------------------------------------

# First we load the dataset
dataset = fetch_openml(name="house_sales",version=3)

X = dataset.data.values.astype(float)
y = dataset.target.values.astype(float)
print('Before normalization range:',y.min(), y.max())

# and also normalize it such that the y values are in the range [0,1]
y = (np.tanh(np.array([(y[i]-y.min())/(y.max()-y.min()) for i in range(len(y))])*2-1)+1)/2.
print('After normalization range:',y.min(), y.max())


# Then we split the data such that we can later on introduce a distribution shift
X_main, X_shift, y_main, y_shift = X[(4e4 >= X[:, 3])] , X[(4e4 < X[:, 3])] , y[(4e4 >= X[:, 3])] , y[(4e4 < X[:, 3])]

# We want to have an even number of samples, to create tuples
if len(X_shift) % 2 == 1:
    X_shift = X_shift[:-1]
    y_shift = y_shift[:-1]
if len(X_main) % 2 == 1:
    X_main = X_main[:-1]
    y_main = y_main[:-1]

## The input will be two copies of X, one for each house
print('Before creating tuples',X_main.shape, X_shift.shape)
X_main = X_main.reshape(X_main.shape[0]//2,-1)
X_shift = X_shift.reshape(X_shift.shape[0]//2,-1)
print('After creating tuples',X_main.shape, X_shift.shape)

## We want to estimate the difference between the two y values
y_main = y_main.reshape(y_main.shape[0]//2,-1)
y_main = y_main[:,0] - y_main[:,1]
print(y_main.shape)

## and also in the shifted dataset
y_shift = y_shift.reshape(y_shift.shape[0]//2,-1)
y_shift = y_shift[:,0] - y_shift[:,1]
print(y_shift.shape)

# We split the data into a training and a test set # todo: this spliting will have to be changed
X_train, X_test, y_train, y_test = train_test_split(X_main, y_main, test_size=0.5)
X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train,
                                                            test_size=0.25)

# Then we train a random forest regressor on the training set
random_forest_model = RandomForestRegressor(n_jobs=-1, n_estimators=500) 
random_forest_model.fit(X_prop_train, y_prop_train)

# and fit a conformal regressor on the residuals, with confidence = 1- delta
delta = 0.01

"""Here we define the first conformal model, cr_std, whose conformal interval sizes are all equal."""
cr_std = ConformalRegressor()
y_hat_cal = random_forest_model.predict(X_cal)
residuals_cal = y_cal - y_hat_cal
# and fit it to the residuals
cr_std.fit(residuals=residuals_cal)

# Using such model we can now predict the residuals on the test set
y_hat_test = random_forest_model.predict(X_test)
intervals_std = cr_std.predict(y_hat=y_hat_test, confidence=1-delta)

# We can use a measure of difficulty to make smaller or larger intervals. 
# To do so we can use the residuals of the k nearest neighbours.
"""Second conformal model, cr_norm_knn, with conformal intervals sizes adjusted for difficulty with knn."""
sigmas_cal_knn = sigma_knn(X=X_cal, residuals=residuals_cal)
cr_norm_knn = ConformalRegressor()
cr_norm_knn.fit(residuals=residuals_cal, sigmas=sigmas_cal_knn)

# We can do that for the test set too
sigmas_test_knn = sigma_knn(X=X_cal, residuals=residuals_cal, X_test=X_test)
intervals_norm_knn = cr_norm_knn.predict(y_hat=y_hat_test, 
                                        sigmas=sigmas_test_knn,
                                        y_min=0, y_max=1)

# We can also improve the conformal predictor by binnin
"""Third conformal model, cr_mond, with conformal intervals sizes adjusted for difficulty with binning."""
bins_cal, bin_thresholds = binning(values=sigmas_cal_knn, bins=20)
cr_mond = ConformalRegressor()
cr_mond.fit(residuals=residuals_cal, bins=bins_cal)
bins_test = binning(values=sigmas_test_knn, bins=bin_thresholds)

intervals_mond = cr_mond.predict(y_hat=y_hat_test, bins=bins_test, 
                                        y_min=0, y_max=1)

# So far we have three prediction intervals based on the standard, normalized and binned models
prediction_intervals = {
    "Std CR":intervals_std,
    "Norm CR knn":intervals_norm_knn,
    "Mond CR":intervals_mond
}

# We can compute the coverage, loss and interval sizes for each model; the loss function will be equal to the 
# quadratic loss of the prediction interval; assuming it follows a gaussian distribution

"""
We want to compute sigma such that
$$\frac{1}{2}(erf(interval_high-mean/(sqrt(2)*sigma))-erf(interval_low-mean/(sqrt(2)*sigma))) = 1-delta,$$
and then compute
$$ loss = int_l^h e^{-x^2/2sigma^2} x^2 dx.$$
"""

def compute_sigma(delta, initial_interval, mean):
    def cdf(sigma_):
        return 0.5*(erf((initial_interval[1]-mean)/(np.sqrt(2)*sigma_))-erf((initial_interval[0]-mean)/(np.sqrt(2)*sigma_))) - (1-delta)

    sigma = fsolve(cdf, (initial_interval[1]-initial_interval[0])/2)[0]
    return sigma

def compute_loss(y_hat, y_gt, sigma, interval_alpha):
    def normal_loss(x, y_gt, mean, sigma):
        return np.exp(-((x-mean)**2)/(2*sigma**2))*(x-y_gt)**2

    normal_l = partial(normal_loss, mean=y_hat, sigma = sigma, y_gt=y_gt)

    loss = quad(normal_l, interval_alpha[0], interval_alpha[1])[0]
    return loss

def quick_loss(delta, initial_interval, y_hat, y_gt):
    sigma = compute_sigma(delta, initial_interval, y_hat)
    return compute_loss(y_hat, y_gt, sigma, initial_interval)

coverages = []
losses = [] 
mean_sizes = []
median_sizes = []

for name in prediction_intervals.keys():
    intervals = prediction_intervals[name]
    coverages.append(np.sum([1 if (y_test[i]>=intervals[i,0] and 
                                    y_test[i]<=intervals[i,1]) else 0
                            for i in range(len(y_test))])/len(y_test))
    mean_sizes.append((intervals[:,1]-intervals[:,0]).mean())
    median_sizes.append(np.median((intervals[:,1]-intervals[:,0])))
    losses.append(np.mean([quick_loss(delta, intervals[i], y_hat_test[i], y_test[i]) for i in range(len(y_test))]))

pred_int_df = pd.DataFrame({"Coverage":coverages,
                            "Loss":losses,
                            "Mean size":mean_sizes, 
                            "Median size":median_sizes},
                            index=list(prediction_intervals.keys()))

pred_int_df.loc["Mean"] = [pred_int_df["Coverage"].mean(), 
                            pred_int_df["Loss"].mean(),
                            pred_int_df["Mean size"].mean(),
                            pred_int_df["Median size"].mean()]

bins_test = binning(values=y_hat_test, bins=bin_thresholds)


# We can also compute p-values from this
cps_std = ConformalPredictiveSystem().fit(residuals=residuals_cal)
cps_norm = ConformalPredictiveSystem().fit(residuals=residuals_cal,
                                        sigmas=sigmas_cal_knn)
bins_cal, bin_thresholds = binning(values=y_hat_cal, bins=5)

cps_mond_std = ConformalPredictiveSystem().fit(residuals=residuals_cal,
                                            bins=bins_cal)

cps_mond_norm = ConformalPredictiveSystem().fit(residuals=residuals_cal,
                                                    sigmas=sigmas_cal_knn,
                                                    bins=bins_cal)
all_cps = {"Std CPS":cps_std,
           "Norm CPS":cps_norm,
           "Mond CPS":cps_mond_std,
           "Mond norm CPS":cps_mond_norm,
          }
p_values = all_cps[name].predict(y_hat=y_hat_test, 
                                sigmas=sigmas_test_knn, 
                                bins=bins_test, 
                                y=y_test)[:,0]






# ----------------------------- PART 2: Learn the test procedure ---------------------------------------

# for each interval we can compute the sigma that corresponds to the interval
def confidence_interval(mean, initial_interval, initial_delta, alpha):
    """
    We want to compute the conformal integral such that the probability density is larger than alpha
    """
    def func(sigma):
        return 0.5*(erf((initial_interval[1]-mean)/(np.sqrt(2)*sigma))-erf((initial_interval[0]-mean)/(np.sqrt(2)*sigma))) - (1-initial_delta)

    sigma = fsolve(func, (initial_interval[1]-initial_interval[0])/2)[0]

    def normal_distribution(x, mean, sigma):
        return np.exp(-((x-mean)**2)/(2*sigma**2))

    interval_alpha = fsolve(lambda x: normal_distribution(x, mean, sigma) - alpha, x0 = initial_interval)

    return interval_alpha, sigma

# we want to check what values of alpha make the loss controlled at level lambda_
lambda_ = 0.5
def pvalue(y_gt, y_hat, initial_interval, alpha, initial_delta):

    # for each y compute confidence interval first
    alpha = np.linspace(0.01, 0.99, 100)
    interval_alpha, sigma = confidence_interval(y_hat, initial_interval, initial_delta=initial_delta, alpha=alpha)

    loss = compute_loss(y_hat, y_gt, sigma, interval_alpha)

    pvalues = np.exp(-2*len(y_hat)*(lambda_ - loss)**2)

    # Step 2: Family-wise error correction
    reject, pvals_corrected, _, bonferroni_delta = multipletests(pvalues, delta, method = 'bonferroni')
    




# ----------------------------- PART 3: Full conformal prediction ---------------------------------------


# We can create a few datasets to train our model on, and as a consequence perform conformal prediction

lambdas_i
for i in range(1000):

    j = np.random.randint(0, len(X_train))
    
    X_train = np.concatenate([X_train[:j],[new_x], X_train[j+1:]])
    y_train = np.concatenate([y_train[:j],[new_y], y_train[j+1:]])

    # We can now fit a model to the data
    random_forest_model = RandomForestRegressor(n_estimators=100)
    random_forest_model.fit(X_train, y_train)

    # We can now compute the residuals
    y_hat_train = random_forest_model.predict(X_train)
    residuals_cal = y_train - y_hat_train

    # We can now fit a model to the residuals
    cr_std = ConformalRegressor()
    cr_std.fit(residuals=residuals_cal)

    # Compute confidence intervals on sample X_train[j]
    y_hat_j = random_forest_model.predict(X_test[j])
    interval_j_std = cr_std.predict(y_hat=y_hat_j, confidence=1-delta)

    # Compute loss
    sigma = compute_sigma(delta = delta, initial_interval = interval_j_std, mean = y_hat_j)
    interval_alpha = confidence_interval(delta = delta, initial_interval = interval_j_std, mean = y_hat_j)
    lambda_i = compute_loss(y_hat_j, y_test[j], sigma, interval_alpha)

    # Compute lambda_i
    lambdas_i.append(lambda_i)









    cr_std = ConformalRegressor()
    y_hat_train = random_forest_model.predict(X_train)
    residuals_cal = y_train - y_hat_train
    # and fit it to the residuals
    cr_std.fit(residuals=residuals_cal)

    # Using such model we can now predict the residuals on the test set
    y_hat_test = random_forest_model.predict(X_test)
    intervals_std = cr_std.predict(y_hat=y_hat_test, confidence=1-delta)




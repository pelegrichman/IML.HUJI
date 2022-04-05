from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import pandas as pn
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    uni_Gaussian_estimator = UnivariateGaussian()
    uni_Gaussian_estimator.fit(X)
    print('({mean},{var})'.format(mean=uni_Gaussian_estimator.mu_, var=uni_Gaussian_estimator.var_))

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, num=100).astype(int)
    estimated_mean = []
    for m in ms:
        X = np.random.normal(10, 1, m)
        uni_Gaussian_estimator.fit(X)
        estimated_mean.append(abs(uni_Gaussian_estimator.mu_ - 10))


    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{Q2 -  absolute distance between the estimated- and true value of the expectation,as a function of the sample size.}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$|\hat\mu - \mu|$",
                  height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = uni_Gaussian_estimator.pdf(X)
    go.Figure([go.Scatter(x=X.flatten(), y=pdfs, mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{Q3- pdf of x_i as a function of x_i value}$",
                  xaxis_title="$X\\text{ - values of samples ordered}$",
                  yaxis_title="r$pdf$",
                  height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1.0, 0.2, 0.0, 0.5],
                      [0.2, 2.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.5, 0.0, 0.0, 1.0]])

    X = np.random.multivariate_normal(mu, cov, 1000)
    multi_Gaussian_estimator = MultivariateGaussian()
    multi_Gaussian_estimator.fit(X)
    print(multi_Gaussian_estimator.mu_)
    print(multi_Gaussian_estimator.cov_)

    # Question 5 - Likelihood evaluation
    F1 = np.linspace(-10, 10, 200)
    F3 = np.linspace(-10, 10, 200)
    log_likelihood_models = np.zeros(shape=(200, 200))
    max_log_likelihood_val = np.NINF
    max_log_likelihood_args = ()

    for i, f1 in enumerate(F1):
        for j, f3 in enumerate(F3):
            mu = np.array([f1, 0, f3, 0])
            log_likelihood_models[i, j] = MultivariateGaussian.log_likelihood(mu, cov, X)

            if log_likelihood_models[i, j] > max_log_likelihood_val:
                max_log_likelihood_val = log_likelihood_models[i, j]
                max_log_likelihood_args = (f1, f3)

    fig = px.imshow(log_likelihood_models, x=F1, y=F3, labels=dict(x='F1 samples values', y='F3 sample values',
                                                                   color='log likelihood values'),
                                                                   title='log likelihood value for different models - [f1, 0, f3, 0]')
    fig.show()

    # Question 6 - Maximum likelihood
    print(round(max_log_likelihood_args[0], 4), round(max_log_likelihood_args[1], 4))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics.loss_functions import mean_square_error
from typing import NoReturn, Tuple, Dict
import numpy as np
import pandas as pd
from pandas import DataFrame
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    hp_df: DataFrame = pd.read_csv(filename)

    # Drop null values and duplicate samples
    hp_df.dropna(inplace=True)
    hp_df.drop_duplicates(inplace=True)

    # Edit features to match thier true range.
    hp_df['date'] = hp_df['date'].str[:4]
    hp_df = hp_df[hp_df['bedrooms'].isin(range(20))]
    hp_df = hp_df[hp_df['bathrooms'].isin(range(9))]
    hp_df = hp_df[hp_df['waterfront'].isin(range(2))]
    hp_df = hp_df[hp_df['view'].isin(range(5))]
    hp_df = hp_df[hp_df['condition'].isin(range(1, 6))]
    hp_df = hp_df[hp_df['grade'].isin(range(1, 14))]

    # For categorical features create dummies.
    hp_df = pd.get_dummies(hp_df,
                           columns=['zipcode', 'yr_built', 'yr_renovated', 'lat', 'long'],
                           )

    # Drop uncorrelated features.
    hp_df.drop(columns=['id', 'date'], inplace=True)
    return hp_df.drop(columns=['price']), hp_df['price']


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for feature_name in X:
        feature: pd.Series = X[feature_name]
        cov: np.ndarray = np.cov(feature, y)
        p_corr: np.ndarray = cov / (feature.std() * y.std())

        data = DataFrame({feature_name: feature, 'response': y})
        # layout1 = dict(title=dict(text="A Figure Specified By A Graph Object"))
        fig = px.scatter(data, x=feature_name, y='response',
                         title=f"Pearson Correlation between feature {feature_name} "
                               f"and the response vector Prices: <br>"
                               f"<b> with value {p_corr[0, 1].round(5)} <b>")

        fig.update_layout(xaxis_title=f"feature {feature_name}", yaxis_title="Response vector - Prices")
        fig.write_image(os.path.join(output_path + f'feature_{feature_name}_corr.png'))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    observations, response = load_data('..\\datasets\\house_prices.csv')
    # print(observations)
    # print('\n --------------------- \n')
    # print(response)

    # Question 2 - Feature evaluation with respect to response
    out_dir_path = 'C:\\Users\\peleg\\Desktop\\university\\year C\\B\\IML\\Exercises\\Ex2\\output Ex2\\'
    feature_evaluation(observations, response, output_path=out_dir_path)

    # Question 3 - Split samples into training- and testing sets.
    # x_train, y_train, x_test, y_test = split_train_test(observations, response, train_proportion=0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # mse: Dict[int, float] = {}
    # for p in range(10, 101):
    #     px_train = x_train.sample(frac=float(p/100)).to_numpy()
    #     py_train = y_train.sample(frac=float(p / 100)).to_numpy()
    #     model = LinearRegression(include_intercept=True).fit(px_train, py_train)
    #
    #     y_predict = model.predict(px_train)
    #     mse[p] = mean_square_error(py_train, y_predict)

    # TODO: Plot!.
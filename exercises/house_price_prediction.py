import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import kaleido

pio.templates.default = "simple_white"
TEST_FRAC = 0.25



def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.g
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    house_data = pd.read_csv(filename)

    # drop corrupted data
    house_data.dropna(inplace=True)
    house_data.drop_duplicates(inplace=True)

    # remove the location and id columns
    house_data.drop(columns=['lat', 'long', 'id'], inplace=True)

    # remove negative and illegal values
    for column in ['price', 'sqft_lot', 'sqft_lot15', 'floors', 'yr_built']:
        house_data = house_data[house_data[column] > 0]
    for column in [('waterfront', range(2)), ('view', range(5)),
                   ('condition', range(1, 6)), ('grade', range(1, 14))]:
        house_data = house_data[house_data[column[0]].isin(column[1])]

    # categorize the columns 'zipcode', 'date', 'yr_built', 'yr_renovated'
    house_data['zipcode'] = house_data['zipcode'].astype(str).str[:3]
    house_data = pd.get_dummies(house_data, columns=['zipcode'],
                                prefix='zipcode_area')
    house_data['date'] = house_data['date'].str[:4]
    house_data = pd.get_dummies(house_data, columns=['date'])
    house_data['yr_built'] = house_data['yr_built'].astype(str).str[:2]
    house_data = pd.get_dummies(house_data, columns=['yr_built'])
    house_data['yr_renovated'] = house_data['yr_renovated'].astype(str).str[:2]
    house_data = pd.get_dummies(house_data, columns=['yr_renovated'])

    # is_basement flag
    house_data['is_basement'] = (house_data['sqft_basement'] >= 1).astype(int)


    return house_data['price'], house_data.drop(columns=['price'])


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
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
    X.drop(columns=['yr_renovated_20', 'yr_renovated_19', 'yr_renovated_0.',
                    'yr_built_20',
                    'yr_built_19', 'date_2015', 'date_2014',
                    'zipcode_area_981', 'zipcode_area_980'], inplace=True)
    for column in X:
        pearson = np.cov(X[column], y) / (np.std(X[column]) * np.std(y))
        df = pd.DataFrame(
            {'intercept': np.ones(X.shape[0]), column: X[column].to_numpy()})
        w = np.linalg.pinv(df) @ y
        y_predict = w[0] + w[1] * X[column].to_numpy()
        fig = px.scatter(x=X[column], y=y)
        fig.update_layout(title=f"feature values of {column} column against the response values\n"
            f"Pearson Correlation {pearson[0][1]}",
                          xaxis_title=f'{column}',
                          yaxis_title='price')
        fig.show()
        fig.write_image(os.path.join(output_path, column + ".png"))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    y_response, matrix_data_frame = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(matrix_data_frame, y_response)

    # # Question 3 - Split samples into training- and testing sets.
    x_train, y_train, x_test, y_test = split_train_test(matrix_data_frame, y_response, n_samples=int)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    list_of_means_per_p = []
    list_of_std_per_p = []
    percent = range(10, 101)
    sample_num = range(0, 10)
    for p in percent:
        mean_loss_by_p = []
        for i in sample_num:
            train_x_matrix = x_train.sample(frac=p / 100)
            train_y_matrix = y_train[train_x_matrix.index]

            linear_regression = LinearRegression()
            linear_regression.fit(train_x_matrix.to_numpy(), train_y_matrix.to_numpy())
            mean_loss_by_p.append(
                linear_regression.loss(x_test.to_numpy(), y_test.to_numpy()))
        list_of_means_per_p.append(np.mean(mean_loss_by_p))
        list_of_std_per_p.append(np.std(mean_loss_by_p))
    x = np.arange(10, 101)
    list_of_means_per_p = np.array(list_of_means_per_p)
    list_of_std_per_p = np.array(list_of_std_per_p)
    go.Figure([go.Scatter(x=x, y=list_of_means_per_p, mode="markers+lines",
                          name="MSE",
                          line=dict(dash="dash"),
                          marker=dict(color="green", opacity=.9)),


               go.Scatter(x=x, y=list_of_means_per_p - 2 *
               list_of_std_per_p, fill=None,
                          mode="lines",
                          line=dict(color="lightgrey"), showlegend=False),

               go.Scatter(x=x, y=list_of_means_per_p  + 2 * list_of_std_per_p,
               fill='tonexty',
                          mode="lines",
                          line=dict(color="lightgrey"),
                          showlegend=False)]).show()


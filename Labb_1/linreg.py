import numpy as np
from scipy.stats import t, f, pearsonr

def aggregate(df):
    grouped = df.groupby(['longitude', 'latitude'])

    mean_cols = [
        'housing_median_age',
        'median_income',
        'median_house_value'
    ]

    sum_cols = [
        'total_rooms',
        'total_bedrooms',
        'population',
        'households'
    ]

    ocean = grouped['ocean_proximity'].first()

    aggregated = (
        grouped[mean_cols].mean()
        .join(grouped[sum_cols].sum())
        .join(ocean)
        .reset_index()
    )

    lat0 = aggregated['latitude'].mean()
    lon0 = aggregated['longitude'].mean()

    aggregated['distance_to_centroid'] = np.sqrt(
        (aggregated['latitude'] - lat0)**2 +
        (aggregated['longitude'] - lon0)**2
    )

    return aggregated

def build_X_Y(df, feat_cols, target_col=None):

    n = df.shape[0]

    X = df[feat_cols].to_numpy(dtype=float)
    Y = df[target_col].to_numpy(dtype=float)

    X = np.column_stack((np.ones(n), X))

    return X, Y

class LinearRegression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.d = X.shape[1] - 1

        self._fit()
    def _fit(self):
        XtX = self.X.T @ self.X

        XtX_inv = np.linalg.inv(XtX)

        XtY = self.X.T @ self.Y

        self.b_est = XtX_inv @ XtY

        self.Y_est = self.X @ self.b_est

        self.residuals = self.Y - self.Y_est

        self.SSE = np.sum(self.residuals**2)

        self.variance = self.SSE / (self.n - self.d - 1)

        self.STD = np.sqrt(self.variance)

        self.RMSE = np.sqrt(self.SSE / self.n)

        self.C = XtX_inv * self.variance

        self.t_stats = self.b_est / np.sqrt(np.diag(self.C))

        self.p_values = 2 * t.sf(np.abs(self.t_stats), df=self.n - self.d - 1)

        self.Y_mean = np.mean(self.Y)

        self.Syy = np.sum((self.Y - self.Y_mean)**2)

        self.SSR = self.Syy - self.SSE

        self.F_stat = (self.SSR / self.d) / self.variance

        self.F_p_value = f.sf(self.F_stat, self.d, self.n - self.d - 1)

        self.R2 = self.SSR / self.Syy

    def pearson_numeric(self, X_full):
        n_vars = X_full.shape[1]
        results = []

        for i in range(n_vars):
            for j in range(i, n_vars):
                r, _ = pearsonr(X_full[:, i], X_full[:, j])
                results.append((i, j, r))

        return results

    def pearson_categorical_numeric(self, X_cat, X_num):
        results = []

        for i in range(X_cat.shape[1]):
            for j in range(X_num.shape[1]):
                r, _ = pearsonr(X_cat[:, i], X_num[:, j])
                results.append((i, j, r))

        return results

    def confidence_intervals(self, alpha):
        t_val = t.ppf(1 - alpha/2, df=self.n - self.d - 1)
        lower = self.b_est - t_val * np.sqrt(np.diag(self.C))
        upper = self.b_est + t_val * np.sqrt(np.diag(self.C))
        return np.column_stack((lower, upper))



import numpy as np
import pandas as pd
from scipy.stats import t, f, pearsonr

def aggregate(df):
    """
    Aggregerar housing dataframen utifrån (longitude, latitude).

    Grupper som medelvärde räknas ut för:
        housing_median_age
        median_income
        median_house_value

    Grupper som summan röknas ut för:
        total_rooms
        total_bedrooms
        population
        households

    Behåller oförändrat:
        ocean_proximity (första värdet per grupp , men det ändrar inte för någon)
    """
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
    return aggregated

def prepare_features(df, cat_cols=None, drop_first=True):
    """
    One-hot encodar angivna kategoriska kolumner och returnerar 
    dataframe redo för regression tillsammans med lista på feature-namn.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame som ska transformerad
    cat_cols : list of str
        Kolumner som ska one-hot encodas
    drop_first : bool, default True
        Drop first category för att undvika multikollinearitet
    
    Returns
    -------
    df_prepared : pd.DataFrame
        DataFrame med en-hot encodade kolumner
    feat_cols : list of str
        Lista på kolumnnamn som ska användas som X
    """

    df_prepared = df.copy()

    if cat_cols is not None:
        df_prepared = pd.get_dummies(df_prepared, columns=cat_cols, drop_first=drop_first)
    
    feat_cols = [col for col in df_prepared.columns if col not in ['median_house_value']]
    
    return df_prepared, feat_cols

def build_X_Y(df, feat_cols, target_col='median_house_value'):
    """
    Bygger X- och Y-matriser för regression och lägger till intercept.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame med data
    feat_cols : list of str
        Kolumner som ska vara regressorer (X)
    target_col : str, default 'median_house_value'
        Målvariabeln (Y)

    Returns
    -------
    X : np.ndarray, shape (n, d+1)
        Designmatris med intercept (första kolumnen är 1)
    Y : np.ndarray, shape (n,)
        Målvariabel
    """
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
        """
        Beräknar OLS-koefficienter (b_est) för modellen.

        1. XtX = Transponera och multiplicera X

        2. Invertera XtX

        3. XtY = Multiplicera transponerad X med Y

        4. Beräkna beta genom att multiplicera inverterad XtX med XtY
        """
        XtX = self.X.T @ self.X

        XtX_inv = np.linalg.inv(XtX)

        XtY = self.X.T @ self.Y

        self.b_est = XtX_inv @ XtY

        """
        Beräknar grundläggande statistiker.

        Y_est = prediktioner, modellens uppskattade värden för varje X

        residuals = vad som blev kvar, d.s.v. hur mycket modellen missade med

        SSE = summan av de total felen kvadrerat

        variance = varians, hur stora residualerna är genomsnittligen justerat för antalet variabler i modellen

        STD = standardavvikelse, hur stora felprediktionerna typiskt är i samma enhet som Y

        RMSE = Root Mean Squared Error, en annan metod för att mäta hur stora felprediktionerna typiskt är i samma enhet som Y

        C = varians-kovariansmatris, hur osäkra modellens koefficienter är

        t_stats = signifikans för varje koefficient

        p_values = sannolikheten att koefficienten egentligen är 0; lågt värde = signifikant
        """

        self.Y_est = self.X @ self.b_est

        self.residuals = self.Y - self.Y_est

        self.SSE = np.sum(self.residuals**2)

        self.variance = self.SSE / (self.n - self.d - 1)

        self.STD = np.sqrt(self.variance)

        self.RMSE = np.sqrt(self.SSE / self.n)

        self.C = XtX_inv * self.variance

        self.t_stats = self.b_est / np.sqrt(np.diag(self.C))

        self.p_values = 2 * t.sf(np.abs(self.t_stats), df=self.n - self.d - 1)
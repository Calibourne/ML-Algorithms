from pandas import DataFrame

class PredictionModel:
    def fit(self, data: DataFrame):
        """
        Fit the model to the data

        Args:
            data (DataFrame): the data to be fitted
        """
        raise NotImplementedError
    
    def predict(self, data: DataFrame):
        """
        Predict the data

        Args:
            data (DataFrame): the data to be predicted

        Returns:
            DataFrame: the predicted data
        """
        raise NotImplementedError
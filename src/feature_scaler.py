from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, features):
        return self.scaler.fit_transform(features)

    def transform(self, features):
        return self.scaler.transform(features)

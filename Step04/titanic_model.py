import typer
import numpy as np
import pandas as pd
from collections import Counter
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix


class TitanicModelCreator:
    def __init__(self):
        pass

    def run(self):
        engine = create_engine('sqlite:///../data/titanic.db')
        sqlite_connection = engine.connect()
        pd.read_sql(
            'SELECT * FROM sqlite_schema WHERE type="table"', con=sqlite_connection
        )
        np.random.seed(42)

        df = pd.read_sql('SELECT * FROM tbl_passengers', con=sqlite_connection)

        targets = pd.read_sql('SELECT * FROM tbl_targets', con=sqlite_connection)

        # df, targets = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

        # parch = Parents/Children, sibsp = Siblings/Spouses
        df['family_size'] = df['parch'] + df['sibsp']
        df['is_alone'] = [
            1 if family_size == 1 else 0 for family_size in df['family_size']
        ]

        df['title'] = [name.split(',')[1].split('.')[0].strip() for name in df['name']]
        rare_titles = {k for k, v in Counter(df['title']).items() if v < 10}
        df['title'] = [
            'rare' if title in rare_titles else title for title in df['title']
        ]

        df = df[
            [
                'pclass',
                'sex',
                'age',
                'ticket',
                'family_size',
                'fare',
                'embarked',
                'is_alone',
                'title',
            ]
        ]

        targets = [int(v) for v in targets['is_survived']]
        X_train, X_test, y_train, y_test = train_test_split(
            df, targets, stratify=targets, test_size=0.2
        )

        X_train_categorical = X_train[
            ['embarked', 'sex', 'pclass', 'title', 'is_alone']
        ]
        X_test_categorical = X_test[['embarked', 'sex', 'pclass', 'title', 'is_alone']]

        oneHotEncoder = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(
            X_train_categorical
        )
        X_train_categorical_one_hot = oneHotEncoder.transform(X_train_categorical)
        X_test_categorical_one_hot = oneHotEncoder.transform(X_test_categorical)

        X_train_numerical = X_train[['age', 'fare', 'family_size']]
        X_test_numerical = X_test[['age', 'fare', 'family_size']]
        knnImputer = KNNImputer(n_neighbors=5).fit(X_train_numerical)
        X_train_numerical_imputed = knnImputer.transform(X_train_numerical)
        X_test_numerical_imputed = knnImputer.transform(X_test_numerical)

        robustScaler = RobustScaler().fit(X_train_numerical_imputed)
        X_train_numerical_imputed_scaled = robustScaler.transform(
            X_train_numerical_imputed
        )
        X_test_numerical_imputed_scaled = robustScaler.transform(
            X_test_numerical_imputed
        )

        X_train_processed = np.hstack(
            (X_train_categorical_one_hot, X_train_numerical_imputed_scaled)
        )
        X_test_processed = np.hstack(
            (X_test_categorical_one_hot, X_test_numerical_imputed_scaled)
        )

        model = LogisticRegression(random_state=0).fit(X_train_processed, y_train)
        y_train_estimation = model.predict(X_train_processed)
        y_test_estimation = model.predict(X_test_processed)

        cm_train = confusion_matrix(y_train, y_train_estimation)

        cm_test = confusion_matrix(y_test, y_test_estimation)

        print('cm_train', cm_train)
        print('cm_test', cm_test)


def main(param: str = 'pass'):
    titanicModelCreator = TitanicModelCreator()
    titanicModelCreator.run()


if __name__ == "__main__":
    typer.run(main)

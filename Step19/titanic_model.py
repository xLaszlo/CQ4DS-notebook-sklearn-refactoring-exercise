import os
import pickle
import typer
import numpy as np
import pandas as pd
from pydantic import BaseModel
from collections import Counter
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix


RARE_TITLES = {
    'Capt',
    'Col',
    'Don',
    'Dona',
    'Dr',
    'Jonkheer',
    'Lady',
    'Major',
    'Mlle',
    'Mme',
    'Ms',
    'Rev',
    'Sir',
    'the Countess'
}


class Passenger(BaseModel):
    pid: int
    pclass: int
    sex: str
    age: float
    family_size: int
    fare: float
    embarked: str
    is_alone: int
    title: str
    is_survived: int
    

def do_test(filename, data):
    if not os.path.isfile(filename):
        pickle.dump(data, open(filename, 'wb'))
    truth = pickle.load(open(filename, 'rb'))
    try:
        np.testing.assert_almost_equal(data, truth)
        print(f'{filename} test passed')
    except AssertionError as ex:
        print(f'{filename} test failed {ex}')


def do_pandas_test(filename, data):
    if not os.path.isfile(filename):
        data.to_pickle(filename)
    truth = pd.read_pickle(filename)
    try:
        pd.testing.assert_frame_equal(data, truth)
        print(f'{filename} pandas test passed')
    except AssertionError as ex:
        print(f'{filename} pandas test failed {ex}')


class SqlLoader:

    def __init__(self, connectionString):
        engine = create_engine(connectionString)
        self.connection = engine.connect()

    def get_passengers(self):
        query = """
            SELECT 
                tbl_passengers.pid,
                tbl_passengers.pclass,
                tbl_passengers.sex,
                tbl_passengers.age,
                tbl_passengers.parch,
                tbl_passengers.sibsp,
                tbl_passengers.fare,
                tbl_passengers.embarked,
                tbl_passengers.name,
                tbl_targets.is_survived 
            FROM 
                tbl_passengers 
            JOIN 
                tbl_targets 
            ON 
                tbl_passengers.pid=tbl_targets.pid
        """
        return pd.read_sql(query, con=self.connection)


class TestLoader:

    def __init__(self, passengers_filename, realLoader):
        self.passengers_filename = passengers_filename
        self.realLoader = realLoader
        if not os.path.isfile(self.passengers_filename):
            df = self.realLoader.get_passengers()
            df.to_pickle(self.passengers_filename)

    def get_passengers(self):
        return pd.read_pickle(self.passengers_filename)

class PassengerLoader:

    def __init__(self, loader, rare_titles=None):
        self.loader = loader
        self.rare_titles = rare_titles

    def get_passengers(self):
        passengers = []
        for data in self.loader.get_passengers().itertuples():
            # parch = Parents/Children, sibsp = Siblings/Spouses
            family_size = int(data.parch + data.sibsp)
            # Allen, Miss. Elisabeth Walton
            title = data.name.split(',')[1].split('.')[0].strip()
            passenger = Passenger(
                pid=int(data.pid),
                pclass=int(data.pclass),
                sex=str(data.sex),
                age=float(data.age),
                family_size=family_size,
                fare=float(data.fare),
                embarked=str(data.embarked),
                is_alone= 1 if family_size==1 else 0,
                title='rare' if title in self.rare_titles else title,
                is_survived=int(data.is_survived)
            )
            passengers.append(passenger)
        return passengers


class TitanicModel:

    def __init__(self):
        self.trained = False
        self.oneHotEncoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.knnImputer = KNNImputer(n_neighbors=5)
        self.robustScaler = RobustScaler()
        self.predictor = LogisticRegression(random_state=0)

    def process_inputs(self, passengers, pids):
        passengers_map = {p.pid: p for p in passengers}
        data = pd.DataFrame([passengers_map[pid].dict() for pid in pids])
        categorical_data = data[['embarked', 'sex', 'pclass', 'title', 'is_alone']]
        numerical_data = data[['age', 'fare', 'family_size']]
        if self.trained:
            categorical_data = self.oneHotEncoder.transform(categorical_data)
            numerical_data = self.robustScaler.transform(self.knnImputer.transform(numerical_data))
        else:
            categorical_data = self.oneHotEncoder.fit_transform(categorical_data)
            numerical_data = self.robustScaler.fit_transform(self.knnImputer.fit_transform(numerical_data))
        return np.hstack((categorical_data, numerical_data))

    def train(self, passengers, pids):
        passengers_map = {p.pid: p for p in passengers}
        targets = [passengers_map[pid].is_survived for pid in pids]
        inputs = self.process_inputs(passengers, pids)     
        self.predictor.fit(inputs, targets)
        self.trained = True

    def estimate(self, passengers, pids):
        inputs = self.process_inputs(passengers, pids)     
        return self.predictor.predict(inputs)


class TitanicModelCreator:

    def __init__(self, loader):
        self.loader = loader
        np.random.seed(42)

    def get_train_pids(self, passengers):
        pids = [passenger.pid for passenger in passengers]
        targets = [passenger.is_survived for passenger in passengers]
        train_pids, test_pids = train_test_split(pids, stratify=targets, test_size=0.2)
        return train_pids, test_pids

    def run(self):
        passengers = self.loader.get_passengers()

        train_pids, test_pids = self.get_train_pids(passengers)

        passengers_map = {p.pid: p for p in passengers}
        y_train = [passengers_map[pid].is_survived for pid in train_pids]
        y_test = [passengers_map[pid].is_survived for pid in test_pids]

        # --- TRAINING --- 
        model = TitanicModel()
        model.train(passengers, train_pids)
        y_train_estimation = model.estimate(passengers, train_pids)
        cm_train = confusion_matrix(y_train, y_train_estimation)

        # --- TESTING ---
        y_test_estimation = model.estimate(passengers, test_pids)
        cm_test = confusion_matrix(y_test, y_test_estimation)

        print('cm_train', cm_train)
        print('cm_test', cm_test)

        do_test('../data/cm_test.pkl', cm_test)
        do_test('../data/cm_train.pkl', cm_train)
        X_train_processed = model.process_inputs(passengers, train_pids)
        do_test('../data/X_train_processed.pkl', X_train_processed)
        X_test_processed = model.process_inputs(passengers, test_pids)
        do_test('../data/X_test_processed.pkl', X_test_processed)

        X_train = pd.DataFrame([passengers_map[pid].dict() for pid in train_pids])
        do_pandas_test('../data/X_train.pkl', X_train)
        do_pandas_test('../data/df_no_tickets.pkl', pd.DataFrame([v.dict() for v in passengers]))
        

def main(param: str='pass'):
    titanicModelCreator = TitanicModelCreator(
        loader=PassengerLoader(
            loader=SqlLoader(
                connectionString='sqlite:///../data/titanic.db'
            ),
            rare_titles=RARE_TITLES
        )
    )
    titanicModelCreator.run()


def test_main(param: str='pass'):
    titanicModelCreator = TitanicModelCreator(
        loader=PassengerLoader(
            loader=TestLoader(
                passengers_filename='../data/passengers.pkl',
                realLoader=SqlLoader(
                    connectionString='sqlite:///../data/titanic.db'
                )
            ),
            rare_titles=RARE_TITLES
        )
    )
    titanicModelCreator.run()


if __name__ == "__main__":
    typer.run(test_main)

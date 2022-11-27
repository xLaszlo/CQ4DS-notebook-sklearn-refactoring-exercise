import numpy as np
import pandas as pd
import typer
from sklearn.datasets import fetch_openml
from sqlalchemy import create_engine


def main():
    print('loading data')
    df, targets = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    targets = pd.DataFrame(np.array([int(v) for v in targets]), columns=['is_survived'])

    print('creating db')
    engine = create_engine('sqlite:///titanic.db', echo=True)
    sqlite_connection = engine.connect()

    print('saving passengers')
    df.to_sql('tbl_passengers', sqlite_connection, index_label='pid')
    print('saving targets')
    targets.to_sql('tbl_targets', sqlite_connection, index_label='pid')

    print('closing db')
    sqlite_connection.close()

    print('done')


if __name__ == "__main__":
    typer.run(main)

"""
This is a script that helps run our titanic model fitting DAG.

It does two things:

1. Creates a DAG to fit a new model and set of encoders.
2. Creates a DAG to reuse the model and set of encoders from step (1).

"""
import pickle
from hamilton import driver, base
import data_loader
import feature_transforms
import model_pipeline

# -- create model
adapter = base.SimplePythonGraphAdapter(base.DictResult())
titanic_dag = driver.Driver(
    {
        "loader": "openml",
        "random_state": 5,
        "test_size": 0.2,
        "model_to_use": "create_new",
    },
    data_loader,
    feature_transforms,
    model_pipeline,
    adapter=adapter,
)
final_vars = [
    "cm_test",
    "cm_train",
    "fit_model",
    "fit_scaler",
    "fit_knn_imputer",
    "fit_categorical_encoder",
    "train_set",
    "test_set",
]
titanic_dag.visualize_execution(final_vars, "data/titanic_model_dag", {'format': 'png'})
results = titanic_dag.execute(final_vars)
print(results)


# -- save things we want to reuse again
def serialize_object(object: object, path: str):
    with open(path, "wb") as f:
        pickle.dump(object, f)


scaler_path = "data/scaler.pkl"
serialize_object(results["fit_scaler"], scaler_path)
knn_imputer_path = "data/knn_imputer.pkl"
serialize_object(results["fit_knn_imputer"], knn_imputer_path)
categorical_encoder_path = "data/categorical_encoder.pkl"
serialize_object(results["fit_categorical_encoder"], categorical_encoder_path)
model_path = "data/model.pkl"
serialize_object(results["fit_model"], model_path)
model_result_and_data_sets = {
    "cm_train": results["cm_train"],
    "cm_test": results["cm_test"],
    "train_passengers": results["train_set"],
    "test_passengers": results["test_set"],
}
results_path = "data/model_result_and_data_sets.pkl"
serialize_object(model_result_and_data_sets, results_path)

# -- Say now we want to reuse the old model -- this is one way we could do it.
adapter = base.SimplePythonGraphAdapter(base.DictResult())
titanic_dag_loading_previous_model = driver.Driver(
    {
        "loader": "openml",
        "random_state": 5,
        "test_size": 0.2,
        "model_to_use": "use_existing",
        "model_path": model_path,
        "scaler_path": scaler_path,
        "knn_imputer_path": knn_imputer_path,
        "categorical_encoder_path": categorical_encoder_path,
    },
    data_loader,
    feature_transforms,
    model_pipeline,
    adapter=adapter,
)
results2 = titanic_dag_loading_previous_model.execute(final_vars)
titanic_dag_loading_previous_model.visualize_execution(
    final_vars, "data/titanic_model_dag_with_reuse", {'format': 'png'}
)
print(results2)

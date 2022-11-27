### Step 20: Save model and move tests to custom model savers

- Create `ModelSaver` that has a `save_model` interface that accepts a model and a result object
- Pickle the model and the result to a file
- Create `TestModelSaver` that has the same interface
- Move the testing code to the `save_model` functon
- Add `model_saver` property to `TitanicModelCreator` and call it after the evaluation code
- Add an instance of `ModelSaver` and `TestModelSaver` respectively in `main` and `test_main` to the construction of `TitanicModelCreator`

Currently `TitanicModelCreator` contains its own testing, while this is intended to run in production. It also have no way to save the model. We will introduce the concept of `ModelSaver` here, anything that need to be preserved after the model training need to be passed to this class.

We will also move testing into a specific `TestModelSaver` that will instead of saving the model, will run the tests that were otherwise be in `run()`. This way the same code can run in production and in testing without change.

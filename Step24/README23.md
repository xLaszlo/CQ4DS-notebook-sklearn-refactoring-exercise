### Step 23: Enable training of different models

- Add `model` property to `TitanicModelCreator` and use in `run()` that instead of the local `TitanicModel` instance.
- Add `TitanicModel` instantiation to the creation of `TitanicModelCreator` in both `main` and `test_main`
- Expose parts of `TitanicModel` (predictor, processing parameter)

At this point the refactoring is pretty much finished. This last step enables the creation of different models. Delete test files to recreate ground truth for tests and to enable future changes and refactoring.

Next steps:

- Use different data: 
    - Update `SqlLoader` to retrieve different data
    - Update `Passenger` class to contain this new data
    - Update `PassengerLoader` class to process this new data into the classes
    - Update `process_inputs` to create features out of this new data
- Use different features
    - Update `process_inputs` in `TitanicModel`, expose parameters as needed
- Use different model:
    - Use different `predictor` in `TitanicModel`

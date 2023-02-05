### Step 21: Enable training of different models

- Add `model` property to `TitanicModelCreator` and use in `run()` that instead of the local `TitanicModel` instance.
- Add `TitanicModel` instantiation to the creation of `TitanicModelCreator` in both `main` and `test_main`
- Expose parts of `TitanicModel` (predictor, processing parameter)

At this point the refactoring is pretty much finished. This last step enables the creation of different models. Use existing implementations as templates to create new shell scripts, main functions (contexts) for each experiment that uses new Loaders to create new datasets. Write different test context to make sure the changes you do are as intended.As more experiments emerge, you will see patterns and opportunities to extract common behaviour from similar implementation while still maintaining validity through thes tests. This allows restructuring your code on the fly and find out what is the most convenient architecture for your system. Most problems in these systems are unforeseeable, there is no possibility to figure out the best structure before you start implementation. This require a workflow that enables radical changes even at later stages of the project. Clean Architecture, end-to-end testing and maintaining code quality provides exactly this feature at very low effort.

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

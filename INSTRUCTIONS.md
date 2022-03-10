### Step 01: Project setup

- Write script to create virtual environment
- Write the first `requirements.txt`
### Step 02: Code setup

- Write python script stub with `typer`
- Write shell script to execute the python script
### Step 03: Move code out of the notebook

- Copy-paste everything into the `run()` function
### Step 04: Move over the tests

- Copy paste tests and testing code into the `run()` function
### Step 05: Decouple from the database

- Write `SQLLoader` class
- Move database related code into it
- Replace database calls with interface calls in `run()`

This is a typical example of the Adapter Pattern. Instead of directly calling the DB, we access it through an intermediary preparing for establishing "Loose Coupling" and "Dependency Inversion".
### Step 06: Decouple from the database

- Create loader propery and argument in `TitanicModelCreator.__init__()`
- Remove the database loader instantiation from the `run()` function
- Update `TitanicModelCreator` construction to create the loader there

This will enable for the `TitanicModelCreator` to load data from any source for example files. Preparing to build a test context for rapid iteration.
### Step 07: Write testing dataloader

- Write a class that loads the required data from files
- Same interface as `SqlLoader`
- Add a "real" loader to it as a property

This will allow the test context to work without DB connection and still have the DB as a fallback when you run it for the first time. For `TitanicModelCreator` the two loaders are indistinguishable as they have the same interface.
### Step 08: Write the test context

- Create `test_main` function
- Make sure `typer` calls that in `typer.run()`
- Copy the code from `main` to `test_main`
- Replace `SqlLoader` in it with `TestLoader`

From now on this is the only code that is tested. The costly connection to the DB is replaced with a file load. Also if it is still not fast enough, additional parameter can reduce the amount of data in the test to make the process faster.
### Step 09: Merge passenger data with targets

- Remove the `get_targets()` interface
- Replace the query in `SqlLoader`
- Remove any code related to `targets`

This is a step to prepare to build the "domain data model". The Titanic model is about survival of her passengers. For the code to align this domain the concept of "passengers" need to be introduced (as a class/object). A passenger either survived or not, it's an attribute of the passenger and it need to be implemented like that.
### Step 10: Create Passenger class

- Import `BaseModel` from `pydantic`
- Create the class by inspecting:
  - The `dtype` of columns used in `df`
  - The actual values in `df`
  - The names of the columns that are used later in the code

There is really no shortcut here. In a "real" project defining this class would be the first step, but in legacy you need to deal with it later. The benefit of domain data objects is that any time you use them you can assume they fulfill a set of assumptions. These can be made explicit with `pydantic's` validators. One goal of the refactoring is to make sure that most interaction between classes happen with domain data objects. This simplifies structuring the project, any future data related change has a well defined place to happen.
### Step 11: Create domain data object based data loader

- Create `PassengerLoader` class that takes a "real"/"old" loader
- In its `get_passengers` function, load the data from the loader and create the `Passenger` objects
- Copy the data transformations from `TitanicModelCreator.run()`

Take a look at how the `rare_titles` variable is used in `run()`. After scanning the entire dataset for titles, the ones that appear less than 10 times are selected. This can be done only if you have access to the entire database and this list needs to be maintained. This can cause problems in a real setting when the above operation is too difficult to do. For example if you have millions of items or a constant stream. This kind of dependencies are common in legacy code and one of the goals of refactoring is to identify these and make explicit. Here we will use a constant but in a productionised environment this might need a whole separate service.
### Step 12: Remove any data that is not explicitly needed

- Update the query in `SqlLoader` to only retrieve the columns that will be used for the model's input

Simplifying down to the minimum is a goal of refactoring. Anything that is not explicitly needed should be removed. If the requirements change they can be added back again. For example the `ticket` column is in `df` but it is never used again in the program. Remove it.
### Step 13: Use Passenger objects in the program

- Add `PassengerLoader` to `main` and `test_main`
- Add the `RARE_TITLES` constant
- Convert the classes back into the `df` dataframe with `passenger.dict()`

It is very important to do refactoring incrementally. Any change should be small enough that if the tests fail the source can be found quickly. So for now we stop at using the new loader but do not change anything else.
### Step 14: Separate training and evaluation functions

- Move all code related to evaluation (variables that has `_test_` in their name) into one group

After creating the model first it is trained, then it is evaluated on the training data, then it is evaluated on the testing data. These should be separated from each other into their own logical place. This will prepare to move them into an actually separated place.
### Step 15: Create `TitanicModel` class

- Create a class that has all the `sklearn` components as member variables
- Instantiate these before the "Training" block
- Use these instead of the local ones

The goal of the whole program is to create a model, despite this until now there was no single object describing this model. The next steps is to establish the concept of this model and what kind of services it is providing for `TitanicModelCreator`.
### Step 16: Passenger class based training and evaluation sets

- Create a function in `TitanicModelCreator` that splits the `pids` stratified by the "targets" (namely if the passenger survived or not)
- Refactor `X_train/X_test` to be created from the list of passengers and the split `pid` lists

This is going to be a real mess. High level of entanglement between the functions of the model (train/evaluate) and splitting the data makes refactoring a challenge. The best is to do incremental stepsand sometimes make temporary changes that on their own do not make sense. These will enable to get to a cleaner state and remove them afterwards.

A simple list comprehension can not be used to recreate `X_train/X_test` because `train_test_split` shuffles the rows and the tests would fail. The separate `pid` lists are in the correct order so we create: `passengers_map = {p.pid: p for p in passengers}` and use it as a lookup table to recreate the dataframes. We will remove these at a later step.
### Step 17: Create input processing for `TitanicModel`

- Move code in `run()` from between instantiating `TitanicModel` and training (`model.predictor.fit`) to the `process_inputs` function of `TitanicModel`.
- Introduce `self.trained` boolean
- Based on `self.trained` either call the `transform` or `fit_transform` of the `sklearn` input processor functions
- For now pass all the `passengers` and the relevant `pids` to the function and do the `passengers_map` to recreate the dataframe internally.

All the input transformation code happen twice. Once for training data once for evaluation data. While transforming the data is a responsibility of the model. This is a codesmell called "feature envy". `TitanicModelCreator` envies the functionality from `TitanicModel`. There will be several steps to resolve this. The resulting code will create a self contained model that can be shipped independetly from its creator.

The messy bit with the `passengers_map` is an intermediate step and will be resolved later.
### Step 18: Move training into `TitanicModel`

- Use the same interface as `process_inputs` with `train()`
- Process the data with `process_inputs` (just pass through the arguments)
- Recreate the required targets with the mapping
- Train the model and set the `trained` boolean to `True`

This is a straightforward step, but it still requires the ugly `passengers_map` thing in `train` as well. We will deal with it separately.
### Step 19: Move prediction to `TitanicModel`

- Create the `estimate` function
- Call `proccess_inputs` and `predictor.predict` in it
- Remove all evaluation input processing code
- Call `estimate` from `run`

Because there was no separation of concerns the input processing code was duplicated and now that we moved it to its own location it can be removed.

`X_train_processed` and `X_test_processed` do not exist any more so to pass the tests they need to be recreated before the test is called. This is usually a sign that some other variable need to be tested. Currently the test tests implementation instead of behaviour. From the outside the only thing that matters if the model is creating the same output. The exact internal processed input that is passed to the predictor in the model doesn't.
### Step 20: Refactor model interface

- Remove `pids` from all the interfaces
- Refactor `run` to pass in only the relevant set of passengers at each point

After finishing the model, it is time to clean the interface and remove those `passengers_maps`. This will be done by moving the filtering of data from the model functions to the caller. We will remove it from there as well in the next step.
### Step 21: Introduce `train_passengers/test_passengers`

- Declare `train_passengers/test_passengers` through the `passengers_map`
- Use the above on all places instead of comprehensions

This is another cleanup step. The concept of "training" and "evaluation" passenger datasets are introduced. This might have been an easier journey if we implement `get_train_passengers` instead of `get_train_pids` but in the course of refactoring this can happen. Currently we have the choice to do this because all dataset related functions are close to each other.
### Step 22: Save model and move tests to custom model savers

- Create `ModelSaver` that has a `save_model` interface that accepts a model and a result object
- Pickle the model and the result to a file
- Create `TestModelSaver` that has the same interface
- Move the testing code to the `save_model` functon
- Add `model_saver` property to `TitanicModelCreator` and call it after the evaluation code
- Add an instance of `ModelSaver` and `TestModelSaver` respectively in `main` and `test_main` to the construction of `TitanicModelCreator`

Currently `TitanicModelCreator` contains its own testing, while this is intended to run in production. It also have no way to save the model. We will introduce the concept of `ModelSaver` here, anything that need to be preserved after the model training need to be passed to this class. 

We will also move testing into a specific `TestModelSaver` that will instead of saving the model, will run the tests that were otherwise be in `run()`. This way the same code can run in production and in testing without change.
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

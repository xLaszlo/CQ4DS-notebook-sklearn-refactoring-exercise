### Step 01: Project setup

- Write script to create virtual environment
- Write the first `requirements.txt`

You can select a different `setuptools` version or pin the package versions.
### Step 02: Code setup

- Write python script stub with `typer`
- Write shell script to execute the python script

`Typer` is an amazing tool that turns any python script into shell scripts. Here we use it for future-proofing because at the moment there are no CLI arguments.

The program will be defined in a class that is instantiated by the `main()` function and call its main `run()` entry point. The `main()` function will be called by `typer` to pass any CLI parameters. This setup will allow us to create a "plugin" architecture and construct different behaviour (e.g.: normal, test, production) in different main functions. This is a form of "Clean Architecture" where the code (the class) is independent of the infrastructure that calls it (`main()`) more on this: [Clean Architecture: How to structure your ML projects to reduce technical debt (PyData London 2022)](https://laszlo.substack.com/p/slides-for-my-talk-at-pydata-london).
### Step 03: Move code out of the notebook

- Copy-paste everything into the `run()` function

First step is to get started. There will be plenty of steps to structure the code better.
### Step 04: Move over the tests

- Copy paste tests and testing code from the notebook in `Step00` into the `run()` function.

This will implement very simple end-to-end testing which is less effort than unit testing given that the code is not really in a testable state. It caches the value of some variables and the next time you run the code it will compare it to this cache. If they match you didn't change the behaviour of the code with the last change. If your intentions was indeed to change the behaviour, verify from the output of the `AssertionError` that the changes are working as intended. If they are, delete the chaches and rerun the code to generate new reference values. The tests should be such that if they fail they produce meaningful differences. So instead of aggregate statistics (like an F1 score) test the datasets itself. That way even small changes won't go undetected. Once the code is refactored you can write different type of tests but that's a different story.
### Step 05: Decouple from the database

- Write `SQLLoader` class
- Move database related code into it
- Replace database calls with interface calls in `run()`

This is a typical example of the Adapter Pattern. Instead of directly calling the DB, we access it through an intermediary preparing for establishing "Loose Coupling" and "Dependency Inversion". In Clean Architecture the main code (the `run()` function) shouldn't know where the data is coming from, just what the data is. This will bring flexibility because this adapter can be replaced with another one that has the same interface but gets the data from a file. After that you can run your main code without a database which makes it more testable. More on this: [You only need 2 Design Patterns to improve the quality of your code in a data science project](https://laszlo.substack.com/p/you-only-need-2-design-patterns-to).
### Step 06: Decouple from the database

- Create loader propery and argument in `TitanicModelCreator.__init__()`
- Remove the database loader instantiation from the `run()` function
- Update `TitanicModelCreator` construction to create the loader there

This will enable for the `TitanicModelCreator` to load data from any source for example files. Preparing to build a test context for rapid iteration. After you created the adapter class, this will do the decoupling. This is an example of "Dependency Injection", when a property of your main code is not written into the main body of the code but instead "plugged in" at constrcution time. The benefit of Dependency Injection is that you can change the behaviour of your code without rewriting it by purely changing its construction. As the saying goes: "Complex behaviour is constructed not written." Dependency Injection Principle is the `D` in the famed `SOLID` principles, and arguably the most important.
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

From now on this is the only code that is tested. The costly connection to the DB is replaced with a file load. Also if it is still not fast enough, additional parameter can reduce the amount of data in the test to make the process faster. [How can a Data Scientist refactor Jupyter notebooks towards production-quality code?](https://laszlo.substack.com/p/how-can-a-data-scientist-refactor) [I appreciate this might be terse. Comment, open an issue, vote on it if you would like to have a detailed discussion on this - Laszlo]

This is the essence of the importance of Clean Architecture and code reuse. Every code will be used in two different context: test and "production" by injecting different dependencies. Because the same code runs in both places there is no time spent on translating from one to another. The test setup should reflect production context as close as possible so when a test fail or pass you can think that the same will happen in production as well. This speed up iteration because you can freely experiment in the test context and only deploy code into "production" when you are convinced it is doing what you think it should do. But it is the same code, so deployment is effortless.
### Step 09: Merge passenger data with targets

- Remove the `get_targets()` interface
- Replace the query in `SqlLoader`
- Remove any code related to `targets`

This is a step to prepare to build the "domain data model". The Titanic model is about survival of her passengers. For the code to align this domain the concept of "passengers" need to be introduced (as a class/object). A passenger either survived or not, it's an attribute of the passenger and it need to be implemented like that.

This is a critical part of the code quality journey and building better systems. Once you introduce these concepts your code will depend directly on the business problem you are solving not the various representations the data is stored (pandas, numpy, csv, etc). I wrote about this many times on my blog:

- [3 Ways Domain Data Models help Data Science Projects](https://laszlo.substack.com/p/3-ways-domain-data-models-help-data)
- [Clean Architecture: How to structure your ML projects to reduce technical debt](https://laszlo.substack.com/p/slides-for-my-talk-at-pydata-london)
- [How did I change my mind about dataclasses in ML projects?](https://laszlo.substack.com/p/how-did-i-change-my-mind-about-dataclasses)
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

`PassengerLoader` implements the Factory Design Pattern. Factories are classes that create other classes, they are a type of adapter that hides away where the data is coming from and how is it stored and return only abstract domain relevant classes that you can use downstream. Factories are one of two (later increased to three) fundamentally relevant Design Patterns for Data Science workflows:

- [You only need 2 Design Patterns to improve the quality of your code in a data science project](https://laszlo.substack.com/p/you-only-need-2-design-patterns-to)
- [Clean Architecture: How to structure your ML projects to reduce technical debt](https://laszlo.substack.com/p/slides-for-my-talk-at-pydata-london)
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

`X_train_processed` and `X_test_processed` do not exist any more so to pass the tests they need to be recreated. This is a good point to think about why this is necessary and find a different way to test behaviour. To keep the project short we set aside this but this would be a good place to introduce more tests. 
### Step 20: Save model and move tests to custom model savers

- Create `ModelSaver` that has a `save_model` interface that accepts a model and a result object
- Pickle the model and the result to a file
- Create `TestModelSaver` that has the same interface
- Move the testing code to the `save_model` functon
- Add `model_saver` property to `TitanicModelCreator` and call it after the evaluation code
- Add an instance of `ModelSaver` and `TestModelSaver` respectively in `main` and `test_main` to the construction of `TitanicModelCreator`

Currently `TitanicModelCreator` contains its own testing, while this is intended to run in production. It also have no way to save the model. We will introduce the concept of `ModelSaver` here, anything that need to be preserved after the model training need to be passed to this class.

We will also move testing into a specific `TestModelSaver` that will instead of saving the model, will run the tests that were otherwise be in `run()`. This way the same code can run in production and in testing without change.
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

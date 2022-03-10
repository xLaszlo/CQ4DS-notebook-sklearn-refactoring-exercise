## Step 16: Passenger class based training and evaluation sets

- Create a function in `TitanicModelCreator` that splits the `pids` stratified by the "targets" (namely if the passenger survived or not)
- Refactor `X_train/X_test` to be created from the list of passengers and the split `pid` lists

This is going to be a real mess. High level of entanglement between the functions of the model (train/evaluate) and splitting the data makes refactoring a challenge. The best is to do incremental stepsand sometimes make temporary changes that on their own do not make sense. These will enable to get to a cleaner state and remove them afterwards. 

A simple list comprehension can not be used to recreate `X_train/X_test` because `train_test_split` shuffles the rows and the tests would fail. The separate `pid` lists are in the correct order so we create: `passengers_map = {p.pid: p for p in passengers}` and use it as a lookup table to recreate the dataframes. We will remove these at a later step.

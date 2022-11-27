### Step 16: Passenger class based training and evaluation sets

- Create a function in `TitanicModelCreator` that splits the `passengers` stratified by the "targets" (namely if the passenger survived or not)
- Refactor `X_train/X_test` to be created from these lists of passengers

Because `train_test_split` works on lists, we extract the pids and the targets from the classes and create the two sets from a mapping from pids to passenger classes.

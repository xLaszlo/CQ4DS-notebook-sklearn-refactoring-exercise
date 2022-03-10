### Step 11: Create domain data object based data loader

- Create `PassengerLoader` class that takes a "real"/"old" loader
- In its `get_passengers` function, load the data from the loader and create the `Passenger` objects
- Copy the data transformations from `TitanicModelCreator.run()`

Take a look at how the `rare_titles` variable is used in `run()`. After scanning the entire dataset for titles, the ones that appear less than 10 times are selected. This can be done only if you have access to the entire database and this list needs to be maintained. This can cause problems in a real setting when the above operation is too difficult to do. For example if you have millions of items or a constant stream. This kind of dependencies are common in legacy code and one of the goals of refactoring is to identify these and make explicit. Here we will use a constant but in a productionised environment this might need a whole separate service.

### Step 11: Create domain data object based data loader

- Create `PassengerLoader` class that takes a "real"/"old" loader
- In its `get_passengers` function, load the data from the loader and create the `Passenger` objects
- Copy the data transformations from `TitanicModelCreator.run()`

Take a look at how the `rare_titles` variable is used in `run()`. After scanning the entire dataset for titles, the ones that appear less than 10 times are selected. This can be done only if you have access to the entire database and this list needs to be maintained. This can cause problems in a real setting when the above operation is too difficult to do. For example if you have millions of items or a constant stream. This kind of dependencies are common in legacy code and one of the goals of refactoring is to identify these and make explicit. Here we will use a constant but in a productionised environment this might need a whole separate service.

`PassengerLoader` implements the Factory Design Pattern. Factories are classes that create other classes, they are a type of adapter that hides away where the data is coming from and how is it stored and return only abstract domain relevant classes that you can use downstream. Factories are one of two (later increased to three) fundamentally relevant Design Patterns for Data Science workflows:

- [You only need 2 Design Patterns to improve the quality of your code in a data science project](https://laszlo.substack.com/p/you-only-need-2-design-patterns-to)
- [Clean Architecture: How to structure your ML projects to reduce technical debt](https://laszlo.substack.com/p/slides-for-my-talk-at-pydata-london)

## Step 06: Decouple from the database

- Create loader propery and argument in `TitanicModelCreator.__init__()`
- Remove the database loader instantiation from the `run()` function
- Update `TitanicModelCreator` construction to create the loader there

This will enable for the `TitanicModelCreator` to load data from any source for example files. Preparing to build a test context for rapid iteration.

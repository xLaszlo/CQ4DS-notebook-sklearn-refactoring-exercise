### Step 07: Write testing dataloader

- Write a class that loads the required data from files
- Same interface as `SqlLoader`
- Add a "real" loader to it as a property

This will allow the test context to work without DB connection and still have the DB as a fallback when you run it for the first time. For `TitanicModelCreator` the two loaders are indistinguishable as they have the same interface.

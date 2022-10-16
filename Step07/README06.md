### Step 06: Decouple from the database

- Create loader propery and argument in `TitanicModelCreator.__init__()`
- Remove the database loader instantiation from the `run()` function
- Update `TitanicModelCreator` construction to create the loader there

This will enable for the `TitanicModelCreator` to load data from any source for example files. Preparing to build a test context for rapid iteration. After you created the adapter class, this will do the decoupling. This is an example of "Dependency Injection", when a property of your main code is not written into the main body of the code but instead "plugged in" at constrcution time. The benefit of Dependency Injection is that you can change the behaviour of your code without rewriting it by purely changing its construction. As the saying goes: "Complex behaviour is constructed not written." Dependency Injection Principle is the `D` in the famed `SOLID` principles, and arguably the most important.

### Step 05: Decouple from the database

- Write `SQLLoader` class
- Move database related code into it
- Replace database calls with interface calls in `run()`

This is a typical example of the Adapter Pattern. Instead of directly calling the DB, we access it through an intermediary preparing for establishing "Loose Coupling" and "Dependency Inversion". In Clean Architecture the main code (the `run()` function) shouldn't know where the data is coming from, just what the data is. This will bring flexibility because this adapter can be replaced with another one that has the same interface but gets the data from a file. After that you can run your main code without a database which makes it more testable. More on this: [You only need 2 Design Patterns to improve the quality of your code in a data science project](https://laszlo.substack.com/p/you-only-need-2-design-patterns-to).

### Step 05: Decouple from the database

- Write `SQLLoader` class
- Move database related code into it
- Replace database calls with interface calls in `run()`

This is a typical example of the Adapter Pattern. Instead of directly calling the DB, we access it through an intermediary preparing for establishing "Loose Coupling" and "Dependency Inversion".

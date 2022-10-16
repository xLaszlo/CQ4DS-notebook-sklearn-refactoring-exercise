### Step 13: Use Passenger objects in the program

- Add `PassengerLoader` to `main` and `test_main`
- Add the `RARE_TITLES` constant
- Convert the classes back into the `df` dataframe with `passenger.dict()`

It is very important to do refactoring incrementally. Any change should be small enough that if the tests fail the source can be found quickly. So for now we stop at using the new loader but do not change anything else.

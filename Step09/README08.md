
## Step 08: Write the test context

- Create `test_main` function 
- Make sure `typer` calls that in `typer.run()`
- Copy the code from `main` to `test_main`
- Replace `SqlLoader` in it with `TestLoader`

From now on this is the only code that is tested. The costly connection to the DB is replaced with a file load. Also if it is still not fast enough, additional parameter can reduce the amount of data in the test to make the process faster.

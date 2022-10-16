### Step 08: Write the test context

- Create `test_main` function
- Make sure `typer` calls that in `typer.run()`
- Copy the code from `main` to `test_main`
- Replace `SqlLoader` in it with `TestLoader`

From now on this is the only code that is tested. The costly connection to the DB is replaced with a file load. Also if it is still not fast enough, additional parameter can reduce the amount of data in the test to make the process faster. [How can a Data Scientist refactor Jupyter notebooks towards production-quality code?](https://laszlo.substack.com/p/how-can-a-data-scientist-refactor) [I appreciate this might be terse. Comment, open an issue, vote on it if you would like to have a detailed discussion on this - Laszlo]

This is the essence of the importance of Clean Architecture and code reuse. Every code will be used in two different context: test and "production" by injecting different dependencies. Because the same code runs in both places there is no time spent on translating from one to another. The test setup should reflect production context as close as possible so when a test fail or pass you can think that the same will happen in production as well. This speed up iteration because you can freely experiment in the test context and only deploy code into "production" when you are convinced it is doing what you think it should do. But it is the same code, so deployment is effortless.

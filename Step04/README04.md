### Step 04: Move over the tests

- Copy paste tests and testing code from the notebook in `Step00` into the `run()` function.

This will implement very simple end-to-end testing which is less effort than unit testing given that the code is not really in a testable state. It caches the value of some variables and the next time you run the code it will compare it to this cache. If they match you didn't change the behaviour of the code with the last change. If your intentions was indeed to change the behaviour, verify from the output of the `AssertionError` that the changes are working as intended. If they are, delete the chaches and rerun the code to generate new reference values. The tests should be such that if they fail they produce meaningful differences. So instead of aggregate statistics (like an F1 score) test the datasets itself. That way even small changes won't go undetected. Once the code is refactored you can write different type of tests but that's a different story.

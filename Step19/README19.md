### Step 19: Move prediction to `TitanicModel`

- Create the `estimate` function
- Call `proccess_inputs` and `predictor.predict` in it
- Remove all evaluation input processing code
- Call `estimate` from `run`

Because there was no separation of concerns the input processing code was duplicated and now that we moved it to its own location it can be removed.

`X_train_processed` and `X_test_processed` do not exist any more so to pass the tests they need to be recreated. This is a good point to think about why this is necessary and find a different way to test behaviour. To keep the project short we set aside this but this would be a good place to introduce more tests.

### Step 19: Move prediction to `TitanicModel`

- Create the `estimate` function
- Call `proccess_inputs` and `predictor.predict` in it
- Remove all evaluation input processing code
- Call `estimate` from `run`

Because there was no separation of concerns the input processing code was duplicated and now that we moved it to its own location it can be removed.

`X_train_processed` and `X_test_processed` do not exist any more so to pass the tests they need to be recreated before the test is called. This is usually a sign that some other variable need to be tested. Currently the test tests implementation instead of behaviour. From the outside the only thing that matters if the model is creating the same output. The exact internal processed input that is passed to the predictor in the model doesn't.

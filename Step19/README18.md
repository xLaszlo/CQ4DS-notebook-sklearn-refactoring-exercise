
## Step 18: Move training into `TitanicModel`

- Use the same interface as `process_inputs` with `train()`
- Process the data with `process_inputs` (just pass through the arguments)
- Recreate the required targets with the mapping
- Train the model and set the `trained` boolean to `True`

This is a straightforward step, but it still requires the ugly `passengers_map` thing in `train` as well. We will deal with it separately.

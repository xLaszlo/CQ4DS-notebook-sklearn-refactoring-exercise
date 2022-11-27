### Step 18: Move training into `TitanicModel`

- Use the same interface as `process_inputs` with `train()`
- Process the data with `process_inputs` (just pass through the arguments)
- Recreate the required targets with the mapping
- Train the model and set the `trained` boolean to `True`

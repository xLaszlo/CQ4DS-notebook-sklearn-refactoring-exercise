### Step 17: Create input processing for `TitanicModel`

- Move code in `run()` from between instantiating `TitanicModel` and training (`model.predictor.fit`) to the `process_inputs` function of `TitanicModel`.
- Introduce `self.trained` boolean
- Based on `self.trained` either call the `transform` or `fit_transform` of the `sklearn` input processor functions

All the input transformation code happen twice. Once for training data once for evaluation data. While transforming the data is a responsibility of the model. This is a codesmell called "feature envy". `TitanicModelCreator` envies the functionality from `TitanicModel`. There will be several steps to resolve this. The resulting code will create a self contained model that can be shipped independetly from its creator.


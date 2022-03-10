
## Step 21: Introduct `train_passengers/test_passengers`

- Declare `train_passengers/test_passengers` through the `passengers_map`
- Use the above on all places instead of comprehensions

This is another cleanup step. The concept of "training" and "evaluation" passenger datasets are introduced. This might have been an easier journey if we implement `get_train_passengers` instead of `get_train_pids` but in the course of refactoring this can happen. Currently we have the choice to do this because all dataset related functions are close to each other.

## Step 12: Remove any data that is not explicitly needed

- Update the query in `SqlLoader` to only retrieve the columns that will be used for the model's input

Simplifying down to the minimum is a goal of refactoring. Anything that is not explicitly needed should be removed. If the requirements change they can be added back again. For example the `ticket` column is in `df` but it is never used again in the program. Remove it.


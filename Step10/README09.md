### Step 09: Merge passenger data with targets

- Remove the `get_targets()` interface
- Replace the query in `SqlLoader`
- Remove any code related to `targets`

This is a step to prepare to build the "domain data model". The Titanic model is about survival of her passengers. For the code to align this domain the concept of "passengers" need to be introduced (as a class/object). A passenger either survived or not, it's an attribute of the passenger and it need to be implemented like that.

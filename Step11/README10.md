## Step 10: Create Passenger class

- Import `BaseModel` from `pydantic`
- Create the class by inspecting:
    - The `dtype` of columns used in `df`
    - The actual values in `df`
    - The names of the columns that are used later in the code

There is really no shortcut here. In a "real" project defining this class would be the first step, but in legacy you need to deal with it later. The benefit of domain data objects is that any time you use them you can assume they fulfill a set of assumptions. These can be made explicit with `pydantic's` validators. One goal of the refactoring is to make sure that most interaction between classes happen with domain data objects. This simplifies structuring the project, any future data related change has a well defined place to happen.




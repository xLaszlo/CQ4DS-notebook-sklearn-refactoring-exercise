### Step 20: Refactor model interface

- Remove `pids` from all the interfaces
- Refactor `run` to pass in only the relevant set of passengers at each point

After finishing the model, it is time to clean the interface and remove those `passengers_maps`. This will be done by moving the filtering of data from the model functions to the caller. We will remove it from there as well in the next step.

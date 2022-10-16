### Step 09: Merge passenger data with targets

- Remove the `get_targets()` interface
- Replace the query in `SqlLoader`
- Remove any code related to `targets`

This is a step to prepare to build the "domain data model". The Titanic model is about survival of her passengers. For the code to align this domain the concept of "passengers" need to be introduced (as a class/object). A passenger either survived or not, it's an attribute of the passenger and it need to be implemented like that.

This is a critical part of the code quality journey and building better systems. Once you introduce these concepts your code will depend directly on the business problem you are solving not the various representations the data is stored (pandas, numpy, csv, etc). I wrote about this many times on my blog:

- [3 Ways Domain Data Models help Data Science Projects](https://laszlo.substack.com/p/3-ways-domain-data-models-help-data)
- [Clean Architecture: How to structure your ML projects to reduce technical debt](https://laszlo.substack.com/p/slides-for-my-talk-at-pydata-london)
- [How did I change my mind about dataclasses in ML projects?](https://laszlo.substack.com/p/how-did-i-change-my-mind-about-dataclasses)

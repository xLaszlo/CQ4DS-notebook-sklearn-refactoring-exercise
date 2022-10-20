# CQ4DS Notebook Sklearn Refactoring Exercise

This step-by-step programme demonstrates how to refactor a Data Science project from notebooks to well-formed classes and scripts.  

### The project:

The notebook demonstrates a typical setup of a data science project:

- Connects to a database (included in the repository as an SQLite file).
- Gathers some data (the classic Titanic example).
- Does feature engineering.
- Fits a model to estimate survival (sklearn's LogisticRegression).
- Evaluates the model.

### Context, vision, 

I wrote a detailed post on the concepts, strategy and big picture thinking. I recommend reading it parallel with the instructions and the steps in the pull request while you are doing the exercises: 

[https://laszlo.substack.com/p/refactoring-the-titanic](https://laszlo.substack.com/p/refactoring-the-titanic)

### Refactoring

The programme demonstrates how to improve code quality, increase agility and prepare for unforeseen changes in a real-world project (see `INSTRUCTIONS.md` for reference reading). You will perform the following steps:

- Create end-to-end functional testing
- Create shell scripts, command line interfaces, virtual environments
- Decouple from external sources (the Database)
- Refactor with simple Design Patterns (Adapter/Factory/Strategy)
- Improve readability
- Reduce code duplication

### Howto:

- Clone the repository.
- Create a virtual environment with `make_venv.sh`.
- Follow the instructions in `INSTRUCTIONS.md`.
- Run the tests with `titanic_model.sh`.
- Check the diffs of the pull request's steps to verify your progress.

### Community:

For more information and help, join our interactive self-help Code Quality for Data Science (CQ4DS) community on discord: [https://discord.gg/8uUZNMCad2](https://discord.gg/8uUZNMCad2).

Original project content from and inspired by: [https://jaketae.github.io/study/sklearn-pipeline/](https://jaketae.github.io/study/sklearn-pipeline/)

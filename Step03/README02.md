### Step 02: Code setup

- Write python script stub with `typer`
- Write shell script to execute the python script

`Typer` is an amazing tool that turns any python script into shell scripts. Here we use it for future-proofing because at the moment there are no CLI arguments.

The program will be defined in a class that is instantiated by the `main()` function and call its main `run()` entry point. The `main()` function will be called by `typer` to pass any CLI parameters. This setup will allow us to create a "plugin" architecture and construct different behaviour (e.g.: normal, test, production) in different main functions. This is a form of "Clean Architecture" where the code (the class) is independent of the infrastructure that calls it (`main()`) more on this: [Clean Architecture: How to structure your ML projects to reduce technical debt (PyData London 2022)](https://laszlo.substack.com/p/slides-for-my-talk-at-pydata-london).

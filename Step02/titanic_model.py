import typer


class TitanicModelCreator:
    def __init__(self):
        pass

    def run(self):
        print('Hello World!')


def main(param: str = 'pass'):
    titanic_model_creator = TitanicModelCreator()
    titanic_model_creator.run()


if __name__ == "__main__":
    typer.run(main)

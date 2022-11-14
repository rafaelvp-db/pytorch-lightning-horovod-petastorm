from carsifier.data.data import CarsDataModule


def test_data():

    dm = CarsDataModule()
    dm.setup("fit")
    assert dm.train is not None
from carsifier.data.data import CarsDataModule


def test_data():

    dm = CarsDataModule(
        data_dir = "/tmp",
        download = False
    )
    dm.setup("fit")
    assert dm.train is not None
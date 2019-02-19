from ..bases import energy2mass, energy2wavelength, energy2sigma, EnergyProperty, HasEnergy


def test_energy2mass():
    assert energy2mass(300e3) == 1.445736928082275e-30


def test_energy2wavelength():
    assert energy2wavelength(300e3) == 0.01968748889772767


def test_energy2sigma():
    assert energy2sigma(300e3) == 0.0006526161464700888


def test_energy_property():
    energy_property = EnergyProperty(energy=300e3)

    assert energy_property.wavelength == energy2wavelength(300e3)


def test_has_energy():
    class Dummy(HasEnergy):
        def __init__(self, energy=None, energy_property=None):
            HasEnergy.__init__(self, energy=energy, energy_property=energy_property)

    energy_property = EnergyProperty(energy=300e3)

    dummy = Dummy(energy_property=energy_property)

    assert dummy.energy == 300e3

    dummy = Dummy(energy=300e3)

    assert dummy.energy == 300e3

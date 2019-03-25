import numpy as np
import pytest

from .test_utils import CallCounter
from ..bases import energy2mass, energy2wavelength, energy2sigma, EnergyProperty, HasEnergy
from ..transfer import FrequencyTransfer, Aperture, TemporalEnvelope, PhaseAberration, CTF
from ..waves import WaveFactory, PlaneWaves, ProbeWaves, PrismWaves


def test_energy2mass():
    assert np.isclose(energy2mass(300e3), 1.445736928082275e-30)


def test_energy2wavelength():
    assert np.isclose(energy2wavelength(300e3), 0.01968748889772767)


def test_energy2sigma():
    assert np.isclose(energy2sigma(300e3), 0.0006526161464700888)


def test_energy_property():
    energy_property = EnergyProperty(energy=300e3)

    assert energy_property.energy == 300e3
    assert energy_property.wavelength == energy2wavelength(300e3)


@pytest.mark.parametrize('has_energy',
                         [HasEnergy,
                          FrequencyTransfer, Aperture, TemporalEnvelope, PhaseAberration, CTF,
                          WaveFactory, PlaneWaves, ProbeWaves, PrismWaves])
def test_has_energy(has_energy):
    instance = has_energy(energy=300e3)
    assert instance.energy == 300e3
    assert instance.wavelength == energy2wavelength(300e3)
    instance = has_energy()
    instance._energy = EnergyProperty(energy=200e3)
    assert instance.energy == 200e3


@pytest.mark.parametrize('tensorfactory_with_energy',
                         [FrequencyTransfer, Aperture, TemporalEnvelope, PhaseAberration, CTF,
                          WaveFactory, PlaneWaves, ProbeWaves, PrismWaves])
def test_energy_update(tensorfactory_with_energy):
    instance = tensorfactory_with_energy(energy=300e3)

    counter = CallCounter(lambda: instance.energy)
    instance._calculate_tensor = counter.func_caller
    instance.check_is_defined = lambda: None

    assert instance.up_to_date == False
    assert instance.get_tensor() == 300e3
    assert instance.up_to_date == True
    assert counter.n == 1
    instance.get_tensor()
    assert counter.n == 1

    instance.energy = 200e3
    assert instance.up_to_date == False
    assert instance.get_tensor() == 200e3
    assert instance.up_to_date == True
    assert counter.n == 2

    instance.energy = 200e3
    assert instance.up_to_date == True
    instance.get_tensor()
    assert counter.n == 2

    instance = tensorfactory_with_energy()

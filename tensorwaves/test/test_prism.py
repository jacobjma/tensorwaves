import numpy as np
from ase import Atoms

from ..detect import RingDetector
from ..waves import ProbeWaves, PrismWaves


def test_prism_probe():
    probe = ProbeWaves(extent=8, gpts=512, energy=100e3, aperture_radius=0.01)
    probe.positions = (4, 4)
    probe_image = np.abs(probe.get_tensor().numpy()) ** 2

    prism = PrismWaves(extent=8, gpts=512, energy=100e3, cutoff=0.01)
    prism_probe_image = np.abs(prism.get_tensor().get_tensor().numpy()) ** 2

    assert np.allclose(probe_image, prism_probe_image, atol=1e-5, rtol=.001)


def test_prism_probe_aberrations():
    probe = ProbeWaves(extent=8, gpts=512, energy=100e3, aperture_radius=0.01)
    probe.positions = (4, 4)
    probe.aberrations.parametrization.defocus = 100
    probe_image = np.abs(probe.get_tensor().numpy()) ** 2

    prism = PrismWaves(extent=8, gpts=512, energy=100e3, cutoff=0.01)
    S = prism.get_tensor()
    S.aberrations.parametrization.defocus = 100
    S.position = (4, 4)

    prism_probe_image = np.abs(S.get_tensor().numpy()) ** 2

    assert np.allclose(probe_image, prism_probe_image, atol=1e-5, rtol=.001)


def test_prism_multislice():
    atoms = Atoms('Au', positions=[(2, 2, 1)], cell=(4, 4, 1))

    probe = ProbeWaves(gpts=512, energy=100e3, aperture_radius=0.02)
    probe.positions = (2, 2)
    probe = probe.multislice(atoms)
    probe_image = np.abs(probe.numpy()) ** 2

    prism = PrismWaves(gpts=512, energy=100e3, cutoff=0.02)
    S = prism.multislice(atoms)
    S.position = (2, 2)
    prism_probe_image = np.abs(S.get_tensor().numpy()) ** 2

    assert np.allclose(probe_image, prism_probe_image, atol=1e-5, rtol=.001)


def test_linescan():
    atoms = Atoms('Au', positions=[(2, 2, 1)], cell=(4, 4, 1))

    probe = ProbeWaves(gpts=128, energy=200e3, aperture_radius=0.01)
    detector = RingDetector(inner=0.04, outer=.2)

    scan = probe.linescan(start=(2, 2), end=(4, 2), num_positions=10, potential=atoms, detectors=(detector,),
                          max_batch=10)

    prism = PrismWaves(gpts=128, energy=200e3, cutoff=.01, interpolation=1)
    S = prism.multislice(atoms)
    prism_scan = S.linescan(start=(2, 2), end=(4, 2), num_positions=10, detectors=(detector,))

    assert np.allclose(scan.numpy(), prism_scan.numpy())


def test_gridscan():
    atoms = Atoms('Au', positions=[(2, 2, 1)], cell=(4, 4, 1))

    probe = ProbeWaves(gpts=128, energy=200e3, aperture_radius=0.01)
    detector = RingDetector(inner=0.04, outer=.2)

    scan = probe.gridscan(start=(0, 0), end=(4, 4), num_positions=(10, 10), potential=atoms, detectors=(detector,),
                          max_batch=10)

    waves = PrismWaves(gpts=128, energy=200e3, cutoff=.01, interpolation=1)
    S = waves.multislice(atoms)

    detector = RingDetector(inner=0.04, outer=.2)

    prism_scan = S.gridscan(start=(0, 0), end=(4, 4), num_positions=(10, 10), detectors=(detector,))

    assert np.allclose(scan.numpy(), prism_scan.numpy())

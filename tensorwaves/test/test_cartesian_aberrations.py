import numpy as np
import tensorflow as tf

from tensorwaves.transfer import PhaseAberration, polar2cartesian, cartesian2polar

tf.enable_eager_execution()


def test_cartesian_aberrations():
    polar = {'C10': -0.32946933106950427,
             'C12': 0.21952131413482778,
             'phi12': 0.7959807283776827,
             'C21': 96.33956134736414,
             'phi21': -2.901658413073904,
             'C23': 18.872887355721325,
             'phi23': 3.052515052950999,
             'C30': -689.6580852868879,
             'C32': 474.95542338446086,
             'phi32': -1.3681637906849324,
             'C34': 359.19550595819817,
             'phi34': 2.301293052118716}

    cartesian = polar2cartesian(polar)

    cartesian_aberrations = PhaseAberration(extent=(10, 5), gpts=56, energy=80e3, parametrization='cartesian',
                                            **cartesian)

    polar_aberrations = PhaseAberration(extent=(10, 5), gpts=56, energy=80e3, parametrization='polar', **polar)

    assert np.all(np.abs(cartesian_aberrations.get_tensor().numpy()[0] -
                         polar_aberrations.get_tensor().numpy()[0]) < 5e-4)


def test_conversion():
    polar = {'C10': -0.32946933106950427,
             'C12': 0.21952131413482778,
             'phi12': 0.7959807283776827,
             'C21': 96.33956134736414,
             'phi21': -2.901658413073904,
             'C23': 18.872887355721325,
             'phi23': 3.052515052950999,
             'C30': -689.6580852868879,
             'C32': 474.95542338446086,
             'phi32': -1.3681637906849324,
             'C34': 359.19550595819817,
             'phi34': 2.301293052118716}

    cartesian1 = polar2cartesian(polar)
    cartesian2 = polar2cartesian(cartesian2polar(cartesian1))

    for (key1, value1), (key2, value2) in zip(cartesian1.items(), cartesian2.items()):
        assert np.isclose(value1, value2)

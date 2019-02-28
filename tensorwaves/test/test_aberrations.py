from ..transfer import PhaseAberration
from .test_utils import CallCounter
import pytest


@pytest.mark.parametrize('parametrization',
                         ['polar', 'cartesian'])
def test_notifying(parametrization):
    aberrations = PhaseAberration(parametrization=parametrization)

    counter = CallCounter(lambda: None)
    aberrations._calculate_tensor = counter.func_caller

    assert aberrations.up_to_date == False

    for symbol in aberrations.parametrization._parameters.keys():
        aberrations.get_tensor()
        assert aberrations.up_to_date == True
        setattr(aberrations.parametrization, symbol, 10)
        assert getattr(aberrations.parametrization, symbol) == 10.
        assert aberrations.up_to_date == False

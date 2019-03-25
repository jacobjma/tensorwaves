from ..bases import Observable, Observer, TensorFactory, notifying_property
import tensorflow as tf
import numpy as np


def test_tensorfactory():
    class DummyObservable(Observable):

        def __init__(self):
            Observable.__init__(self)
            self._x = 0

        x = notifying_property('_x')

    class DummyTensorFactory(TensorFactory):

        def __init__(self, save_tensor=True):
            TensorFactory.__init__(self, save_tensor=save_tensor)
            self.observable = DummyObservable()
            self.observable.register_observer(self)

        def check_is_defined(self):
            pass

        def _calculate_tensor(self):
            return self.observable.x * tf.ones((2, 2), dtype=tf.float32)

    updatable = DummyTensorFactory()

    assert updatable.up_to_date == False

    updatable.observable.x = 5
    updatable.get_tensor()

    assert updatable.up_to_date == True
    assert np.all(np.isclose(updatable._tensor, 5 * tf.ones((2, 2), dtype=tf.float32)))

    updatable.observable.x = 5

    assert updatable.up_to_date == True

    updatable.observable.x = 4

    assert updatable.up_to_date == False

    assert np.all(np.isclose(updatable.get_tensor(), 4 * tf.ones((2, 2), dtype=tf.float32)))

    assert updatable.up_to_date == True

    updatable.clear_tensor()

    assert updatable.up_to_date == False

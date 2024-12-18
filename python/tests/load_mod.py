# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

def test_import_wavelet():
    import wavelet
    assert wavelet.__version__ is not None


def test_simple_exec():
    import wavelet as wl
    a = wl.Tensor.const([1, 4, 1])
    assert a.max().scalar() == 4
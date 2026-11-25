import numpy as np
import pytest

from nemotron_image_mlx import noise


@pytest.mark.parametrize("ctr,key,expected", [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0), (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
])
def test_philox_known_answers(ctr, key, expected):
    out = noise.philox4x32_10(np.array(ctr, np.uint32), np.array(key, np.uint32))
    assert tuple(int(x) for x in out) == expected


def test_launch_geometry_h100():
    assert noise.n_threads(327680, "h100-sxm") == 1056 * 256


def test_exponential_row_is_exp1():
    q = noise.exponential_row_numpy(42, 0, 327680)
    assert q.shape == (327680,) and (q > 0).all()
    assert abs(q.mean() - 1.0) < 0.01

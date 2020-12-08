import numpy as np
from pyquaternion import Quaternion
import pytest
from typing import Any, Dict, List, Tuple

from nuscenes import NuScenes

from det3d.datasets.nuscenes import nusc_common

DATA_PATH = "/srv/share3/hchittanuru3/nuscenes-mini/"


class MockNuScenes(NuScenes):
    def __init__(
        self, sample_annotations: List[Dict[str, Any]], samples: List[Dict[str, Any]]
    ):
        self._sample_annotation = {r["token"]: r for r in sample_annotations}
        self._sample = {r["token"]: r for r in samples}

    def get(self, table_name: str, token: str) -> Dict[str, Any]:
        assert table_name in {"sample_annotation", "sample"}
        return getattr(self, "_" + table_name)[token]


def test_quaternion_yaw():
    assert nusc_common.quaternion_yaw(
        Quaternion([0, 0.7071, 0.7071, 0])
    ) == pytest.approx(1.5708, 1e-3)


def test_remove_close():
    points = np.array([[5, 1, 0.7], [3, 1, 3]])
    filtered_points = nusc_common.remove_close(points, 1.5)
    expected_points = np.array([[5, 0.7], [3, 3]])
    assert np.array_equal(filtered_points, expected_points)

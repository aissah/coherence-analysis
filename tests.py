


from datetime import datetime
from unittest.mock import MagicMock
import argparse
import numpy as np
import pytest
import pickle
from unittest import mock
from coherence_analysis.coherence_analysis import CoherenceAnalysis


@pytest.fixture
def valid_args():
    """Fixture for mock command-line arguments."""
    return {
        "method": "exact",
        "data_path": "test_data",
        "averaging_window_length": 60,
        "sub_window_length": 5,
        "overlap": 10,
        "time_range": "('06/01/23 07:32:09', '06/01/23 07:42:09')",
        "channel_range": "(0, 10)",
        "channel_offset": 2,
        "time_step": 0.002,
        "result_path": "test_results",
    }


@pytest.fixture
def instance():
    """Fixture for CoherenceAnalysis instance."""
    return CoherenceAnalysis()


def test_parse_args_valid(mocker, valid_args, instance):
    """Test valid argument parsing."""
    mocker.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(**valid_args),
    )

    instance._parse_args()

    assert instance.method == "exact"
    assert instance.data_path == "test_data"
    assert instance.averaging_window_length == 60
    assert instance.sub_window_length == 5
    assert instance.overlap == 10
    assert instance.channel_offset == 2
    assert instance.time_step == 0.002
    assert instance.save_location == "test_results"
    assert instance.time_range == [
        datetime.strptime("06/01/23 07:32:09", "%m/%d/%y %H:%M:%S"),
        datetime.strptime("06/01/23 07:42:09", "%m/%d/%y %H:%M:%S"),
    ]
    assert instance.channel_range == (0, 10)


def test_parse_args_invalid_method(mocker, instance):
    """Test invalid method raises ValueError."""
    invalid_args = {
        "method": "invalid_method",
        "data_path": "test_data",
        "averaging_window_length": 60,
        "sub_window_length": 5,
        "overlap": 10,
        "time_range": "('06/01/23 07:32:09', '06/01/23 07:42:09')",
        "channel_range": "(0, 10)",
        "channel_offset": 2,
        "time_step": 0.002,
        "result_path": "test_results",
    }
    mocker.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(**invalid_args),
    )

    with pytest.raises(
        ValueError, match="not available for coherence analysis"
    ):
        instance._parse_args()


@pytest.mark.benchmark
def test_read_data(mocker, instance):
    """Test data reading with mocked dascore.spool."""
    mock_spool = mocker.patch("dascore.spool", return_value=MagicMock())
    mock_spool_instance = mock_spool.return_value
    mock_spool_instance.get_contents.return_value = {
        "time_step": [MagicMock(total_seconds=MagicMock(return_value=0.002))]
    }
    mock_spool_instance.chunk.return_value = mock_spool_instance
    mock_spool_instance.select.return_value = mock_spool_instance

    instance.data_path = "test_data"
    instance.averaging_window_length = 60
    instance.sub_window_length = 5
    instance.time_range = [datetime.now(), datetime.now()]
    instance.channel_range = (0, 10)
    instance.channel_offset = 1

    instance.read_data()

    mock_spool.assert_called_once_with("test_data")
    mock_spool_instance.chunk.assert_called_once_with(time=60)
    mock_spool_instance.select.assert_any_call(time=instance.time_range)
    mock_spool_instance.select.assert_any_call(
        distance=mock.ANY, samples=False
    )


def test_run(mocker, instance):
    """Test coherence analysis execution."""
    mock_map = mocker.patch.object(instance, "spool", MagicMock())
    mock_map.map.return_value = [
        (np.random.rand(5, 5), np.random.rand(5, 5)) for _ in range(3)
    ]

    instance.spool = mock_map
    instance.time_step = 0.002
    instance.sub_window_length = 5
    instance.overlap = 10
    instance.method = "exact"

    instance.run()

    assert instance.detection_significance.shape[-1] == 3
    assert instance.eig_estimates.shape[-1] == 3


def test_save_results(mocker, instance):
    """Test saving results with mocked file I/O."""
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    mock_pickle_dump = mocker.patch("pickle.dump")

    instance.time_step = 0.002
    instance.averaging_window_length = 60
    instance.sub_window_length = 5
    instance.overlap = 10
    instance.channel_range = (0, 10)
    instance.channel_offset = 1
    instance.method = "exact"
    instance.save_location = "test_results"
    instance.contents = {
        "time_min": [datetime.now()],
        "time_max": [datetime.now()],
    }
    instance.detection_significance = np.random.rand(10, 10, 3)
    instance.eig_estimates = np.random.rand(10, 10, 3)

    instance.save_results()

    assert mock_open.call_count > 0
    assert mock_pickle_dump.call_count == 3

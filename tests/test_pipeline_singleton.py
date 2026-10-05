import pytest
from unittest.mock import patch
from crew.pipeline_singleton import warm

@patch("crew.pipeline_singleton.get_pipeline")
@patch("crew.pipeline_singleton.get_damage_analyzer")
def test_warm(mock_analyzer, mock_pipeline):
    warm()
    mock_pipeline.return_value.analyze.assert_called_once()

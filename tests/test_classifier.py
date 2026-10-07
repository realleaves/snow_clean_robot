"""tests/test_classifier.py -- three-level pollution classification boundaries."""
import pytest

from fusion.pollution_classifier import PollutionClassifier, PollutionLevel
from fusion.pollution_score import PollutionScorer
from perception.pollution_region import PollutionRegion
from tests.helpers import FUSION_CFG, LEVEL_CFG
from utils.errors import ConfigError, PerceptionError

#: spec section 14
BOUNDARY_TABLE = [
    (0.000, PollutionLevel.LIGHT),
    (0.349, PollutionLevel.LIGHT),
    (0.350, PollutionLevel.MEDIUM),
    (0.351, PollutionLevel.MEDIUM),
    (0.699, PollutionLevel.MEDIUM),
    (0.700, PollutionLevel.HEAVY),
    (0.701, PollutionLevel.HEAVY),
    (1.000, PollutionLevel.HEAVY),
]


@pytest.mark.parametrize('score,expected', BOUNDARY_TABLE)
def test_level_boundaries(score, expected):
    classifier = PollutionClassifier(light_max=0.35, medium_max=0.70)
    assert classifier.classify(score) is expected
    assert classifier.classify_name(score) == expected.name


def test_configured_thresholds_match_the_spec():
    classifier = PollutionClassifier(**LEVEL_CFG)
    assert classifier.light_max == pytest.approx(0.35)
    assert classifier.medium_max == pytest.approx(0.70)


@pytest.mark.parametrize('score', [-0.01, 1.01, float('nan'), float('inf')])
def test_out_of_range_score_is_rejected(score):
    with pytest.raises(PerceptionError):
        PollutionClassifier(0.35, 0.70).classify(score)


def test_non_numeric_score_is_rejected():
    with pytest.raises(PerceptionError):
        PollutionClassifier(0.35, 0.70).classify(None)
    with pytest.raises(PerceptionError):
        PollutionClassifier(0.35, 0.70).classify('HEAVY')


@pytest.mark.parametrize('light,medium', [(-0.1, 0.7), (0.8, 0.4), (0.0, 0.7), (0.35, 1.5)])
def test_invalid_thresholds_are_rejected(light, medium):
    with pytest.raises(ConfigError):
        PollutionClassifier(light, medium)


def test_classify_region_writes_the_level_back():
    region = PollutionRegion(1, (0, 0, 10, 10), (5, 5), 100.0, 0.5, 0.1,
                             pollution_score=0.9)
    assert PollutionClassifier(0.35, 0.70).classify_region(region) == 'HEAVY'
    assert region.pollution_level == 'HEAVY'


def test_classify_region_without_score_is_rejected():
    region = PollutionRegion(1, (0, 0, 10, 10), (5, 5), 100.0, 0.5, 0.1)
    with pytest.raises(PerceptionError):
        PollutionClassifier(0.35, 0.70).classify_region(region)


def test_scorer_and_classifier_agree_on_the_spec_example():
    scorer = PollutionScorer(FUSION_CFG)
    score = scorer.score(0.60, 0.80, 0.40)
    assert score == pytest.approx(0.62)
    assert PollutionClassifier(**LEVEL_CFG).classify_name(score) == 'MEDIUM'


@pytest.mark.parametrize('visual,humidity,area', [
    (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
])
def test_score_extremes_stay_in_unit_interval(visual, humidity, area):
    score = PollutionScorer(FUSION_CFG).score(visual, humidity, area)
    assert 0.0 <= score <= 1.0

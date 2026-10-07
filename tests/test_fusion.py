import unittest

from fusion.humidity_processor import HumidityCalibration, normalize_humidity
from fusion.pollution_classifier import PollutionClassifier, PollutionLevel
from fusion.pollution_score import PollutionScorer
from utils.errors import ConfigError, HumidityError, PerceptionError


class FusionTests(unittest.TestCase):
    def test_humidity_both_polarities(self):
        self.assertEqual(normalize_humidity(700, 700, 3500), 0)
        self.assertEqual(normalize_humidity(3500, 700, 3500), 1)
        self.assertEqual(normalize_humidity(3500, 3500, 700), 0)
        self.assertEqual(normalize_humidity(700, 3500, 700), 1)
        with self.assertRaises(HumidityError):
            normalize_humidity(1, 700, 700)
        calibration = HumidityCalibration(3500, 700)
        self.assertTrue(calibration.reversed_polarity)
        self.assertAlmostEqual(calibration.normalize(2100), 0.5)

    def test_score_validation_and_boundaries(self):
        scorer = PollutionScorer(dict(visual_weight=.5, humidity_weight=.3,
                                      area_weight=.2, max_area_ratio=.2))
        self.assertEqual(scorer.score(0, 0, 0), 0)
        self.assertEqual(scorer.score(1, 1, 1), 1)
        with self.assertRaises(PerceptionError):
            scorer.score(float('nan'), 0, 0)
        with self.assertRaises(ConfigError):
            PollutionScorer(dict(visual_weight=.5, humidity_weight=.5,
                                 area_weight=.2, max_area_ratio=.2))
        classifier = PollutionClassifier(.35, .70)
        self.assertEqual(classifier.classify(.349), PollutionLevel.LIGHT)
        self.assertEqual(classifier.classify(.35), PollutionLevel.MEDIUM)
        self.assertEqual(classifier.classify(.70), PollutionLevel.HEAVY)

    def test_spec_example_scores_to_medium(self):
        """V = 0.60, H = 0.80, A = 0.40 -> S = 0.62 -> MEDIUM."""
        scorer = PollutionScorer(dict(visual_weight=.5, humidity_weight=.3,
                                      area_weight=.2, max_area_ratio=.2))
        score = scorer.score(0.60, 0.80, 0.40)
        self.assertAlmostEqual(score, 0.62, places=9)
        self.assertEqual(PollutionClassifier(.35, .70).classify(score), PollutionLevel.MEDIUM)

    def test_invalid_inputs_are_rejected(self):
        scorer = PollutionScorer(dict(visual_weight=.5, humidity_weight=.3,
                                      area_weight=.2, max_area_ratio=.2))
        for bad in (-0.01, 1.01, float('inf'), None):
            with self.assertRaises(PerceptionError):
                scorer.score(bad, 0.5, 0.5)
        with self.assertRaises(HumidityError):
            normalize_humidity(-1, 700, 3500)
        with self.assertRaises(ConfigError):
            PollutionClassifier(0.8, 0.4)


if __name__ == '__main__':
    unittest.main()

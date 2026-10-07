import unittest
from fusion.humidity_processor import normalize_humidity
from fusion.pollution_score import PollutionScorer
from fusion.pollution_classifier import PollutionClassifier, PollutionLevel


class FusionTests(unittest.TestCase):
    def test_humidity_both_polarities(self):
        self.assertEqual(normalize_humidity(700, 700, 3500), 0)
        self.assertEqual(normalize_humidity(3500, 700, 3500), 1)
        self.assertEqual(normalize_humidity(3500, 3500, 700), 0)
        self.assertEqual(normalize_humidity(700, 3500, 700), 1)
        with self.assertRaises(ValueError):
            normalize_humidity(1, 700, 700)

    def test_score_validation_and_boundaries(self):
        scorer = PollutionScorer(dict(visual_weight=.5, humidity_weight=.3,
                                      area_weight=.2, max_area_ratio=.2))
        self.assertEqual(scorer.score(0, 0, 0), 0)
        self.assertEqual(scorer.score(1, 1, 1), 1)
        with self.assertRaises(ValueError):
            scorer.score(float('nan'), 0, 0)
        with self.assertRaises(ValueError):
            PollutionScorer(dict(visual_weight=.5, humidity_weight=.5,
                                 area_weight=.2, max_area_ratio=.2))
        classifier = PollutionClassifier(.35, .70)
        self.assertEqual(classifier.classify(.349), PollutionLevel.LIGHT)
        self.assertEqual(classifier.classify(.35), PollutionLevel.MEDIUM)
        self.assertEqual(classifier.classify(.70), PollutionLevel.HEAVY)

import unittest
from training_validation import record_validation
class ValidationTests(unittest.TestCase):
    def test_selection(self):
        state, improved = record_validation({}, 2., 0, 'fixed')
        self.assertTrue(improved)
        state, improved = record_validation(state, 1., 1, 'fixed')
        self.assertTrue(improved)
        self.assertEqual(state['latest']['improvement_percent'], 50.)
        state, improved = record_validation(state, 1.5, 2, 'fixed')
        self.assertFalse(improved)
        self.assertEqual(state['best_epoch'], 1)
        self.assertEqual(state['latest']['epoch'], 2)
        state, improved = record_validation(state, 1., 3, 'fixed')
        self.assertFalse(improved)
    def test_nonfinite(self):
        for score in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError): record_validation({}, score, 0, 'fixed')
if __name__ == '__main__': unittest.main()

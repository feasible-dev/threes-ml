import unittest

from external_agent import encode_public_observation
from threes import Game


class ExternalAgentObservationTests(unittest.TestCase):
    def test_public_21_value_contract(self):
        game = Game(seed=42)
        observation, legal = encode_public_observation(game)
        self.assertEqual(observation.tolist(), [[
            1, 0, 0, 3, 3, 2, 1, 0, 0, 2, 0, 1, 2, 0, 2, 0,
            3, 1, 0, 1, 2,
        ]])
        self.assertEqual(legal, (True, True, True, True))

    def test_bonus_preview_uses_first_public_option(self):
        game = Game(seed=42)
        game.next_hint = (6, 12, 24)
        observation, _ = encode_public_observation(game)
        self.assertEqual(int(observation[0, 16]), 4)
        self.assertEqual(observation[0, 17:].tolist(), [1, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()

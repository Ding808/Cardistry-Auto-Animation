"""Retirement boundaries only; no synthetic or real population scaling runs."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from cardcap.solve.apply_hand_prior import apply_hand_prior, transform_hand_capture, main

class ApplyHandPriorTests(unittest.TestCase):
    def test_retired_api_never_mutates_payload_or_files(self):
        payload = {"hands": [{"sample": [1, 2, 3]}]}
        original = deepcopy(payload)
        with self.assertRaisesRegex(RuntimeError, "retired"):
            transform_hand_capture(payload, None)
        self.assertEqual(payload, original)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"unchanged.txt"; target.write_text("keep",encoding="utf-8")
            new_output = Path(folder)/"new_output"
            with self.assertRaisesRegex(RuntimeError, "retired"):
                apply_hand_prior(target, target, new_output)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")
            self.assertFalse(new_output.exists())

    def test_retired_cli_fails_before_parsing_or_reading_paths(self):
        self.assertEqual(main(["--capture", "does-not-exist", "--output", "unused"]), 2)

if __name__ == "__main__": unittest.main()

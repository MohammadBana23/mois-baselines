from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch

from fcg_ia5.src.runner import run
from fcg_ia5.src.algorithm import GROUPS
from fcg_paper_faithful.src.synthetic import evaluate, predict
from tests.fixtures import development, support


class OfflineIntegrationTests(unittest.TestCase):
    def test_prompt_boundary(self):
        dev = development()
        self.assertEqual(evaluate(("a",), dev), evaluate(("a",), tuple(replace(r, label=1-r.label) for r in dev)))
        with self.assertRaises(ValueError):
            predict((), '{"id": "synthetic:dev:0", "signal": 1, "label": 0}')

    def test_full_resume_quota_budget_and_no_op(self):
        rows, dev = support(), development()
        params = tomllib.loads(Path("fcg_ia5/configs/synthetic.toml").read_text())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "resume.checkpoint.json"
            paused = run(rows, dev, checkpoint=path, max_batches=12, **params)
            self.assertEqual((paused["status"], paused["selected"]), ("PAUSED", None))
            self.assertEqual(paused["predictions_completed"], 96)
            resumed = run(rows, dev, checkpoint=path, **params)
            replay = run(rows, dev, checkpoint=Path(directory) / "full.checkpoint.json", **params)
            self.assertEqual(replay, resumed)
            self.assertEqual(resumed["predictions_completed"], 648)
            self.assertEqual(resumed["predictions_attempted_or_reserved"], 648)
            self.assertEqual(resumed["fitness_evaluations"], 80)
            ids = resumed["selected"]
            self.assertEqual(len(set(ids)), 5)
            picked = [r for r in rows if r.identifier in ids]
            self.assertEqual(set((r.sex, r.race) for r in picked), set(GROUPS))
            self.assertEqual(len({(r.sex, r.race, r.label) for r in picked}), 5)
            self.assertIn(sum(r.label for r in picked), (2, 3))
            with patch("fcg_ia5.src.runner.fake_response", side_effect=AssertionError("prediction was invoked")):
                self.assertEqual(run(rows, dev, checkpoint=path, **params), replay)
            with self.assertRaises(ValueError):
                run(rows, dev[::-1], checkpoint=path, **params)
            wrapper = json.loads(path.read_text())
            wrapper["state"]["attempts"] -= 8
            path.write_text(json.dumps(wrapper))
            with self.assertRaisesRegex(ValueError, "integrity"):
                run(rows, dev, checkpoint=path, **params)

    def test_failure_reservation_and_resume(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "failure.checkpoint.json"
            options = {"checkpoint": path, "budget": 656}
            with patch("fcg_ia5.src.runner.fake_response", return_value='{"prediction":2}'):
                with self.assertRaises(ValueError):
                    run(support(), development(), **options)
            state = json.loads(path.read_text())["state"]
            self.assertEqual((state["attempts"], state["events"]), (1, []))
            result = run(support(), development(), **options)
            self.assertEqual((result["predictions_completed"], result["predictions_attempted_or_reserved"]), (648, 649))

    def test_insufficient_or_exhausted_budget_and_real_mode_guard(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "budget.checkpoint.json"
            with self.assertRaises(PermissionError):
                run((), (), checkpoint=path, budget=0, mode="calibration")
            with self.assertRaises(ValueError):
                run(support(), development(), checkpoint=path, budget=647)
            self.assertFalse(path.exists())
            with patch("fcg_ia5.src.runner.fake_response", return_value='{"prediction":2}'):
                with self.assertRaises(ValueError):
                    run(support(), development(), checkpoint=path, budget=648)
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                run(support(), development(), checkpoint=path, budget=648)



class PromptJournalTests(unittest.TestCase):
    def test_render_no_query_label_and_resume_without_repeating(self):
        from fcg_ia5.src.runner import fake_response

        def capture(checkpoint, rows):
            with patch("fcg_ia5.src.runner.fake_response", wraps=fake_response) as fake:
                run(support(), rows, checkpoint=checkpoint, budget=648, max_batches=2)
                return [call.args[0] for call in fake.call_args_list]

        rows = development()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = capture(root / "first.json", rows)
            second = capture(root / "second.json", tuple(replace(r, label=1-r.label) for r in rows))
            self.assertEqual(first, second)
            demo_text, query = first[8][-1]["content"].split("\nUnlabeled query:\n")
            self.assertEqual(len(demo_text.splitlines()[1:]), 5)
            self.assertEqual(set(json.loads(query)), {"features"})
            self.assertEqual(len(json.loads(query)["features"]), 12)
            self.assertTrue(all("label" in json.loads(line) for line in demo_text.splitlines()[1:]))

            path = root / "resources.json"
            safe = (7 * 2**30, 0)
            low = (511 * 2**20, 0)
            with patch("fcg_ia5.src.runner.fake_response", wraps=fake_response) as fake:
                with self.assertRaisesRegex(RuntimeError, "Resource limit"):
                    run(support(), rows, checkpoint=path, budget=648, max_batches=1,
                        sample_resources=iter([safe, safe, low, low]).__next__)
                self.assertEqual(fake.call_count, 2)
                fake.reset_mock()
                run(support(), rows, checkpoint=path, budget=648, max_batches=1)
                self.assertEqual(fake.call_count, 6)
            event = json.loads(path.read_text())["state"]["events"][0]
            self.assertEqual([q["query"] for q in event["queries"]], [r.identifier for r in rows])
            self.assertTrue(all(q["status"] == "ok" and len(q["prompt_sha256"]) == 64
                                for q in event["queries"]))

    def test_unknown_attempt_stops_without_retry(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "timeout.json"
            with patch("fcg_ia5.src.runner.fake_response", side_effect=TimeoutError):
                with self.assertRaises(TimeoutError):
                    run(support(), development(), checkpoint=path, budget=648, max_batches=1)
            with patch("fcg_ia5.src.runner.fake_response", side_effect=AssertionError):
                with self.assertRaisesRegex(RuntimeError, "Unresolved"):
                    run(support(), development(), checkpoint=path, budget=648, max_batches=1)

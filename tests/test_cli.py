import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

from author_match.cli import main


class CommandLineTest(unittest.TestCase):
    def invoke(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = main(list(map(str, args)))
        self.assertEqual(code, 0)
        return json.loads(out.getvalue())

    def test_prepare_train_reload_evaluate_and_compare(self):
        rng = random.Random(7)
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            source, data, model = base / "posts.jsonl", base / "prepared", base / "test.joblib"
            with source.open("w") as out:
                for writer in range(48):
                    for post in range(8):
                        text = " ".join(rng.choices("one two three four five six seven eight".split(), k=10))
                        row = dict(id=f"p{writer}-{post}", author=str(writer), thread=f"t{writer}-{post}",
                                   text=text + ("!" * (writer % 3 + 1)), community=str(post % 2))
                        print(json.dumps(row), file=out)
            audit = self.invoke("prepare", source, data)
            self.assertGreater(audit["splits"]["test"]["pairs"], 0)
            # The model is selected without reading final-test data.
            final = data / "test.pairs.jsonl"
            final.rename(base / "locked")
            fit = self.invoke("train", data, model, "--false-match-rate", "0.15",
                              "--false-reject-rate", "0.15")
            (base / "locked").rename(final)
            self.assertEqual(set(fit["development_candidates"]), {"word", "char", "style"})
            report_file = base / "report.json"
            result = self.invoke("evaluate", data, model, "--split", "test",
                                 "--bootstrap-samples", "0", "--output", report_file)
            self.assertEqual(result, json.loads(report_file.read_text()))
            self.assertEqual(result["overall"]["pairs"], audit["splits"]["test"]["pairs"])
            external = base / "external"
            self.invoke("prepare", source, external, "--external")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(main(["evaluate", str(external), str(model), "--split", "external"]), 2)
            self.assertIn("writer IDs", err.getvalue())
            left, right = base / "left", base / "right"
            left.write_text("Yeah, actually that sounds pretty good to me.")
            right.write_text("idk about this one but we could try again")
            ab = self.invoke("compare", left, right, "--model", model)
            ba = self.invoke("compare", right, left, "--model", model)
            self.assertAlmostEqual(ab["score"], ba["score"])
            self.assertEqual(ab["decision"], ba["decision"])
            left.write_text("> the entire input was quoted")
            denied = self.invoke("compare", left, right, "--model", model)
            self.assertEqual(denied["decision"], "inconclusive")
            self.assertIsNone(denied["score"])

    def test_cli_gives_concise_errors(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["compare", "a", "b", "--model", "/missing/model"]), 2)
        self.assertIn("author-match:", err.getvalue())

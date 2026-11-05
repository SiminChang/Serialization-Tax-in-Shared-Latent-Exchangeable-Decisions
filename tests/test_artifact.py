import json
import unittest
from pathlib import Path

from make_control_tables import main as verify_tables
from make_figures import write_movielens_audit_tables


ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def test_public_assets_and_scope(self):
        self.assertTrue((ROOT / "paper/serialization_tax.pdf").is_file())
        for name in ("posterior_interface", "taxscore_routing", "movielens_results", "covtype_mechanism"):
            self.assertGreater((ROOT / "assets" / (name + ".png")).stat().st_size, 10000)
        for path in ROOT.rglob("*"):
            if not path.is_file() or any(part in (".git", "__pycache__", "figures") for part in path.parts):
                continue
            self.assertNotIn("rebut" + "tal", path.relative_to(ROOT).as_posix().lower())
            if path.suffix in (".md", ".py", ".json"):
                content = path.read_text(encoding="utf-8").lower()
                self.assertNotIn("rebut" + "tal", content)
                self.assertNotIn("review" + "er", content)

    def test_paired_table_shows_means_without_intervals(self):
        import re
        data = json.loads((ROOT / "results/aggregate_results.json").read_text())
        (ROOT / "figures").mkdir(exist_ok=True)
        write_movielens_audit_tables(data)
        table = (ROOT / "figures/movielens_audit_tables.tex").read_text()
        paired = table.split(r"\label{tab:movie-paired-ci}", 1)[1].split(r"\end{longtable}", 1)[0]
        self.assertNotRegex(paired, r"\[-?\d+\.\d+,\s*-?\d+\.\d+\]")
        self.assertIn("NLL wins/5", paired)
        self.assertTrue(re.search(r"\d/5", paired))

    def test_paired_records_and_latency(self):
        verify_tables()

    def test_aggregate_studies(self):
        data = json.loads((ROOT / "results/aggregate_results.json").read_text())
        for key in ("movielens_proxy_robustness", "movielens_threshold_robustness",
                    "batch_inspection_benchmark", "covtype_threshold_benchmark"):
            self.assertIn(key, data)

    def test_plot_seed_counts(self):
        data = json.loads((ROOT / "results/plot_seed_metrics.json").read_text())
        for cfg in data.values():
            for model in cfg["high"].values():
                for values in model.values():
                    self.assertEqual(len(values), 5)
                    self.assertTrue(all(isinstance(v, (float, int)) for v in values))


if __name__ == "__main__":
    unittest.main()

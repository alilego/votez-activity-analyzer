import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import full_update


class FullUpdateTests(unittest.TestCase):
    def test_deployment_removes_only_obsolete_generated_identity_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src,dst = root/'outputs',root/'frontend'
            (src/'members').mkdir(parents=True)
            (dst/'members').mkdir(parents=True)
            (src/'members/interventions_deputat_cdep_2024_2_18_bara.json').write_text('{}')
            (dst/'members/interventions_deputat_16_bara.json').write_text('{}')
            (dst/'members/manual.json').write_text('{"preserve":true}')
            full_update.deploy_analyzer_outputs(src,dst)
            self.assertFalse((dst/'members/interventions_deputat_16_bara.json').exists())
            self.assertTrue((dst/'members/manual.json').exists())
            self.assertTrue((dst/'members/interventions_deputat_cdep_2024_2_18_bara.json').exists())

    def test_failed_export_cannot_deploy_preexisting_outputs(self):
        with patch.object(sys, "argv", ["full_update.py", "--skip-scrape", "--skip-sync", "--skip-pipeline", "--skip-productivity", "--skip-crawler", "--skip-adopted-law-enrichment"]), \
                patch.object(full_update, "run_export_outputs", return_value=False), \
                patch.object(full_update, "deploy_db") as deploy:
            self.assertEqual(full_update.main(), 1)
            deploy.assert_not_called()

    def test_invalid_exports_cannot_overwrite_frontend(self):
        with patch.object(sys, "argv", ["full_update.py", "--only-step", "8"]), \
                patch.object(full_update.subprocess, "run", return_value=SimpleNamespace(returncode=1)), \
                patch.object(full_update, "deploy_analyzer_outputs") as deploy, \
                patch.object(full_update, "deploy_db") as deploy_db:
            self.assertEqual(full_update.main(), 1)
            deploy.assert_not_called()
            deploy_db.assert_not_called()

    def test_crawler_receives_new_laws_only_scope(self):
        with patch.object(full_update.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            self.assertTrue(full_update.run_crawler(True, True, True))
        self.assertIn("--hydrate-new-laws-only", run.call_args.args[0])

    def test_scrape_failure_stops_before_analysis_and_deploy(self):
        with patch.object(sys, "argv", ["full_update.py"]), \
                patch.object(full_update, "run_scraper", return_value=False), \
                patch.object(full_update, "sync_stenograms") as sync, \
                patch.object(full_update, "run_pipeline") as pipeline, \
                patch.object(full_update, "deploy_db") as deploy:
            self.assertEqual(full_update.main(), 1)
            sync.assert_not_called()
            pipeline.assert_not_called()
            deploy.assert_not_called()


if __name__ == "__main__":
    unittest.main()

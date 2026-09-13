"""Consent reuse and release metadata checks without installation or downloads."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))

from cardcap import __version__
from setup_runtime import MODEL_TERMS_COMMIT, accepted_licenses, manifest

NOTICE_HASH = "a" * 64


def saved_acceptance(**changes):
    record = {"release": "v0.0.1", "runtime_license_notice_sha256": NOTICE_HASH,
              "accepted_model_licenses": True, "model_terms_commit": MODEL_TERMS_COMMIT}
    record.update(changes)
    return record


class SetupLicenseTests(unittest.TestCase):
    def test_existing_release_consent_survives_version_upgrade(self):
        self.assertEqual(accepted_licenses(saved_acceptance(), NOTICE_HASH),
                         {"runtime": True, "models": True})

    def test_release_number_does_not_affect_matching_terms(self):
        for release in ("v0.0.1", "v0.0.2", "v99.0.0", None):
            with self.subTest(release=release):
                self.assertEqual(accepted_licenses(saved_acceptance(release=release), NOTICE_HASH),
                                 {"runtime": True, "models": True})

    def test_runtime_notice_change_requires_runtime_consent(self):
        self.assertEqual(accepted_licenses(saved_acceptance(), "b" * 64),
                         {"runtime": False, "models": True})

    def test_model_terms_change_requires_model_consent(self):
        record = saved_acceptance(model_terms_commit="different-terms")
        self.assertEqual(accepted_licenses(record, NOTICE_HASH), {"runtime": True, "models": False})

    def test_missing_receipt_does_not_grant_consent(self):
        self.assertEqual(accepted_licenses({}, NOTICE_HASH), {"runtime": False, "models": False})

    def test_model_consent_must_be_a_boolean(self):
        for value in ("true", "false", 1, None, False):
            with self.subTest(value=value):
                self.assertFalse(accepted_licenses(saved_acceptance(accepted_model_licenses=value), NOTICE_HASH)["models"])

    def test_empty_notice_hash_does_not_grant_consent(self):
        self.assertFalse(accepted_licenses(saved_acceptance(runtime_license_notice_sha256=""), "")["runtime"])

    def test_release_descriptor_and_runtime_url_match_package(self):
        descriptor = json.loads((PIPELINE.parent / "CardistryCapture.uplugin").read_text(encoding="utf-8-sig"))
        runtime = manifest()
        self.assertEqual(descriptor["VersionName"], __version__)
        self.assertEqual(runtime["release"], "v" + __version__)
        self.assertTrue(runtime["archive_url"].endswith(
            f"/v{__version__}/CardistryCapture-RuntimeDeps-v{__version__}.zip"))

    @unittest.skipUnless(shutil.which("powershell.exe"), "Windows PowerShell is required")
    def test_powershell_reuses_exactly_the_same_consent_cases(self):
        cases = [
            {"record": saved_acceptance(), "hash": NOTICE_HASH},
            {"record": saved_acceptance(release="v99.0.0"), "hash": NOTICE_HASH},
            {"record": saved_acceptance(), "hash": "b" * 64},
            {"record": saved_acceptance(model_terms_commit="different-terms"), "hash": NOTICE_HASH},
            {"record": {}, "hash": NOTICE_HASH},
            {"record": saved_acceptance(accepted_model_licenses="true"), "hash": NOTICE_HASH},
            {"record": saved_acceptance(accepted_model_licenses=1), "hash": NOTICE_HASH},
            {"record": saved_acceptance(runtime_license_notice_sha256=""), "hash": ""},
        ]
        # Extract only the actual production helper; never invoke Setup's body.
        script = r'''
param([string]$SetupPath, [string]$CasesPath)
$ErrorActionPreference = 'Stop'
$tokens = $null
$parseErrors = $null
$tree = [System.Management.Automation.Language.Parser]::ParseFile($SetupPath, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw 'Setup.ps1 contains a parse error.' }
$helper = $tree.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-SavedLicenseAcceptance' }, $true)
if ($null -eq $helper) { throw 'Production consent helper is missing.' }
. ([scriptblock]::Create($helper.Extent.Text))
$cases = Get-Content -LiteralPath $CasesPath -Raw | ConvertFrom-Json
$results = @(foreach ($case in $cases) {
    $result = Get-SavedLicenseAcceptance $case.record $case.hash
    @{ runtime = $result.Runtime; models = $result.Models }
})
$results | ConvertTo-Json -Compress
'''
        with tempfile.TemporaryDirectory(prefix="cardistry-consent-") as directory:
            folder = Path(directory)
            script_path, cases_path = folder / "check.ps1", folder / "cases.json"
            script_path.write_text(script, encoding="utf-8")
            cases_path.write_text(json.dumps(cases), encoding="utf-8")
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(script_path), "-SetupPath", str(PIPELINE.parent / "Scripts/Setup.ps1"),
                 "-CasesPath", str(cases_path)], capture_output=True, text=True, check=True)
        expected = [accepted_licenses(case["record"], case["hash"]) for case in cases]
        self.assertEqual(json.loads(result.stdout), expected)


if __name__ == "__main__":
    unittest.main()

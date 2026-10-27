import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pipeline.validation_selection import (
    CAA_LAYERS,
    CAA_STRENGTHS,
    CURVEBALL_CAA_RECIPE,
    CAA_MODEL,
    assert_curveball_caa_recipe_configuration,
    assert_locked_configuration,
    method_strength,
    write_manifest,
)
from scripts.build_validation_jobs import (
    _caa_select_jobs,
    _jail_select_jobs,
    _locked_test_jobs,
)


def _arg_value(argv, flag):
    return argv[argv.index(flag) + 1]


def _curveball_caa_manifest():
    return {
        "schema_version": 1,
        "experiment": "caa",
        "grid": {
            "curveball": {
                **CURVEBALL_CAA_RECIPE,
                "models": [CAA_MODEL],
                "layers": list(CAA_LAYERS),
                "alphas": list(CAA_STRENGTHS),
                "strength_parameter": "curveball_alpha",
            }
        },
        "selections": [
            {
                "model": CAA_MODEL,
                "method": "curveball",
                "behavior": "hallucination",
                "prompt_condition": "none",
                "application_mode": "all_tokens",
                "strength_parameter": "curveball_alpha",
                "selected": {"layer": 23, "strength": 20.0},
            }
        ],
    }


class CurveballCaaJobTests(unittest.TestCase):
    def test_caa_job_counts_and_exact_recipe(self):
        all_jobs = _caa_select_jobs(
            SimpleNamespace(
                model_path=f"/models/{CAA_MODEL}",
                artifact_root="/artifacts/caa",
                prompt_condition="all",
            )
        )
        self.assertEqual(len(all_jobs), 1440)
        curveball_jobs = [job for job in all_jobs if "-curveball-" in job["id"]]
        self.assertEqual(len(curveball_jobs), 720)

        curveball_only_jobs = _caa_select_jobs(
            SimpleNamespace(
                model_path=f"/models/{CAA_MODEL}",
                artifact_root="/artifacts/caa-curveball",
                prompt_condition="all",
                method="curveball",
            )
        )
        self.assertEqual(len(curveball_only_jobs), 720)
        self.assertTrue(
            all("-curveball-" in job["id"] for job in curveball_only_jobs)
        )

        argv = curveball_jobs[0]["argv"]
        expected = {
            "--steer_method": "curveball",
            "--pos": "-2",
            "--curveball_extraction_source": "caa",
            "--curveball_kernel": "poly",
            "--curveball_degree": "3",
            "--curveball_coef0": "1.0",
            "--curveball_k": "20",
            "--curveball_fit_inverse_alpha": "1.0",
            "--evaluation_interventions": "curveball",
        }
        for flag, value in expected.items():
            self.assertEqual(_arg_value(argv, flag), value)
        self.assertNotIn("--curveball_gamma", argv)
        self.assertIn(float(_arg_value(argv, "--curveball_alpha")), CAA_STRENGTHS)

        one_prompt_jobs = _caa_select_jobs(
            SimpleNamespace(
                model_path=f"/models/{CAA_MODEL}",
                artifact_root="/artifacts/caa",
                prompt_condition="none",
            )
        )
        self.assertEqual(len(one_prompt_jobs), 480)

        negative_curveball_only_jobs = _caa_select_jobs(
            SimpleNamespace(
                model_path=f"/models/{CAA_MODEL}",
                artifact_root="/artifacts/caa-curveball-neg",
                prompt_condition="neg",
                method="curveball",
            )
        )
        self.assertEqual(len(negative_curveball_only_jobs), 240)

    def test_non_gemma_caa_rejected_and_jailbreak_counts(self):
        with self.assertRaisesRegex(ValueError, "only gemma-2-9b-it"):
            _caa_select_jobs(
                SimpleNamespace(
                    model_path="/models/Llama-3.1-8B-Instruct",
                    artifact_root="/artifacts/caa",
                    prompt_condition="all",
                )
            )

        gemma_jobs = _jail_select_jobs(
            SimpleNamespace(
                model_path=f"/models/{CAA_MODEL}",
                artifact_root="/artifacts/jail",
                application_mode="all",
            )
        )
        llama_jobs = _jail_select_jobs(
            SimpleNamespace(
                model_path="/models/Llama-3.1-8B-Instruct",
                artifact_root="/artifacts/jail",
                application_mode="all",
            )
        )
        self.assertEqual(len(gemma_jobs), 81)
        self.assertEqual(len(llama_jobs), 81)

    def test_locked_job_reconstructs_curveball_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest_path = directory / "caa_curveball.json"
            write_manifest(_curveball_caa_manifest(), manifest_path)
            jobs = _locked_test_jobs(
                SimpleNamespace(
                    model_path=f"/models/{CAA_MODEL}",
                    artifact_root=str(directory / "artifacts"),
                    manifest=str(manifest_path),
                )
            )
        self.assertEqual(len(jobs), 2)
        curveball = next(job for job in jobs if job["id"].endswith("-curveball"))
        argv = curveball["argv"]
        self.assertEqual(_arg_value(argv, "--steer_method"), "curveball")
        self.assertEqual(_arg_value(argv, "--curveball_alpha"), "20.0")
        self.assertEqual(_arg_value(argv, "--curveball_extraction_source"), "caa")
        self.assertEqual(_arg_value(argv, "--pos"), "-2")
        self.assertEqual(_arg_value(argv, "--evaluation_interventions"), "curveball")


class CurveballCaaRecipeTests(unittest.TestCase):
    def setUp(self):
        self.manifest = _curveball_caa_manifest()
        self.kwargs = {
            "model": CAA_MODEL,
            "kernel": "poly",
            "degree": 3,
            "gamma": None,
            "coef0": 1.0,
            "kpca_components": 20,
            "inverse_transform_alpha": 1.0,
            "extraction_source": "caa",
            "position": -2,
        }

    def test_strength_mapping_and_recipe_acceptance(self):
        self.assertEqual(
            method_strength("curveball", eta=1.0, curveball_alpha=20.0),
            20.0,
        )
        assert_curveball_caa_recipe_configuration(
            self.manifest,
            **self.kwargs,
        )

    def test_recipe_and_locked_alpha_rejections(self):
        mismatches = {
            "model": "Llama-3.1-8B-Instruct",
            "kernel": "rbf",
            "degree": 2,
            "gamma": 0.5,
            "coef0": 0.0,
            "kpca_components": 10,
            "inverse_transform_alpha": 0.5,
            "extraction_source": "legacy",
            "position": -1,
        }
        for field, value in mismatches.items():
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    assert_curveball_caa_recipe_configuration(
                        self.manifest,
                        **{**self.kwargs, field: value},
                    )

        bad_manifest = _curveball_caa_manifest()
        bad_manifest["grid"]["curveball"]["degree"] = 2
        with self.assertRaisesRegex(ValueError, "Manifest CAA Curveball recipe mismatch"):
            assert_curveball_caa_recipe_configuration(
                bad_manifest,
                **self.kwargs,
            )

        with self.assertRaisesRegex(ValueError, "does not match"):
            assert_locked_configuration(
                self.manifest,
                experiment="caa",
                model=CAA_MODEL,
                method="curveball",
                mode="all_tokens",
                layer=23,
                strength=50.0,
                behavior="hallucination",
                prompt_condition="none",
            )


if __name__ == "__main__":
    unittest.main()

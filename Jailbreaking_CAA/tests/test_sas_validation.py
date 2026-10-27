import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.sas_validation import (
    SAS_COEFFICIENTS,
    SAS_LAYERS,
    SAS_MODEL,
    SAS_RECIPES,
    assert_sas_recipe_configuration,
)
from pipeline.validation_selection import (
    CAA_LAYERS,
    CAA_STRENGTHS,
    CURVEBALL_CAA_RECIPE,
    JAILBREAK_STRENGTHS,
    select_caa_candidates,
    select_jailbreak_candidates,
    write_manifest,
)
from scripts.build_validation_jobs import (
    _caa_select_jobs,
    _jail_select_jobs,
    _locked_test_jobs,
)
from scripts.run_locked_tinybench import _validate_steer_config


def _arg_value(argv, flag):
    return argv[argv.index(flag) + 1]


def _selected_coefficient(experiment):
    return 20.0 if experiment == "caa" else -20.0


def _sas_manifest(experiment, mode="all_tokens"):
    recipe = dict(SAS_RECIPES[experiment])
    coefficient = _selected_coefficient(experiment)
    selection = {
        "model": SAS_MODEL,
        "method": "sas",
        "application_mode": mode,
        "strength_parameter": "sas_coefficient",
        "selected": {
            "layer": 23,
            "strength": coefficient,
        },
    }
    if experiment == "caa":
        selection.update({"behavior": "hallucination", "prompt_condition": "none"})
    return {
        "schema_version": 1,
        "experiment": experiment,
        "grid": {
            "sas": {
                **recipe,
                "models": [SAS_MODEL],
                "layers": list(SAS_LAYERS),
                "coefficients": list(SAS_COEFFICIENTS[experiment]),
                "strength_parameter": "sas_coefficient",
            }
        },
        "selections": [selection],
    }


class SasJobGenerationTests(unittest.TestCase):
    def test_caa_and_jailbreak_job_counts(self):
        caa_jobs = _caa_select_jobs(
            SimpleNamespace(
                model_path=f"/models/{SAS_MODEL}",
                artifact_root="/artifacts/caa",
                prompt_condition="all",
            )
        )
        self.assertEqual(len(caa_jobs), 1440)
        self.assertFalse(any("-sas-" in job["id"] for job in caa_jobs))

        caa_sas_only = _caa_select_jobs(
            SimpleNamespace(
                model_path=f"/models/{SAS_MODEL}",
                artifact_root="/artifacts/caa-sas",
                prompt_condition="all",
                method="sas",
            )
        )
        self.assertEqual(len(caa_sas_only), 720)
        self.assertTrue(all("-sas-" in job["id"] for job in caa_sas_only))
        self.assertEqual(
            {
                float(_arg_value(job["argv"], "--sas_coefficient"))
                for job in caa_sas_only
            },
            set(SAS_COEFFICIENTS["caa"]),
        )
        self.assertEqual(_arg_value(caa_sas_only[0]["argv"], "--sae_width"), "131k")
        self.assertEqual(_arg_value(caa_sas_only[0]["argv"], "--pos"), "-2")

        gemma_jobs = _jail_select_jobs(
            SimpleNamespace(
                model_path=f"/models/{SAS_MODEL}",
                artifact_root="/artifacts/jail",
                application_mode="all",
            )
        )
        self.assertEqual(len(gemma_jobs), 81)
        self.assertFalse(any("-sas-" in job["id"] for job in gemma_jobs))

        gemma_sas_only = _jail_select_jobs(
            SimpleNamespace(
                model_path=f"/models/{SAS_MODEL}",
                artifact_root="/artifacts/jail-sas",
                application_mode="all",
                method="sas",
            )
        )
        self.assertEqual(len(gemma_sas_only), 41)
        self.assertEqual(
            sum("-sas-" in job["id"] for job in gemma_sas_only),
            40,
        )
        self.assertEqual(
            sum(job["id"].endswith("-baseline") for job in gemma_sas_only),
            1,
        )
        gemma_sas = [job for job in gemma_sas_only if "-sas-" in job["id"]]
        self.assertEqual(
            _arg_value(gemma_sas[0]["argv"], "--ce_loss_batch_size"), "1"
        )

        llama_jobs = _jail_select_jobs(
            SimpleNamespace(
                model_path="/models/Llama-3.1-8B-Instruct",
                artifact_root="/artifacts/jail",
                application_mode="all",
            )
        )
        self.assertEqual(len(llama_jobs), 81)
        self.assertFalse(any("-sas-" in job["id"] for job in llama_jobs))
        with self.assertRaisesRegex(ValueError, "only gemma-2-9b-it"):
            _jail_select_jobs(
                SimpleNamespace(
                    model_path="/models/Llama-3.1-8B-Instruct",
                    artifact_root="/artifacts/jail-sas",
                    application_mode="all",
                    method="sas",
                )
            )

    def test_single_mode_gemma_job_count(self):
        jobs = _jail_select_jobs(
            SimpleNamespace(
                model_path=f"/models/{SAS_MODEL}",
                artifact_root="/artifacts/jail",
                application_mode="prompt_only",
            )
        )
        self.assertEqual(len(jobs), 41)
        sas_only = _jail_select_jobs(
            SimpleNamespace(
                model_path=f"/models/{SAS_MODEL}",
                artifact_root="/artifacts/jail",
                application_mode="prompt_only",
                method="sas",
            )
        )
        self.assertEqual(len(sas_only), 21)

    def test_locked_jobs_reconstruct_sas_recipes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for experiment, expected_width, expected_coefficient in (
                ("caa", "131k", "20.0"),
                ("jailbreak", "16k", "-20.0"),
            ):
                manifest_path = directory / f"{experiment}.json"
                mode = "all_tokens" if experiment == "caa" else "prompt_only"
                write_manifest(_sas_manifest(experiment, mode), manifest_path)
                jobs = _locked_test_jobs(
                    SimpleNamespace(
                        model_path=f"/models/{SAS_MODEL}",
                        artifact_root=str(directory / "artifacts"),
                        manifest=str(manifest_path),
                    )
                )
                sas_job = next(job for job in jobs if job["id"].endswith("-sas"))
                argv = sas_job["argv"]
                self.assertEqual(_arg_value(argv, "--steer_method"), "sas")
                self.assertEqual(_arg_value(argv, "--sae_width"), expected_width)
                self.assertEqual(
                    _arg_value(argv, "--sas_coefficient"), expected_coefficient
                )
                self.assertEqual(
                    _arg_value(argv, "--evaluation_interventions"), "sas"
                )


class SasRecipeAndTinyBenchTests(unittest.TestCase):
    def test_recipe_rejects_unsupported_model_and_mismatch(self):
        manifest = _sas_manifest("jailbreak", "prompt_only")
        recipe = SAS_RECIPES["jailbreak"]
        kwargs = {
            "experiment": "jailbreak",
            "model": SAS_MODEL,
            "coefficient": _selected_coefficient("jailbreak"),
            "sae_release": recipe["sae_release"],
            "sae_width": recipe["sae_width"],
            "feature_freq_threshold": recipe["feature_freq_threshold"],
            "extraction_source": recipe["extraction_source"],
            "position": recipe["position"],
            "use_delta_correction": recipe["use_delta_correction"],
            "apply_post_activation": recipe["apply_post_activation"],
        }
        assert_sas_recipe_configuration(manifest, **kwargs)
        with self.assertRaisesRegex(ValueError, "only gemma-2-9b-it"):
            assert_sas_recipe_configuration(
                manifest, **{**kwargs, "model": "Llama-3.1-8B-Instruct"}
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            assert_sas_recipe_configuration(
                manifest, **{**kwargs, "sae_width": "65k"}
            )
        with self.assertRaisesRegex(ValueError, "not present"):
            assert_sas_recipe_configuration(
                manifest, **{**kwargs, "coefficient": -750.0}
            )

        legacy_manifest = _sas_manifest("jailbreak", "prompt_only")
        legacy_sas = legacy_manifest["grid"]["sas"]
        legacy_sas.pop("coefficients")
        legacy_sas["coefficient"] = -20.0
        assert_sas_recipe_configuration(legacy_manifest, **kwargs)

    def test_tinybench_validates_serialized_sas_recipe(self):
        manifest = _sas_manifest("jailbreak", "prompt_only")
        recipe = SAS_RECIPES["jailbreak"]
        config = {
            "layers.23": {
                "action": "sas_sparse",
                "steering_coefficient": _selected_coefficient("jailbreak"),
                "prompt_only": True,
                "release": recipe["sae_release"],
                "sae_id": "layer_23/width_16k/canonical",
                "layer": 23,
                "tau": recipe["feature_freq_threshold"],
                "use_delta_correction": True,
                "apply_post_activation": True,
            }
        }
        fake_torch = SimpleNamespace(
            load=lambda *args, **kwargs: config,
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            _validate_steer_config(
                "unused.pt", "sas", 23, -20.0, True, manifest, SAS_MODEL
            )
            config["layers.23"]["tau"] = 0.6
            with self.assertRaisesRegex(ValueError, "does not match"):
                _validate_steer_config(
                    "unused.pt", "sas", 23, -20.0, True, manifest, SAS_MODEL
                )


def _sas_fields(experiment):
    return {
        "strength_parameter": "sas_coefficient",
        "fixed_parameters": {"sas": dict(SAS_RECIPES[experiment])},
    }


def _caa_records(method, behavior="hallucination"):
    layers = SAS_LAYERS if method == "sas" else CAA_LAYERS
    strengths = SAS_COEFFICIENTS["caa"] if method == "sas" else CAA_STRENGTHS
    records = []
    for layer in layers:
        for strength in strengths:
            record = {
                "experiment": "caa",
                "split": "validation",
                "model": SAS_MODEL,
                "method": method,
                "behavior": behavior,
                "prompt_condition": "none",
                "application_mode": "all_tokens",
                "layer": layer,
                "strength": strength,
                "validation_target_probability": (
                    0.95 if layer == 23 and strength == 20.0 else 0.5
                ),
                "n_validation_examples": 50,
                "split_fingerprints": {"validation": "fp"},
            }
            if method == "sas":
                record.update(_sas_fields("caa"))
            elif method == "curveball":
                record.update(
                    {
                        "strength_parameter": "curveball_alpha",
                        "fixed_parameters": {"curveball": dict(CURVEBALL_CAA_RECIPE)},
                    }
                )
            records.append(record)
    return records


def _jail_sas_records():
    records = [
        {
            "experiment": "jailbreak",
            "split": "validation",
            "model": SAS_MODEL,
            "method": "baseline",
            "application_mode": "prompt_only",
            "layer": None,
            "strength": None,
            "validation_asr": 0.1,
            "validation_perplexity": 10.0,
            "n_harmful_validation_examples": 39,
            "n_harmless_validation_examples": 256,
            "split_fingerprints": {"all": "same"},
        }
    ]
    for layer in SAS_LAYERS:
        for coefficient in SAS_COEFFICIENTS["jailbreak"]:
            records.append(
                {
                    "experiment": "jailbreak",
                    "split": "validation",
                    "model": SAS_MODEL,
                    "method": "sas",
                    "application_mode": "prompt_only",
                    "layer": layer,
                    "strength": coefficient,
                    "validation_asr": (
                        0.8
                        if layer == 23 and coefficient in (-20.0, -40.0)
                        else 0.7
                    ),
                    "validation_perplexity": 10.2,
                    "n_harmful_validation_examples": 39,
                    "n_harmless_validation_examples": 256,
                    "split_fingerprints": {"all": "same"},
                    **_sas_fields("jailbreak"),
                }
            )
    return records


class SasSelectionSeparationTests(unittest.TestCase):
    def test_sas_only_caa_campaign_selects_on_its_complete_grid(self):
        manifest = select_caa_candidates(_caa_records("sas"))
        self.assertEqual(len(manifest["selections"]), 1)
        selection = manifest["selections"][0]
        self.assertEqual(selection["method"], "sas")
        self.assertEqual(selection["strength_parameter"], "sas_coefficient")
        self.assertEqual(selection["selected"], {"layer": 23, "strength": 20.0})
        self.assertEqual(manifest["grid"]["sas"]["coefficients"], list(SAS_COEFFICIENTS["caa"]))

        incomplete = _caa_records("sas")[1:]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_caa_candidates(incomplete)

        bad_recipe = _caa_records("sas")
        bad_recipe[0]["fixed_parameters"] = {
            "sas": {**SAS_RECIPES["caa"], "feature_freq_threshold": 0.6}
        }
        with self.assertRaisesRegex(ValueError, "recipe mismatch"):
            select_caa_candidates(bad_recipe)

    def test_other_methods_never_require_sas(self):
        records = []
        for method in ("actadd", "kas", "curveball"):
            records.extend(_caa_records(method))
        manifest = select_caa_candidates(records)
        self.assertNotIn("sas", manifest["grid"])
        self.assertEqual(
            {item["method"] for item in manifest["selections"]},
            {"actadd", "kas", "curveball"},
        )

    def test_sas_in_a_mixed_campaign_is_required_in_every_gemma_scope(self):
        records = []
        for method in ("actadd", "kas", "curveball", "sas"):
            records.extend(_caa_records(method))
        manifest = select_caa_candidates(records)
        self.assertIn("sas", manifest["grid"])

        other_behavior = []
        for method in ("actadd", "kas", "curveball"):
            other_behavior.extend(_caa_records(method, behavior="sycophancy"))
        with self.assertRaisesRegex(ValueError, "Missing=.*sas"):
            select_caa_candidates(records + other_behavior)

    def test_sas_only_jailbreak_campaign_breaks_ties_by_magnitude(self):
        manifest = select_jailbreak_candidates(_jail_sas_records())
        self.assertEqual(len(manifest["selections"]), 1)
        selection = manifest["selections"][0]
        self.assertEqual(selection["selected"], {"layer": 23, "strength": -20.0})
        self.assertEqual(selection["tie_break"][2], "lower_strength_magnitude")
        self.assertEqual(
            manifest["grid"]["sas"]["coefficients"],
            [-strength for strength in JAILBREAK_STRENGTHS],
        )


if __name__ == "__main__":
    unittest.main()

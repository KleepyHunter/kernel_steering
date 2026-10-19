import json
import random
import tempfile
import unittest
from pathlib import Path

from dataset.load_dataset import (
    load_caa_reference_validation,
    load_dataset,
    load_dataset_split,
)
from pipeline.validation_selection import (
    CAA_BEHAVIORS,
    CAA_LAYERS,
    CAA_STRENGTHS,
    CURVEBALL_CAA_RECIPE,
    JAILBREAK_LAYERS,
    JAILBREAK_STRENGTHS,
    assert_locked_configuration,
    canonical_question,
    load_manifest,
    paired_bootstrap_difference,
    paired_scores_from_predictions,
    partition_caa_generation_records,
    select_caa_candidates,
    select_jailbreak_candidates,
    write_manifest,
)


class GridTests(unittest.TestCase):
    def test_grid_sizes(self):
        self.assertEqual(len(CAA_STRENGTHS), 20)
        self.assertEqual(len(CAA_LAYERS) * len(CAA_STRENGTHS), 40)
        self.assertEqual(len(JAILBREAK_STRENGTHS), 10)
        self.assertTrue(all(len(layers) == 2 for layers in JAILBREAK_LAYERS.values()))
        self.assertEqual(CURVEBALL_CAA_RECIPE["kernel"], "poly")
        self.assertEqual(CURVEBALL_CAA_RECIPE["position"], -2)

    def test_caa_partition_is_deterministic_and_disjoint(self):
        generation = [
            {
                "question": f"Question {i}",
                "answer_matching_behavior": "(A)",
                "answer_not_matching_behavior": "(B)",
            }
            for i in range(60)
        ]
        generation.append(dict(generation[1]))
        test = [dict(generation[0]), {
            "question": "Untouched test question",
            "answer_matching_behavior": "(A)",
            "answer_not_matching_behavior": "(B)",
        }]
        reference, validation, metadata = partition_caa_generation_records(
            generation, test, behavior="hallucination", validation_size=10, seed=42
        )
        reference_reversed, validation_reversed, metadata_reversed = (
            partition_caa_generation_records(
                list(reversed(generation)),
                list(reversed(test)),
                behavior="hallucination",
                validation_size=10,
                seed=42,
            )
        )
        self.assertEqual(len(reference), 49)
        self.assertEqual(len(validation), 10)
        self.assertEqual(metadata["duplicates_removed"], 1)
        self.assertEqual(metadata["test_overlaps_removed"], 1)
        self.assertEqual(metadata["fingerprints"], metadata_reversed["fingerprints"])
        self.assertEqual(reference, reference_reversed)
        self.assertEqual(validation, validation_reversed)


class SelectionTests(unittest.TestCase):
    @staticmethod
    def _curveball_caa_fields():
        return {
            "strength_parameter": "curveball_alpha",
            "fixed_parameters": {
                "curveball": dict(CURVEBALL_CAA_RECIPE),
            },
        }

    def _caa_records(self, behavior="hallucination", method="kas", fingerprint="fp"):
        records = []
        for layer in CAA_LAYERS:
            for strength in CAA_STRENGTHS:
                metric = 0.5
                if layer == 23 and strength == 20.0:
                    metric = 0.9
                record = {
                    "experiment": "caa",
                    "split": "validation",
                    "model": "gemma-2-9b-it",
                    "method": method,
                    "behavior": behavior,
                    "prompt_condition": "none",
                    "application_mode": "all_tokens",
                    "layer": layer,
                    "strength": strength,
                    "validation_target_probability": metric,
                    "n_validation_examples": 50,
                    "split_fingerprints": {"validation": fingerprint},
                }
                if method == "curveball":
                    record.update(self._curveball_caa_fields())
                records.append(record)
        return records

    def _caa_scope_records(self, behavior="hallucination", fingerprint="fp"):
        records = []
        for method in ("actadd", "kas", "curveball"):
            records.extend(self._caa_records(behavior, method, fingerprint))
        return records

    def test_caa_selection_and_behavior_specific_fingerprints(self):
        records = self._caa_scope_records()
        records += self._caa_scope_records(
            behavior="sycophancy", fingerprint="other"
        )
        manifest = select_caa_candidates(records)
        self.assertEqual(len(manifest["selections"]), 6)
        for selection in manifest["selections"]:
            expected_strength = 20.0
            self.assertEqual(
                selection["selected"],
                {"layer": 23, "strength": expected_strength},
            )
        self.assertEqual(len(manifest["validation_candidates"]), 240)
        self.assertEqual(len(manifest["split_fingerprints"]), 2)
        self.assertEqual(
            manifest["grid"]["curveball"]["strength_parameter"],
            "curveball_alpha",
        )

    def test_caa_tie_break_and_duplicate_rejection(self):
        records = self._caa_scope_records()
        for record in records:
            record["validation_target_probability"] = 0.5
        manifest = select_caa_candidates(records)
        selections = {item["method"]: item for item in manifest["selections"]}
        self.assertEqual(
            selections["kas"]["selected"],
            {"layer": 22, "strength": 0.0},
        )
        self.assertEqual(
            selections["curveball"]["selected"],
            {"layer": 22, "strength": 0.0},
        )
        duplicate = next(item for item in records if item["method"] == "kas")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_caa_candidates(records + [dict(duplicate)])

    def test_caa_requires_complete_scope_and_recipe(self):
        records = self._caa_scope_records()
        without_curveball = [
            item for item in records if item["method"] != "curveball"
        ]
        with self.assertRaisesRegex(ValueError, "Missing=.*curveball"):
            select_caa_candidates(without_curveball)

        incomplete_curveball = [dict(item) for item in records]
        incomplete_curveball.pop(
            next(
                index
                for index, item in enumerate(incomplete_curveball)
                if item["method"] == "curveball"
            )
        )
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_caa_candidates(incomplete_curveball)

        bad_curveball_recipe = [dict(item) for item in records]
        curveball_record = next(
            item for item in bad_curveball_recipe if item["method"] == "curveball"
        )
        curveball_record["fixed_parameters"] = {
            "curveball": {**CURVEBALL_CAA_RECIPE, "degree": 2}
        }
        with self.assertRaisesRegex(ValueError, "recipe mismatch"):
            select_caa_candidates(bad_curveball_recipe)

    def _jail_records(self):
        model = "gemma-2-9b-it"
        records = [
            {
                "experiment": "jailbreak",
                "split": "validation",
                "model": model,
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
        for method in ("kas", "actadd", "curveball"):
            for layer in JAILBREAK_LAYERS[model]:
                for strength in JAILBREAK_STRENGTHS:
                    ppl = 11.0 if method == "curveball" else 10.2
                    asr = strength / 1000.0
                    if method == "kas" and layer == 23 and strength in (20.0, 40.0):
                        asr = 0.9
                        ppl = 10.4 if strength == 20.0 else 10.3
                    records.append(
                        {
                            "experiment": "jailbreak",
                            "split": "validation",
                            "model": model,
                            "method": method,
                            "application_mode": "prompt_only",
                            "layer": layer,
                            "strength": strength,
                            "validation_asr": asr,
                            "validation_perplexity": ppl,
                            "n_harmful_validation_examples": 39,
                            "n_harmless_validation_examples": 256,
                            "split_fingerprints": {"all": "same"},
                        }
                    )
        return records

    def test_jailbreak_utility_cap_and_ties(self):
        manifest = select_jailbreak_candidates(self._jail_records())
        selections = {item["method"]: item for item in manifest["selections"]}
        self.assertEqual(selections["kas"]["selected"], {"layer": 23, "strength": 40.0})
        self.assertIsNone(selections["curveball"]["selected"])
        self.assertEqual(selections["curveball"]["reason"], "no_admissible_configuration")
        self.assertEqual(len(manifest["validation_candidates"]), 61)



class ManifestAndBootstrapTests(unittest.TestCase):
    def _manifest(self):
        return {
            "schema_version": 1,
            "experiment": "caa",
            "selections": [
                {
                    "model": "gemma-2-9b-it",
                    "method": "kas",
                    "behavior": "hallucination",
                    "prompt_condition": "none",
                    "application_mode": "all_tokens",
                    "selected": {"layer": 23, "strength": 20.0},
                }
            ],
        }

    def test_manifest_digest_and_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(self._manifest(), path)
            loaded = load_manifest(path)
            assert_locked_configuration(
                loaded,
                experiment="caa",
                model="gemma-2-9b-it",
                method="kas",
                mode="all_tokens",
                layer=23,
                strength=20.0,
                behavior="hallucination",
                prompt_condition="none",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                assert_locked_configuration(
                    loaded,
                    experiment="caa",
                    model="gemma-2-9b-it",
                    method="kas",
                    mode="all_tokens",
                    layer=22,
                    strength=20.0,
                    behavior="hallucination",
                    prompt_condition="none",
                )
            payload = json.loads(path.read_text())
            payload["experiment"] = "jailbreak"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_manifest(path)

    def test_paired_bootstrap_and_id_pairing(self):
        method, comparison = paired_scores_from_predictions(
            [{"prompt": "b", "score": 2}, {"prompt": "a", "score": 1}],
            [{"prompt": "a", "score": 0}, {"prompt": "b", "score": 1}],
            metric_key="score",
        )
        result = paired_bootstrap_difference(
            method, comparison, n_resamples=100, seed=42
        )
        self.assertEqual(result["mean_difference"], 1.0)
        self.assertEqual(result["ci_lower"], 1.0)
        self.assertEqual(result["ci_upper"], 1.0)


class RepositorySplitTests(unittest.TestCase):
    def test_real_jailbreak_splits_are_disjoint(self):
        rng = random.Random(42)
        harmful_reference = rng.sample(
            load_dataset_split("harmful", "train", instructions_only=True), 128
        )
        harmless_reference = rng.sample(
            load_dataset_split("harmless", "train", instructions_only=True), 128
        )
        harmful_validation = load_dataset_split("harmful", "val")
        harmless_validation = load_dataset_split("harmless", "val")
        jailbreak_test = load_dataset("jailbreakbench")
        self.assertEqual(len(harmful_validation), 39)
        self.assertEqual(len(jailbreak_test), 100)
        harmful_sets = [
            {canonical_question(item) for item in harmful_reference},
            {canonical_question(item["instruction"]) for item in harmful_validation},
            {canonical_question(item["instruction"]) for item in jailbreak_test},
        ]
        self.assertFalse(harmful_sets[0] & harmful_sets[1])
        self.assertFalse(harmful_sets[0] & harmful_sets[2])
        self.assertFalse(harmful_sets[1] & harmful_sets[2])
        self.assertFalse(
            {canonical_question(item) for item in harmless_reference}
            & {canonical_question(item["instruction"]) for item in harmless_validation}
        )

    def test_real_caa_splits_have_expected_sizes_and_hashes(self):
        for behavior in CAA_BEHAVIORS:
            reference, validation, metadata = load_caa_reference_validation(
                behavior=behavior, validation_size=50, seed=42
            )
            self.assertEqual(len(validation), 50)
            self.assertEqual(metadata["n_test"], 50)
            self.assertEqual(len(reference), metadata["n_reference"])
            self.assertEqual(
                metadata["n_reference"] + metadata["n_validation"],
                metadata["n_generation_input"]
                - metadata["duplicates_removed"]
                - metadata["test_overlaps_removed"],
            )
            self.assertEqual(set(metadata["fingerprints"]), {"reference", "validation", "test"})


if __name__ == "__main__":
    unittest.main()

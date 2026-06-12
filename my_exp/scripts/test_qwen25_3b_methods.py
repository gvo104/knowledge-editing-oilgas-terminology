import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from easyeditor import (
    BaseEditor,
    FTHyperParams,
    KNHyperParams,
    LoRAHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
)
from easyeditor.editors.utils import summary_metrics


METHODS = {
    "FT": (FTHyperParams, os.path.join(ROOT, "my_exp", "hparams", "ft_qwen25_3b_smoke.yaml")),
    "KN": (KNHyperParams, os.path.join(ROOT, "my_exp", "hparams", "kn_qwen25_3b_smoke.yaml")),
    "LoRA": (LoRAHyperParams, os.path.join(ROOT, "my_exp", "hparams", "lora_qwen25_3b_smoke.yaml")),
    "MEMIT": (MEMITHyperParams, os.path.join(ROOT, "my_exp", "hparams", "memit_qwen25_3b_smoke.yaml")),
    "ROME": (ROMEHyperParams, os.path.join(ROOT, "my_exp", "hparams", "rome_qwen25_3b_smoke.yaml")),
}


def build_case(name: str):
    cases = {
        "france": {
            "prompt": "The capital of France is",
            "ground_truth": "Paris",
            "target_new": "Lyon",
            "subject": "France",
        },
        "eiffel": {
            "prompt": "The Eiffel Tower is located in",
            "ground_truth": "Paris",
            "target_new": "Lyon",
            "subject": "Eiffel Tower",
        },
    }
    if name not in cases:
        raise ValueError(f"Unknown case: {name}")
    return cases[name]


def run_method(method_name: str, case_name: str, output_dir: str):
    hparams_cls, hparams_path = METHODS[method_name]
    case = build_case(case_name)
    hparams = hparams_cls.from_hparams(hparams_path)
    editor = BaseEditor.from_hparams(hparams)

    metrics, _, _ = editor.edit(
        prompts=case["prompt"],
        ground_truth=case["ground_truth"],
        target_new=case["target_new"],
        subject=case["subject"],
        keep_original_weight=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{method_name.lower()}_{case_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved {method_name} metrics to {output_path}")
    summary_metrics(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["FT", "LoRA", "ROME", "MEMIT"],
        choices=sorted(METHODS.keys()),
    )
    parser.add_argument(
        "--case",
        default="france",
        choices=["france", "eiffel"],
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "my_exp", "results", "qwen25-3b"),
    )
    args = parser.parse_args()

    for method_name in args.methods:
        print(f"\n=== Running {method_name} on case={args.case} ===")
        run_method(method_name, args.case, args.output_dir)


if __name__ == "__main__":
    main()



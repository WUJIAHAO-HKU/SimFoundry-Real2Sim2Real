#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in one-frame OpenAI preflight. Reports and generated images stay on EBCloud."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--allow-paid-openai-test", action="store_true")
    parser.add_argument("--image-edit", action="store_true")
    parser.add_argument("--image-edit-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("/root/data/SOA/runs/openai_preflight"))
    args = parser.parse_args()
    if not args.allow_paid_openai_test:
        parser.error("API calls are billable; specify --allow-paid-openai-test to run")
    if not args.frame.is_file():
        parser.error("The input frame does not exist")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.pop("TEST_MODE", None)
    os.environ.pop("CACHE_MODE", None)
    from PIL import Image
    from omegaconf import OmegaConf
    from simfoundry import REPO_DIR
    from simfoundry.models.vlm import create_vlm

    cfg = OmegaConf.load(Path(REPO_DIR) / "scripts/cfg/real2sim_cfg.yaml")
    cfg.vlm_backend = "openai"
    cfg.openai.max_output_tokens = 2048
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = args.output_root / stamp
    folder.mkdir(parents=True)
    with Image.open(args.frame) as image:
        image = image.convert("RGB")
        image.thumbnail((512, 512))
        prepared = folder / "input.jpg"
        image.save(prepared, quality=85)
    report = {"started_at": stamp, "frame": str(args.frame), "checks": [], "success": False}

    def check(name, model, operation):
        start = time.monotonic()
        record = {"check": name, "model": model.model}
        try:
            record.update(operation(), success=True, http_status=200)
        except Exception as exc:
            # The adapter's RemoteCallFailed message contains status/code only.
            from simfoundry.models.vlm import RemoteCallFailed
            record.update(success=False, error_type=type(exc).__name__)
            record["http_status"] = getattr(exc, "status_code", None)
            if isinstance(exc, RemoteCallFailed):
                record["error"] = str(exc)
        record["elapsed_seconds"] = round(time.monotonic() - start, 2)
        report["checks"].append(record)
        print(json.dumps(record), flush=True)
        return record["success"]

    try:
        model = create_vlm(cfg)

        def vision():
            result = model(prompt="Name up to five visible objects. One short sentence only.",
                           image_paths=prepared, n_retries=1)
            text = model.get_result_text(result)
            (folder / "vision_response.txt").write_text(text)
            return {"response_text": text[:500]}

        report["success"] = args.image_edit_only or check("vision", model, vision)
        if (args.image_edit or args.image_edit_only) and report["success"]:
            image_model = create_vlm(cfg, purpose="image")

            def edit():
                result = image_model(prompt="Preserve the objects and support surface; remove any visible people.",
                                     image_path=prepared, n_retries=1, n_images=1)
                images = image_model.get_result_images(result)
                if not images:
                    raise RuntimeError("The image model returned no image")
                images[0].save(folder / "image_edit.png")
                return {"size": list(images[0].size)}

            report["success"] = check("image_edit", image_model, edit)
        return 0 if report["success"] else 1
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest = folder / "manifest.json"
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Preflight manifest: {manifest}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

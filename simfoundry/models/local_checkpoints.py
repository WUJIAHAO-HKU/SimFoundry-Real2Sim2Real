# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional local checkpoint selection for native SAM3 builders."""

import os
from pathlib import Path


def sam3_checkpoint_kwargs():
    value = os.environ.get("SIMFOUNDRY_SAM3_CHECKPOINT")
    if not value:
        return {}
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"SAM3 checkpoint is missing: {path}. Run the EBCloud model downloader first."
        )
    return {"checkpoint_path": str(path.resolve()), "load_from_HF": False}

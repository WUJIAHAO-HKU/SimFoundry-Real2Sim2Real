# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise installer commands without touching a conda environment or GPU."""

import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/installation/python_packages.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")


@pytest.fixture
def fake_pip(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "pip.jsonl"
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['PIP_TEST_LOG'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(int(os.environ.get('PIP_TEST_EXIT', '0')))\n"
    )
    python.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SIMFOUNDRY_")}
    env.update(PATH=str(bin_dir) + os.pathsep + env["PATH"], PIP_TEST_LOG=str(log))
    return env, log


def run(command, env=None):
    return subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(HELPER))}; {command}"],
        env=env, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("cache", [False, True])
def test_torch_install_resolves_online_dependencies_even_with_partial_cache(fake_pip, tmp_path, cache):
    env, log = fake_pip
    wheel_dir = tmp_path / "partial wheel cache"
    if cache:
        wheel_dir.mkdir()
        env["SIMFOUNDRY_WHEEL_DIR"] = str(wheel_dir)
    result = run("install_torch_stack 12.8", env)
    assert result.returncode == 0, result.stderr
    torch, pillow = [json.loads(line) for line in log.read_text().splitlines()]
    assert "torch==2.7.0+cu128" in torch
    assert "https://download.pytorch.org/whl/cu128" in torch
    assert "pillow==11.3.0" in pillow and "https://pypi.org/simple" in pillow
    assert "--no-index" not in torch and "--no-deps" not in torch
    assert ("--find-links" in torch) == cache
    if cache:
        assert torch[torch.index("--find-links") + 1] == str(wheel_dir)


def test_failed_torch_install_stops_before_pillow(fake_pip):
    env, log = fake_pip
    env["PIP_TEST_EXIT"] = "19"
    result = run("install_torch_stack", env)
    assert result.returncode == 19
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.parametrize("setting,value,command", [
    ("SIMFOUNDRY_WHEEL_DIR", "/missing-wheel-cache", "install_torch_stack"),
    ("SIMFOUNDRY_FLASH_ATTN_WHEEL", "/missing.whl", "install_flash_attention"),
    ("UNUSED", "", "install_torch_stack 12.6"),
])
def test_invalid_configuration_fails_before_install(fake_pip, setting, value, command):
    env, log = fake_pip
    env[setting] = value
    assert run(command, env).returncode != 0
    assert not log.exists()


@pytest.mark.parametrize("cached", [False, True])
def test_flash_attention_online_and_explicit_wheel(fake_pip, tmp_path, cached):
    env, log = fake_pip
    wheel = tmp_path / "flash_attn-compatible.whl"
    if cached:
        wheel.touch()
        env["SIMFOUNDRY_FLASH_ATTN_WHEEL"] = str(wheel)
    result = run("install_flash_attention", env)
    assert result.returncode == 0, result.stderr
    args = json.loads(log.read_text())
    assert (str(wheel) if cached else "flash-attn==2.8.3") in args
    assert "--no-build-isolation" in args


def make_archive(path):
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("example/setup.py")
        payload = b"# source\n"
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))


def test_source_cache_populates_from_empty_directory_and_reuses_valid_archive(tmp_path):
    source = tmp_path / "upstream.tar.gz"
    make_archive(source)
    dest = tmp_path / "new cache" / "source.tar.gz"
    command = f"cache_source_archive {shlex.quote(source.as_uri())} {shlex.quote(str(dest))}"
    result = run(command)
    assert result.returncode == 0, result.stderr
    assert dest.read_bytes() == source.read_bytes()
    source.unlink()
    assert run(command).returncode == 0


def test_failed_download_never_promotes_truncated_archive(tmp_path):
    source = tmp_path / "broken.tar.gz"
    source.write_bytes(b"not an archive")
    dest = tmp_path / "cache" / "source.tar.gz"
    result = run(f"cache_source_archive {shlex.quote(source.as_uri())} {shlex.quote(str(dest))}")
    assert result.returncode != 0
    assert not dest.exists()
    assert not list(dest.parent.glob("*.part.*"))


def test_corrupt_existing_archive_is_replaced(tmp_path):
    source = tmp_path / "source.tar.gz"
    make_archive(source)
    dest = tmp_path / "cached.tar.gz"
    dest.write_bytes(b"previous failed download")
    result = run(f"cache_source_archive {shlex.quote(source.as_uri())} {shlex.quote(str(dest))}")
    assert result.returncode == 0, result.stderr
    assert dest.read_bytes() == source.read_bytes()

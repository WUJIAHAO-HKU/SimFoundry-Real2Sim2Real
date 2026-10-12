#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared online installation policy; sourcing this file does not install anything.
install_torch_stack() {
  local cuda_version="${1:-12.8}"
  if [[ "$cuda_version" != 12.8 ]]; then
    echo "ERROR: this pinned torch/xformers profile requires --cuda-version 12.8." >&2
    return 2
  fi
  local wheel_dir="${SIMFOUNDRY_WHEEL_DIR:-}"
  local torch_index="${SIMFOUNDRY_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  local pypi_index="${SIMFOUNDRY_PYPI_INDEX_URL:-https://pypi.org/simple}"
  local -a cache_args=()
  if [[ -n "$wheel_dir" ]]; then
    if [[ ! -d "$wheel_dir" ]]; then
      echo "ERROR: SIMFOUNDRY_WHEEL_DIR is not a directory: $wheel_dir" >&2
      return 2
    fi
    cache_args=(--find-links "$wheel_dir")
  fi
  # Resolve runtime dependencies as well: a partial wheel cache must not produce
  # a torch install missing CUDA libraries, sympy, or other transitive packages.
  python -m pip install "${cache_args[@]}" --index-url "$torch_index" \
    'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' \
    'torchaudio==2.7.0+cu128' 'xformers==0.0.30' || return $?
  python -m pip install "${cache_args[@]}" --index-url "$pypi_index" 'pillow==11.3.0'
}

install_flash_attention() {
  # A cached wheel is opt-in and must match Python, torch, CUDA and the C++ ABI.
  # Do not guess compatibility from a renamed wheel in a machine-specific folder.
  local wheel="${SIMFOUNDRY_FLASH_ATTN_WHEEL:-}"
  if [[ -n "$wheel" ]]; then
    if [[ ! -f "$wheel" ]]; then
      echo "ERROR: SIMFOUNDRY_FLASH_ATTN_WHEEL does not exist: $wheel" >&2
      return 2
    fi
    python -m pip install "$wheel" --no-build-isolation
  else
    python -m pip install --index-url \
      "${SIMFOUNDRY_PYPI_INDEX_URL:-https://pypi.org/simple}" \
      'flash-attn==2.8.3' --no-build-isolation
  fi
}

cache_source_archive() (
  set -e
  local url="${1:?source URL required}" destination="${2:?cache path required}"
  local temporary=""
  mkdir -p "$(dirname "$destination")"
  if [[ -s "$destination" ]] && tar -tzf "$destination" >/dev/null 2>&1; then
    return 0
  fi
  temporary=$(mktemp "${destination}.part.XXXXXX")
  trap 'if [[ -n "$temporary" ]]; then rm -f -- "$temporary"; fi' EXIT
  curl --fail --location --retry 3 --connect-timeout 20 --max-time 900 \
    "$url" --output "$temporary"
  # A failed or truncated download must never become a reusable cache entry.
  tar -tzf "$temporary" >/dev/null
  mv -- "$temporary" "$destination"
  temporary=""
)

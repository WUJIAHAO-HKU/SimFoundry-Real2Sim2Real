#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

git_clone_pinned() {
  local url="${1:?repository URL required}"
  local destination="${2:?destination required}"
  local revision="${3:?commit required}"
  local staging attempt fetched=0
  if [[ -e "$destination" ]]; then
    echo "ERROR: refusing to replace existing dependency: $destination" >&2
    return 1
  fi
  staging=$(mktemp -d "${destination}.fetch.XXXXXX")
  git -C "$staging" init -q
  git -C "$staging" remote add origin "$url"
  for attempt in 1 2 3; do
    if timeout 600 git -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=90 \
        -C "$staging" fetch --depth 1 origin "$revision"; then
      fetched=1
      break
    fi
    echo "Pinned source fetch failed (attempt $attempt/3): $url" >&2
  done
  if [[ "$fetched" != 1 ]]; then
    echo "ERROR: fetch incomplete; preserved for diagnosis: $staging" >&2
    return 1
  fi
  git -C "$staging" checkout --detach "$revision"
  [[ "$(git -C "$staging" rev-parse HEAD)" == "$revision" ]] || return 1
  mv -T "$staging" "$destination"
}

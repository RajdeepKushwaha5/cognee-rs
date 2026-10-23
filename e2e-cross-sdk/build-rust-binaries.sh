#!/usr/bin/env bash
# Build the Rust binaries the e2e harness image expects — cognee-cli and
# cognee-telemetry-emit, release profile — into <repo>/dist/, linked against
# the harness image's own glibc and libstdc++.
#
#   bash e2e-cross-sdk/build-rust-binaries.sh
#
# This is the supported way to produce them, locally and in CI
# (.github/workflows/http-parity.yml runs exactly this). A plain host
# `cargo build --release` links against the HOST's glibc, and the binaries
# then fail to start in the (older) harness image with "GLIBC_2.xx not found"
# whenever the host is newer — which is what broke CI when ubuntu-latest moved
# to 26.04 (glibc 2.43) while the harness runs Debian Trixie (2.41).
#
# How: build the Dockerfile's `rust-build-env` stage (HARNESS_BASE plus C/C++
# build deps; no sources, no Rust toolchain), then run `cargo build` in it with
# the repo, the cargo home and the rustup home bind-mounted at their host paths
# and as the calling uid. The target dir therefore stays on the host —
# incremental across runs, and cacheable by rust-cache in CI — while the
# compile and link use the harness's libraries. Finally both binaries are
# checked to load in the bare HARNESS_BASE image.
#
# Requirements: docker, and a standard rustup installation (CARGO_HOME /
# RUSTUP_HOME, default ~/.cargo / ~/.rustup). The toolchain runs inside a
# Debian container, so it must be the stock rustup build (glibc 2.17
# baseline); a distro- or Nix-patched toolchain whose ELF interpreter points
# elsewhere will not start there — point CARGO_HOME/RUSTUP_HOME at a
# rustup-init installation instead.
#
# Passed through to cargo when set: ORT_CACHE_DIR, CARGO_INCREMENTAL,
# CARGO_PROFILE_RELEASE_DEBUG, CARGO_TARGET_DIR, CARGO_TERM_COLOR.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="$repo/e2e-cross-sdk/Dockerfile"
image=cognee-e2e-rust-build-env

# The single source of truth for the base image is the Dockerfile's
# `ARG HARNESS_BASE=` line; the build-env and harness stages both use it.
base=$(sed -n 's/^ARG HARNESS_BASE=//p' "$dockerfile")
if [ -z "$base" ]; then
    echo "error: no 'ARG HARNESS_BASE=' line in $dockerfile" >&2
    exit 1
fi
echo "── harness base: $base"

# The stage COPYs nothing, so it needs no build context: an empty dir keeps
# the build independent of the monorepo layout the full image needs.
ctx=$(mktemp -d)
trap 'rmdir "$ctx"' EXIT
docker build --target rust-build-env -t "$image" -f "$dockerfile" "$ctx"
docker run --rm "$image" sh -c 'ldd --version | head -1; g++ --version | head -1'

cargo_home="${CARGO_HOME:-$HOME/.cargo}"
rustup_home="${RUSTUP_HOME:-$HOME/.rustup}"
target_dir="${CARGO_TARGET_DIR:-$repo/target}"
mkdir -p "$target_dir"

# One invocation for both binaries: separate `cargo build -p X` calls resolve
# features per-subgraph and build duplicate variants of shared deps (that once
# cost 44 min in CI). HOME points at a scratch dir: the host HOME is not
# mounted, and nothing the build needs lives there (cargo/rustup homes are
# passed explicitly).
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e CARGO_HOME="$cargo_home" -e RUSTUP_HOME="$rustup_home" \
    -e PATH="$cargo_home/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    -e CARGO_TARGET_DIR="$target_dir" \
    -e ORT_CACHE_DIR -e CARGO_INCREMENTAL -e CARGO_PROFILE_RELEASE_DEBUG -e CARGO_TERM_COLOR \
    -v "$cargo_home:$cargo_home" -v "$rustup_home:$rustup_home" \
    -v "$repo:$repo" -v "$target_dir:$target_dir" \
    -w "$repo" \
    "$image" \
    cargo build --release -p cognee-cli -p cognee-telemetry-emit

mkdir -p "$repo/dist"
for b in cognee-cli cognee-telemetry-emit; do
    cp "$target_dir/release/$b" "$repo/dist/$b"
    test -x "$repo/dist/$b" || { echo "error: dist/$b missing or not executable" >&2; exit 1; }
done
ls -la "$repo/dist/"

# Resolve the binaries' libraries in the bare harness base — the image they
# will actually run in, without the build env's extra packages. Three
# independent failure signals, because each alone misses cases: a non-zero
# status (docker failed, or ldd rejected the file: "not a dynamic executable",
# wrong ELF class/arch); "not found" (a missing library, or a too-new symbol
# version, which ldd reports but still exits 0); and no resolved libc (any
# output that is not a normal dynamic-link listing).
for b in cognee-cli cognee-telemetry-emit; do
    echo "── ldd $b in $base"
    rc=0
    out=$(docker run --rm -v "$repo/dist:/dist:ro" "$base" ldd "/dist/$b" 2>&1) || rc=$?
    echo "$out"
    if [ "$rc" -ne 0 ]; then
        echo "error: ldd of dist/$b in $base exited $rc (see output above)" >&2
        exit 1
    fi
    if grep -q "not found" <<<"$out"; then
        echo "error: dist/$b does not load in $base (see ldd output above)" >&2
        exit 1
    fi
    if ! grep -qE '^\s*libc\.so\.6 => /' <<<"$out"; then
        echo "error: ldd did not resolve libc.so.6 for dist/$b in $base (see output above)" >&2
        exit 1
    fi
done
# And actually run one: catches what ldd cannot (e.g. a failure at relocation
# time). set -e fails the script on any error.
docker run --rm -v "$repo/dist:/dist:ro" "$base" /dist/cognee-cli --version

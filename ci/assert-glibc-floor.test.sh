#!/usr/bin/env bash
# Self-test for ci/assert-glibc-floor.sh, run in ci.yml / community.yml lint.
#
# The script's real callers are the Linux legs of ts-prebuild, java-prebuild
# and capi-release (tags, and PRs touching their build environment) and the
# ci.yml capi-check job (the x86_64 floor on every PR). This exercises the
# script itself on every PR:
#   - against the runner's own /bin/ls, with floors chosen so the result does
#     not depend on the image (every glibc binary needs more than 2.2 and
#     less than 99);
#   - against canned objdump/readelf output (OBJDUMP/READELF stubs), for the
#     cases no system binary reliably provides: weak-only version needs, and
#     the CXXABI_/GCC_ families.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
assert="$here/assert-glibc-floor.sh"
bin=/bin/ls

expect_pass() {
    if ! out=$("$@" 2>&1); then
        printf 'FAIL: expected success: %s\n%s\n' "$*" "$out" >&2
        exit 1
    fi
    echo "ok (passed): ${*: -1}"
}
expect_fail() {
    if out=$("$@" 2>&1); then
        printf 'FAIL: expected failure: %s\n%s\n' "$*" "$out" >&2
        exit 1
    fi
    echo "ok (failed as expected): ${*: -1}"
}

# ── Real binary ──────────────────────────────────────────────────────────────
expect_pass bash "$assert" --glibc 99.0 "$bin"
expect_fail bash "$assert" --glibc 2.2 "$bin"
# Every target in the floor file resolves to a complete row (explicit floors
# after --target override the file's values, keeping this image-independent).
while read -r target; do
    expect_pass bash "$assert" --target "$target" \
        --glibc 99 --glibcxx 99 --cxxabi 99 --gcc 99 "$bin"
done < <(awk '$0 !~ /^#/ && NF { print $1 }' "$here/linux-glibc-floor.txt")
# Errors are failures, not passes.
expect_fail bash "$assert" --target no-such-target "$bin"
expect_fail bash "$assert" --glibc 99.0 /nonexistent
expect_fail bash "$assert" --glibc 99.0 "$here/linux-glibc-floor.txt"

# ── Canned tool output ───────────────────────────────────────────────────────
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/bin" "$tmp/fx"
# Stubs: `objdump -T <file>` / `readelf -V <file>` print <file>.objdump /
# <file>.readelf from the fixture dir.
printf '#!/bin/sh\ncat "$2.objdump"\n' >"$tmp/bin/objdump"
printf '#!/bin/sh\ncat "$2.readelf"\n' >"$tmp/bin/readelf"
chmod +x "$tmp/bin/objdump" "$tmp/bin/readelf"
stub() { OBJDUMP="$tmp/bin/objdump" READELF="$tmp/bin/readelf" "$@"; }

UND='0000000000000000      DF *UND*	0000000000000000'
WUND='0000000000000000  w   DF *UND*	0000000000000000'
# fixture <name> <version-needs Flags of GLIBC_2.39> : weak pidfd refs at 2.39,
# plus strong 2.38 / C++ runtime imports.
fixture() {
    : >"$tmp/fx/$1"
    cat >"$tmp/fx/$1.objdump" <<EOF

$1:     file format elf64-x86-64

DYNAMIC SYMBOL TABLE:
$UND (GLIBC_2.2.5) malloc
$UND (GLIBC_2.38) __isoc23_strtol
$WUND (GLIBC_2.39) pidfd_spawnp
$WUND (GLIBC_2.39) pidfd_getpid
$UND (GLIBCXX_3.4.32) _ZSt21ios_base_library_initv
$UND (CXXABI_1.3.15) __cxa_call_terminate
$UND (GCC_14.0.0) __extendbfsf2
0000000000001000 g    DF .text	0000000000000010  Base        exported_fn
EOF
    cat >"$tmp/fx/$1.readelf" <<EOF
Version needs section '.gnu.version_r' contains 3 entries:
 Addr: 0x0000000000000400  Offset: 0x00000400  Link: 7 (.dynstr)
  000000: Version: 1  File: libc.so.6  Cnt: 3
  0x0010:   Name: GLIBC_2.2.5  Flags: none  Version: 2
  0x0020:   Name: GLIBC_2.38  Flags: none  Version: 3
  0x0030:   Name: GLIBC_2.39  Flags: $2  Version: 4
  0x0040: Version: 1  File: libstdc++.so.6  Cnt: 2
  0x0050:   Name: GLIBCXX_3.4.32  Flags: none  Version: 5
  0x0060:   Name: CXXABI_1.3.15  Flags: none  Version: 6
  0x0070: Version: 1  File: libgcc_s.so.1  Cnt: 1
  0x0080:   Name: GCC_14.0.0  Flags: none  Version: 7
EOF
}
fixture weakver WEAK
fixture strongver none
lax=(--glibcxx 3.4.32 --cxxabi 1.3.15 --gcc 14.0.0)

# A version whose version-needs entry is WEAK is not enforced by ld.so.
expect_pass stub bash "$assert" --glibc 2.38 "${lax[@]}" "$tmp/fx/weakver"
# Weak *symbols* under a non-weak entry still are — this is the real
# pidfd_* case (Fedora 39 / glibc 2.38 refuses such a file).
expect_fail stub bash "$assert" --glibc 2.38 "${lax[@]}" "$tmp/fx/strongver"
expect_pass stub bash "$assert" --glibc 2.39 "${lax[@]}" "$tmp/fx/strongver"
# libstdc++ / libgcc_s families, each on its own.
expect_fail stub bash "$assert" --glibc 2.39 --glibcxx 3.4.31 "$tmp/fx/strongver"
expect_fail stub bash "$assert" --glibc 2.39 --cxxabi 1.3.14 "$tmp/fx/strongver"
expect_fail stub bash "$assert" --glibc 2.39 --gcc 13.0.0 "$tmp/fx/strongver"
# Exported (defined) versions never count.
expect_pass stub bash "$assert" --glibc 2.39 "${lax[@]}" "$tmp/fx/strongver"

echo "assert-glibc-floor.sh self-test passed"

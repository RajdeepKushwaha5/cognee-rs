#!/usr/bin/env bash
# Fail if an ELF binary / shared object needs a newer glibc, libstdc++ or
# libgcc_s than the floor we promise for its target.
#
#   ci/assert-glibc-floor.sh --target <rust-target> <elf>...
#       floors from ci/linux-glibc-floor.txt (the shipped-artifact promise)
#   ci/assert-glibc-floor.sh --glibc X.YY [--glibcxx V] [--cxxabi V] [--gcc V] <elf>...
#       explicit floors; a family without a floor is not checked
#
# Checked symbol-version families:
#   GLIBC_    glibc (libc.so.6, libm, libpthread, ...)
#   GLIBCXX_  libstdc++ library ABI
#   CXXABI_   libstdc++ C++ runtime ABI (exceptions, RTTI, guards)
#   GCC_      libgcc_s (unwinder, soft-float / bf16 helpers)
#
# For each family it takes the highest version among the file's *imports*:
# the versioned undefined dynamic symbols (`objdump -T`, `*UND*` rows). That
# is the oldest library the file will load against, because the dynamic
# loader refuses a file whose version-needs entry names a version the system
# library does not define ("version `GLIBC_2.43' not found"). Defined symbols
# are skipped: a file that bundles libstdc++ statically *exports* GLIBCXX_*
# versions, which say nothing about the system it needs.
#
# Weak references: the loader skips a missing version only when its
# version-needs entry carries VER_FLG_WEAK (`readelf -V` shows `Flags: WEAK`),
# so such versions are ignored here. A weak *symbol* (`w` in objdump -T) is
# NOT enough on its own — the per-version flag decides. Both GNU ld and lld
# leave that flag unset even when every reference to a version is weak (Rust
# std's weak pidfd_spawnp/pidfd_getpid yield a `Flags: none` GLIBC_2.39
# entry), and glibc 2.38 then refuses to load the file. Verified by
# dlopen()ing such an artifact on Fedora 39 (glibc 2.38).
#
# Why this exists: a native Linux build takes its floor from whatever image it
# happens to run on. On 2026-10-22 ubuntu-latest moved to 26.04 (glibc 2.43)
# and binaries built there stopped loading on older systems, with nothing in
# the build to say so. This makes the floor a failing assertion on the
# artifact itself.
#
# objdump/readelf (binutils) read foreign-arch ELF — aarch64 files on an
# x86_64 runner — without a cross toolchain. OBJDUMP / READELF override them
# (the self-test, ci/assert-glibc-floor.test.sh, uses that to feed canned
# output).
set -euo pipefail

OBJDUMP="${OBJDUMP:-objdump}"
READELF="${READELF:-readelf}"
FLOOR_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/linux-glibc-floor.txt"
FAMILIES=(GLIBC GLIBCXX CXXABI GCC)

usage() {
    echo "usage: $0 --target <rust-target> <elf>..." >&2
    echo "       $0 --glibc X.YY [--glibcxx V] [--cxxabi V] [--gcc V] <elf>..." >&2
    exit 2
}

declare -A floor=()
while [ $# -gt 0 ]; do
    case "$1" in
        --target)
            [ $# -ge 2 ] || usage
            row=$(awk -v t="$2" '$0 !~ /^#/ && $1 == t { print $2, $3, $4, $5 }' "$FLOOR_FILE")
            read -r f_glibc f_glibcxx f_cxxabi f_gcc <<<"$row"
            if [ -z "${f_gcc:-}" ]; then
                echo "::error::no complete floor row for target '$2' in $FLOOR_FILE" >&2
                exit 2
            fi
            floor[GLIBC]=$f_glibc floor[GLIBCXX]=$f_glibcxx
            floor[CXXABI]=$f_cxxabi floor[GCC]=$f_gcc
            shift 2
            ;;
        --glibc)   [ $# -ge 2 ] || usage; floor[GLIBC]=$2;   shift 2 ;;
        --glibcxx) [ $# -ge 2 ] || usage; floor[GLIBCXX]=$2; shift 2 ;;
        --cxxabi)  [ $# -ge 2 ] || usage; floor[CXXABI]=$2;  shift 2 ;;
        --gcc)     [ $# -ge 2 ] || usage; floor[GCC]=$2;     shift 2 ;;
        --) shift; break ;;
        -*) usage ;;
        *) break ;;
    esac
done
[ -n "${floor[GLIBC]:-}" ] && [ $# -ge 1 ] || usage

# Highest version on stdin (sort -V is version-aware: 2.9 < 2.17 < 2.38).
highest() { sort -V | tail -n1; }

# version_gt A B: true when A is strictly newer than B.
version_gt() { [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | highest)" = "$1" ]; }

fail=0
for f in "$@"; do
    if [ ! -f "$f" ]; then
        echo "::error::$f: no such file" >&2
        fail=1
        continue
    fi
    if ! dyn=$("$OBJDUMP" -T "$f" 2>&1); then
        echo "::error::$f: objdump cannot read its dynamic symbols — not an ELF shared object/executable? ($dyn)" >&2
        fail=1
        continue
    fi
    if ! verinfo=$("$READELF" -V "$f" 2>&1); then
        echo "::error::$f: readelf cannot read its version sections ($verinfo)" >&2
        fail=1
        continue
    fi
    # Versions whose version-needs entry is marked weak (see header).
    weak_versions=$(printf '%s\n' "$verinfo" |
        awk '/Name: / && /Flags: .*WEAK/ { for (i = 1; i <= NF; i++) if ($i == "Name:") print $(i + 1) }')

    # "<VERSION> <symbol>" for every versioned import. The last two columns of
    # an *UND* row are e.g. `(GLIBC_2.38) __isoc23_strtol` (parenthesised when
    # the version is hidden); unversioned rows have a hex size there instead.
    imports=$(printf '%s\n' "$dyn" | awk '/\*UND\*/ { print $(NF-1), $NF }' | tr -d '()' |
        { grep -E '^[A-Z]+_[0-9]' || true; })
    if [ -n "$weak_versions" ]; then
        imports=$(awk 'NR == FNR { weak[$1] = 1; next } !($1 in weak)' \
            <(printf '%s\n' "$weak_versions") <(printf '%s\n' "$imports"))
        echo "note: $f: ignoring weak-only version needs: $(echo $weak_versions)"
    fi

    if ! grep -q '^GLIBC_' <<<"$imports"; then
        # Every dynamically-linked glibc object imports at least one versioned
        # glibc symbol; none means objdump read nothing useful (a static
        # binary, the wrong file, or an objdump that cannot parse it).
        echo "::error::$f: no GLIBC_* imports found — is this a dynamically-linked glibc ELF?" >&2
        fail=1
        continue
    fi

    for fam in "${FAMILIES[@]}"; do
        max="${floor[$fam]:-}"
        [ -n "$max" ] || continue
        rows=$({ grep -E "^${fam}_[0-9]" <<<"$imports" || true; } | sed "s/^${fam}_//")
        if [ -z "$rows" ]; then
            echo "ok: $f imports no ${fam}_* symbols"
            continue
        fi
        need=$(cut -d' ' -f1 <<<"$rows" | highest)
        if version_gt "$need" "$max"; then
            echo "::error::$f needs ${fam}_$need, newer than the promised floor ${fam}_$max. Imports above the floor:" >&2
            sort -uV <<<"$rows" | while read -r v sym; do
                if version_gt "$v" "$max"; then echo "    ${fam}_$v $sym" >&2; fi
            done
            fail=1
        else
            echo "ok: $f needs ${fam}_$need (floor $max)"
        fi
    done
done
exit "$fail"

# x86_64 glibc build image for the shipped Linux prebuilds (cross-rs).
#
# The x86_64 legs of ts-prebuild.yml, java-prebuild.yml and capi-release.yml
# used to build natively on the GitHub runner, so the glibc and libstdc++ the
# artifacts require were whatever that image shipped — and moved silently when
# the image did (ubuntu-latest -> 26.04 on 2026-10-22 is glibc 2.43). Building
# through `cross` in this image pins them instead. ci/linux-glibc-floor.txt
# records the promised floor and ci/assert-glibc-floor.sh enforces it on every
# artifact.
#
# Why ubuntu:24.04 (glibc 2.39, GCC 13) and not something older: ort-sys's
# prebuilt x86_64 libonnxruntime.a imports __isoc23_strtol & co., which need
# glibc >= 2.38. An older sysroot (manylinux_2_28, Ubuntu 22.04) cannot
# satisfy those references — an executable fails to link, and a cdylib links
# with them unresolved and fails at load time — so the x86_64 floor cannot go
# below 2.38 without building ONNX Runtime from source. 24.04 is the oldest
# supported LTS at or above 2.38. GCC 13 also covers lbug's C++20 needs.
#
# No CMake toolchain file and no CC_/CXX_ overrides: host and target are both
# x86_64, so cc-rs and cmake use the image's native gcc/g++ — the point is the
# sysroot, not a cross compiler.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
# build-essential: gcc/g++ 13 + libc6-dev (the glibc 2.39 headers/sysroot).
# cmake: lbug's bundled C++ tree. protobuf-compiler + libprotobuf-dev: prost
# codegen (lance-encoding's protos import google/protobuf/*.proto, which ship
# in libprotobuf-dev — only a Recommends of protobuf-compiler).
# pkg-config + libssl-dev: ort-sys's download path (ureq/native-tls ->
# openssl-sys) builds against system OpenSSL, as on the runner before.
RUN apt-get update && \
    apt-get install --assume-yes --no-install-recommends \
        build-essential cmake pkg-config libssl-dev protobuf-compiler libprotobuf-dev \
        ca-certificates curl git python3 && \
    g++ --version | head -1 && \
    ldd --version | head -1 && \
    protoc --version && \
    rm -rf /var/lib/apt/lists/*

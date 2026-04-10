#!/usr/bin/env bash

# Clean after installation to minimize image size.
set -e
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $@
rm -rf /var/lib/apt/lists/*

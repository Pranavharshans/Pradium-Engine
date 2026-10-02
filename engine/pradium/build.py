#!/usr/bin/env python3
"""Materialize Pradium's patch series without modifying the upstream submodule."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

PIN = 'd3739fd393337b1ff4d6c2a342b12f0c87a9592f'
HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / 'exllamav3'


def materialize(destination: Path) -> dict:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f'Destination must not exist: {destination}')
    if destination == UPSTREAM or UPSTREAM in destination.parents:
        raise ValueError('Destination must be outside the upstream submodule')
    # Export the immutable pin; do not copy any uncommitted files from the checkout.
    archive = subprocess.run(
        ['git', '-C', str(UPSTREAM), 'archive', '--format=tar', PIN],
        check=True, capture_output=True,
    ).stdout
    patches = sorted((HERE / 'patches').glob('*.patch'))
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(destination, filter='data')
    # Keep git apply outside the parent repository's path filtering. Otherwise
    # git can silently skip patches whose destination is outside its worktree.
    environment = os.environ.copy()
    environment['GIT_CEILING_DIRECTORIES'] = str(destination.parent)
    for patch in patches:
        command = ['git', 'apply']
        subprocess.run(command + ['--check', str(patch)], cwd=destination,
                       env=environment, check=True)
        subprocess.run(command + [str(patch)], cwd=destination,
                       env=environment, check=True)
    manifest = {
        'upstream': 'https://github.com/turboderp-org/exllamav3',
        'upstream_commit': PIN,
        'patches': [{'name': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in patches],
        'gpu_validated': False,
    }
    (destination / 'PRADIUM_BUILD.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output), indent=2))


if __name__ == '__main__':
    main()

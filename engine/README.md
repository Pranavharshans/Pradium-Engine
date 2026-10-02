# Inference engines

This directory holds upstream inference engines used by Pradium tooling.

## ExLlamaV3

`exllamav3/` is a Git submodule pointing to the official
[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) repository.
Pradium Engine remains an independent repository; upstream source and history
are maintained by ExLlamaV3's authors. Its license remains in
`exllamav3/LICENSE`.

Initial pin: **v1.5.3**, commit
`d3739fd393337b1ff4d6c2a342b12f0c87a9592f`, the official default branch
(`master`) HEAD verified when added. The Git submodule entry is the authoritative
revision; updates must explicitly change that pin.

Clone Pradium with the engine:

```bash
git clone --recurse-submodules https://github.com/Pranavharshans/Pradium-Engine.git
```

Populate the engine in an existing checkout:

```bash
git submodule update --init --recursive
```

Adding the source does not install dependencies or validate GPU inference.

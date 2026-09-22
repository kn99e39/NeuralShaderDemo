# Worklog — Checkpoint Bundle Transport Policy (2026-09-22)

## What changed

Added `experiments/dynamic_transport_validity/tools/checkpoint_bundle.py` for
lossless checkpoint transport in Git-friendly chunks. `pack` writes 50 MiB
chunks by default plus a SHA-256 manifest; `verify` validates every part; and
`unpack` reconstructs the original checkpoint atomically only after full-hash
verification.

The root ignore policy now excludes original `.ckpt`, `.h5`, and `.exr` files.
Chunk parts and their manifest can be deliberately placed in a small curated
artifact directory and committed when that is justified.

## Why

The prior push failure came from cumulative generated artifacts in Git history,
not an individual checkpoint above GitHub's file limit. Splitting a live
checkpoint does not make it directly runnable; it must be reassembled first.
The bundle format is therefore archival transport only, while normal RNA
execution continues to consume the original `.ckpt` file.

## Validation and limits

No production checkpoint was present after the clean reset, so none was packed
or committed. A temporary 3 MiB binary fixture was packed into three 1 MiB
parts, verified, unpacked, and byte-compared successfully.

For a future checkpoint, use:

```text
python experiments/dynamic_transport_validity/tools/checkpoint_bundle.py pack \
  --source <best-or-last.ckpt> --output-dir artifacts/checkpoints/<run>
python experiments/dynamic_transport_validity/tools/checkpoint_bundle.py verify \
  --manifest artifacts/checkpoints/<run>/<name>.ckpt.manifest.json
python experiments/dynamic_transport_validity/tools/checkpoint_bundle.py unpack \
  --manifest artifacts/checkpoints/<run>/<name>.ckpt.manifest.json \
  --output-dir <runtime-checkpoint-directory>
```

Only a curated best checkpoint and, where needed, `last.ckpt` should be bundled.
Chunking avoids the individual-file ceiling but does not remove GitHub repository
size, transfer, or LFS quota constraints; it is not suitable for bulk epoch
archives or training datasets.

## Open item

When a new checkpoint is produced, bundle it into an explicit artifact directory,
verify/reassemble it once, then commit only that small selected bundle if its
total repository cost remains acceptable.

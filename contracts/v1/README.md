# V1 compatibility contract

`contract.json` records the public behaviour that V2 must preserve during the
migration window. Required JSON paths are a minimum compatible subset: writers
may add fields, and historical files may contain variable-length arrays.

This contract deliberately separates interface compatibility from numerical
correctness. Scientific behaviour is guarded by `benchmarks/v1-manual-gt.json`.
The baseline includes the known `Ff4-15_f24r` failure so that a rewrite cannot
hide it by changing metrics or selecting a different output field.

The contract is not a claim that V1's multi-file result format is the desired
V2 design. V2 should introduce a versioned aggregate result and run manifest,
while exporting these V1 artifacts until the compatibility window closes.

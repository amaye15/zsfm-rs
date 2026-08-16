#!/usr/bin/env python3
"""Export weights zsfm can't read natively into a .safetensors file it can.

Covers TensorFlow SavedModel directories and TF checkpoints (TensorBundle
format, no pure-Rust reader exists). Keras .h5/.keras files do NOT need this —
`zsfm convert` reads them directly.

Usage:
    python export-weights.py <input> <output.safetensors>

  <input> is one of:
    - a SavedModel directory (contains saved_model.pb)
    - a TF checkpoint prefix (path/to/ckpt for ckpt.index + ckpt.data-*)

Requires: tensorflow, numpy, safetensors  (pip install tensorflow safetensors)

Then convert as usual:
    zsfm convert <output.safetensors> -o model.gguf --dtype f16
"""

import os
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]

    import numpy as np
    from safetensors.numpy import save_file

    import tensorflow as tf  # noqa: deferred so --help works without TF

    if os.path.isdir(src):
        if not os.path.exists(os.path.join(src, "saved_model.pb")):
            print(f"error: {src} is a directory but not a SavedModel "
                  "(no saved_model.pb)", file=sys.stderr)
            return 1
        variables_prefix = os.path.join(src, "variables", "variables")
        ckpt = tf.train.latest_checkpoint(os.path.join(src, "variables")) or variables_prefix
    else:
        ckpt = src

    reader = tf.train.load_checkpoint(ckpt)
    shape_map = reader.get_variable_to_shape_map()

    tensors, skipped = {}, []
    for name in sorted(shape_map):
        arr = np.asarray(reader.get_tensor(name))
        if arr.dtype.kind not in "fiub":  # float/int/uint/bool only
            skipped.append(f"{name} (dtype {arr.dtype})")
            continue
        if arr.dtype.kind in "iub":
            arr = arr.astype(np.float32)
        # safetensors keys must be non-empty; TF names like "a/b/.ATTRIBUTES/x"
        # are kept verbatim — rename downstream with zsfm's --strip-prefix.
        tensors[name] = arr

    if skipped:
        print(f"skipped {len(skipped)} non-numeric variable(s):", file=sys.stderr)
        for s in skipped:
            print(f"  {s}", file=sys.stderr)
    if not tensors:
        print("error: checkpoint contains no numeric tensors", file=sys.stderr)
        return 1

    save_file(tensors, dst)
    print(f"wrote {len(tensors)} tensors to {dst}")
    print(f"next: zsfm convert {dst} -o model.gguf")
    return 0


if __name__ == "__main__":
    sys.exit(main())

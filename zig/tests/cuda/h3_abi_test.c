// Compile the public C layout alongside the exported Zig library on host and Linux.
#include <stddef.h>
#include "../../src/families/h3/cuda_lib.h"
_Static_assert(sizeof(tf_h3_inputs) == 96, "H3 ABI 1 size");
_Static_assert(offsetof(tf_h3_inputs, attention_out) == 88, "H3 ABI 1 attention output");

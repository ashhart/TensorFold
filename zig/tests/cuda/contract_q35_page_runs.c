// Compile and execute the exact header that the two CUDA cache formats include.
#include "qwen35_page_run.h"
#include <stddef.h>
_Static_assert(sizeof(TfQ35PageRun) == 32, "page-run ABI size");
_Static_assert(offsetof(TfQ35PageRun, stride) == 16, "page-run ABI stride");
uint64_t tf_test_q35_page_base(TfQ35PageRun* run, int pos) {
    return tf_q35_page_base(run, pos);
}

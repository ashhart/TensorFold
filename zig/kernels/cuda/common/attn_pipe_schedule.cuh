// Two half slots: each consumer's entrance also fences the previous consumer before reusing its slot.
#pragma once
#ifndef TF_ATTN_PIPE_FN
#define TF_ATTN_PIPE_FN __device__ __forceinline__
#endif
namespace tf_attn_pipe {
template <class Wait, class Barrier, class Stage>
TF_ATTN_PIPE_FN void advance(int item, int items, Wait wait, Barrier barrier, Stage stage) {
    wait(item == 0);
    barrier();
    if (item > 0 && item + 1 < items) stage(item + 1);
}
}  // namespace tf_attn_pipe

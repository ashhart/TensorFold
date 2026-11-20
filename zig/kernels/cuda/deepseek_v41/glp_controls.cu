// Planted GLPv3 additive faults, built into their own image that no serve loads: a tail column left unedited, an
// FP16 sum, nonfinite inputs flushed to zero, a tap that sums instead of averaging, a writer that adds its ranks
// last to first, and truncation in place of round to nearest even. Each must go red at its declared gate while the
// production entry points of glp.cu stay green on the same inputs; the contract and credits are in glp.cuh.

#include "glp.cuh"

using namespace dsv41_glp;

#define GLP_STREAM(NAME, CTL)                                                                                      \
    extern "C" __global__ void __launch_bounds__(NT) dsv41_glp_add_stream_##NAME(const StreamArgs a) {             \
        stream_body<CTL>(a);                                                                                       \
    }
#define GLP_WRITER(NAME, CTL)                                                                                      \
    extern "C" __global__ void __launch_bounds__(NT) dsv41_glp_add_writer_##NAME(const WriterArgs a) {             \
        writer_body<CTL>(a);                                                                                       \
    }

GLP_STREAM(sum, SUM)                    // the tap only: its mean without the 1/4 scale
GLP_STREAM(half, HALF)
GLP_WRITER(half, HALF)
GLP_STREAM(drop_tail, DROP_TAIL)
GLP_WRITER(drop_tail, DROP_TAIL)
GLP_STREAM(drop_nonfinite, DROP_NONFINITE)
GLP_WRITER(drop_nonfinite, DROP_NONFINITE)
GLP_WRITER(rank_order, RANK_ORDER)      // ranks only: the stream has none to reorder
GLP_STREAM(truncate, TRUNCATE)
GLP_WRITER(truncate, TRUNCATE)

// GLPv3 additive fold/writer/final entry points, loaded only for active steering; contract and credits are in glp.cuh.

#include "glp.cuh"

using namespace dsv41_glp;

extern "C" {

__global__ void __launch_bounds__(NT) dsv41_glp_add_stream(const StreamArgs a) { stream_body<NONE>(a); }

__global__ void __launch_bounds__(NT) dsv41_glp_add_writer(const WriterArgs a) { writer_body<NONE>(a); }

__global__ void __launch_bounds__(NT) dsv41_glp_add_final(const FinalArgs a) { final_body(a); }

}

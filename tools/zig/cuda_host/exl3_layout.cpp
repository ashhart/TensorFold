#include "../../../zig/kernels/cuda/exl3_decode.cuh"

// Invert the documented mma B-fragment coordinates independently, covering every lane and value.
constexpr bool layout() {
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
            const int lane = 4 * (col % 8) + (row % 8) / 2;
            const int value = 4 * (col / 8) + 2 * (row / 8) + row % 2;
            if (tf_exl3::value_row(lane, value) != row || tf_exl3::value_col(lane, value) != col) return false;
        }
    }
    return true;
}
static_assert(layout(), "EXL3 unpack coordinates must bijectively cover the mma 16x16 tile");

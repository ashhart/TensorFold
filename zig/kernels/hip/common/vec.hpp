#pragma once

// The 128-bit vector the tiles load code words and activation rows with.

namespace tf {
namespace rocm {

using u32x4 = unsigned __attribute__((ext_vector_type(4)));

}  // namespace rocm
}  // namespace tf

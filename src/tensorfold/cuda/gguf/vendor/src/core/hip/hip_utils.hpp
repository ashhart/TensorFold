#ifndef GUFO_CORE_HIP_HIP_UTILS_HPP_
#define GUFO_CORE_HIP_HIP_UTILS_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace gufo::hip {
// Cleanup must not throw while another GPU failure is unwinding the request.
inline void LogCleanupError(hipError_t error) noexcept {
  if (error != hipSuccess)
    std::fprintf(stderr, "HIP cleanup failed: %s\n", hipGetErrorString(error));
}
inline void LogCleanupError(hipblasStatus_t error) noexcept {
  if (error != HIPBLAS_STATUS_SUCCESS)
    std::fprintf(stderr, "hipBLAS cleanup failed: %d\n",
                 static_cast<int>(error));
}
}  // namespace gufo::hip

#define HIP_CHECK(call)                                                 \
  do {                                                                  \
    hipError_t err = (call);                                            \
    if (err != hipSuccess) {                                            \
      std::string msg = std::string("HIP error in ") + __FILE__ + ":" + \
                        std::to_string(__LINE__) +                      \
                        " (" #call "): " + hipGetErrorString(err);      \
      throw std::runtime_error(msg);                                    \
    }                                                                   \
  } while (0)

#define HIPBLAS_CHECK(call)                                                 \
  do {                                                                      \
    hipblasStatus_t status = (call);                                        \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                 \
      std::string msg = std::string("hipBLAS error in ") + __FILE__ + ":" + \
                        std::to_string(__LINE__) +                          \
                        " (" #call "): " + std::to_string(status);          \
      throw std::runtime_error(msg);                                        \
    }                                                                       \
  } while (0)

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_CORE_HIP_HIP_UTILS_HPP_

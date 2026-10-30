// cp.async groups: commit the copies issued so far as one group, wait until at most N groups are still in flight.
#pragma once

namespace tf_common {

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

} // namespace tf_common

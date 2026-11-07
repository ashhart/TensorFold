// Gate 5's planted XPU controls: a wide group, SLM over the work-group's, 256-GRF work-items, and sub-group 32.
#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(256, 1, 1)))
kernel void tf_ctl_big_slm(global int *out) {
  local uchar big[160 * 1024];
  big[get_local_id(0)] = 1;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_local_id(0) == 0) out[0] = big[0];
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(1024, 1, 1)))
kernel void tf_ctl_large_grf(global int *out) {
  if (get_local_id(0) == 0) out[0] = 1;
}

KERNEL(256) void tf_ctl_wide_group(global int *out) {
  if (get_local_id(0) == 0) out[0] = 1;
}

__attribute__((intel_reqd_sub_group_size(32))) __attribute__((reqd_work_group_size(256, 1, 1)))
kernel void tf_ctl_subgroup32(global int *out) {
  if (get_local_id(0) == 0) out[0] = 1;
}

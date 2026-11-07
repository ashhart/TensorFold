// Sub-group 8: ocloc refuses it for bmg, so the records check its declaration.
__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(256, 1, 1)))
kernel void tf_ctl_subgroup8(global int *out) {
  if (get_local_id(0) == 0) out[0] = 1;
}

#include "nemotron_mtp_policy.h"
extern "C" unsigned tf_mtp_host_step(const TfMtpPolicy* p, double* run, unsigned j, unsigned* n, float probability) {
    return tf_mtp_advance(*p, *run, j, *n, probability);
}
extern "C" double tf_mtp_host_square(double run) { return tf_mtp_square(run); }
extern "C" unsigned tf_mtp_policy_bytes() { return sizeof(TfMtpPolicy); }
extern "C" unsigned tf_mtp_state_bytes() { return sizeof(TfMtpState); }

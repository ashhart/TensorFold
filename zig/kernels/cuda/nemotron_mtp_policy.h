// The post-level DepthRule decision for the native CUDA MTP graph and its host differential test.
#pragma once
#include <math.h>
#include <stdint.h>
#ifdef __CUDACC__
#define TF_MTP_HD __host__ __device__
#else
#define TF_MTP_HD
#endif
struct TfMtpPolicy {
    double row[33];
    double rate, level, floor;
    uint32_t most, has_floor;
};
// Match std.math.pow(f64, run, 2): normalize before squaring, including subnormal results.
TF_MTP_HD inline double tf_mtp_square(double run) {
    if (run == 0.0) return 0.0;
    if (run == 1.0 || isnan(run)) return run;
    int exponent = 0;
    const double x = frexp(run, &exponent);
    double y = x * x;
    exponent *= 2;
    if (y < 0.5) { y += y; --exponent; }
    return scalbn(y, exponent);
}
TF_MTP_HD inline bool tf_mtp_advance(const TfMtpPolicy& p, double& run, uint32_t j, uint32_t& n, float probability) {
    run *= static_cast<double>(probability);
    const double score = tf_mtp_square(run);
    const bool keep = j == 1 || (p.has_floor ? run >= p.floor : score >= p.rate * p.row[j]);
    if (!keep) return false;
    n = j;
    return j < p.most && (p.has_floor ? run >= p.floor : score >= p.rate * (p.row[j + 1] + p.level));
}
struct TfMtpState {
    TfMtpPolicy policy;
    double run;
    uint32_t j, n, levels, pad;
    uint32_t meta[2], token;
    float probability;
    uint32_t ids[31];
    float probs[31];
};

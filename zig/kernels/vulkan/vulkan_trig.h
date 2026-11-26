// Clean-room float32 sin/cos helpers for the Vulkan kernels.
// One reduction path: k = round(x * 2/pi), r = x - k * (pi/2) via a three-part
// Cody-Waite pi/2, then minimax polynomials on |r| <= pi/4. Accurate to a few
// ULPs for |x| up to ~8192 on every driver.
#ifndef VULKAN_TRIG_H
#define VULKAN_TRIG_H

float vulkan_trig_reduce(float x, out int q) {
    const float inv_pio2 = 0.63661977236758134308; // 2/pi
    float k = floor(x * inv_pio2 + 0.5);
    q = int(k);
    float r = x - k * 1.57079625129699707031e+00; // pi/2 head
    r = r - k * 7.54978941586159635335e-08;       // pi/2 mid
    r = r - k * 5.39030295341723897719e-15;       // pi/2 tail
    return r;
}

float vulkan_trig_sin_r(float r) {
    float r2 = r * r;
    return r + r * r2 * (-1.66666666666666324348e-01
        + r2 * (8.33333333332248946124e-03
        + r2 * (-1.98412698298579493134e-04
        + r2 * (2.75573137070700676789e-06
        + r2 * -2.50507602534068634195e-08))));
}

float vulkan_trig_cos_r(float r) {
    float r2 = r * r;
    return 1.0 - 0.5 * r2
        + r2 * r2 * (4.16666666666666019037e-02
        + r2 * (-1.38888888888741095749e-03
        + r2 * (2.48015872894767294178e-05
        + r2 * (-2.75573143513906633035e-07
        + r2 * 2.08757232129817482790e-09))));
}

float vulkan_trig_sin(float x) {
    int q;
    float r = vulkan_trig_reduce(x, q);
    float s = vulkan_trig_sin_r(r);
    float c = vulkan_trig_cos_r(r);
    switch (q & 3) {
        case 0:  return s;
        case 1:  return c;
        case 2:  return -s;
        default: return -c;
    }
}

float vulkan_trig_cos(float x) {
    int q;
    float r = vulkan_trig_reduce(x, q);
    float s = vulkan_trig_sin_r(r);
    float c = vulkan_trig_cos_r(r);
    switch (q & 3) {
        case 0:  return c;
        case 1:  return -s;
        case 2:  return -c;
        default: return s;
    }
}

void vulkan_trig_cos_sin(float x, out float c, out float s) {
    int q;
    float r = vulkan_trig_reduce(x, q);
    float sr = vulkan_trig_sin_r(r);
    float cr = vulkan_trig_cos_r(r);
    switch (q & 3) {
        case 0:  s = sr;  c = cr;  break;
        case 1:  s = cr;  c = -sr; break;
        case 2:  s = -sr; c = -cr; break;
        default: s = -cr; c = sr;  break;
    }
}

#endif // VULKAN_TRIG_H

// Planes buffer: three FP16 planes of n values, n FP32 restore values, then eight flag words.
inline ulong sam_planes_floats(ulong n) { return (3 * n + 1) / 2; }

// Three FP16 planes keep every FP32 bit at or above 2^-46, so only values below 2^-22 can round.
inline bool sam_planes_split(float value, thread half& high, thread half& middle, thread half& low) {
    high = half(value);
    const float remainder = value - float(high);
    middle = half(ldexp(remainder, 11));
    low = half(ldexp(remainder - ldexp(float(middle), -11), 22));
    return isfinite(float(high));
}

inline void sam_planes_store(device half* planes, ulong n, ulong at, float value, thread uint& invalid) {
    half high, middle, low;
    if (!sam_planes_split(value, high, middle, low)) {
        ((device float*)planes)[sam_planes_floats(n) + at] = value;
        invalid = 1;
    }
    planes[at] = high;
    planes[at + n] = middle;
    planes[at + 2 * n] = low;
}

inline device atomic_uint* sam_planes_flags(device half* planes, ulong n) {
    return (device atomic_uint*)((device float*)planes + sam_planes_floats(n) + n);
}

// Grid row of a tile-ordered row: nine 24x24 tiles of a 72x72 grid, enumerated column by column.
inline uint sam_planes_tile_row(uint row) {
    const uint index = row / 576, local = row % 576;
    return (index % 3 * 24 + local / 24) * 72 + index / 3 * 24 + local % 24;
}

// Pair buffer: two FP16 planes (hi, lo * 2^11), n FP32 restore values, then eight flag words.
inline device atomic_uint* sam_pair_flags(device half* pair, ulong n) {
    return (device atomic_uint*)((device float*)pair + 2 * n);
}

inline void sam_pair_store(device half* pair, ulong n, ulong at, float value, thread uint& invalid) {
    if ((as_type<uint>(value) & 0x7fffffffu) > 0x47000000u) {
        ((device float*)pair)[n + at] = value;
        invalid = 1;
        pair[at] = half(NAN);
        pair[at + n] = half(0.0f);
        return;
    }
    const half h = half(value);
    pair[at] = h;
    pair[at + n] = half((value - float(h)) * 2048.0f);
}

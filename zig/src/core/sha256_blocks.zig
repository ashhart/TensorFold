//! Optional ARM SHA2 and x86 SHA-NI compression use the same SHA256 words and round constants.
const std = @import("std");
const V = @Vector(4, u32);
const K = [64]u32{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

fn words(b: *const [64]u8) [16]u32 {
    var out: [16]u32 = undefined;
    for (&out, 0..) |*w, i| w.* = std.mem.readInt(u32, b[i * 4 ..][0..4], .big);
    return out;
}

pub fn arm(state: *[8]u32, b: *const [64]u8) void {
    const input = words(b);
    var schedule: [16]V = undefined;
    for (0..4) |i| schedule[i] = input[4 * i ..][0..4].*;
    var a: V = state[0..4].*;
    var e: V = state[4..8].*;
    inline for (0..16) |i| {
        if (i >= 4) {
            const s0 = asm volatile (
                \\.arch_extension crypto
                \\sha256su0.4s %[out], %[next]
                : [out] "=&w" (-> V),
                : [_] "0" (schedule[i - 4]),
                  [next] "w" (schedule[i - 3]),
            );
            schedule[i] = asm volatile (
                \\.arch_extension crypto
                \\sha256su1.4s %[out], %[two], %[three]
                : [out] "=&w" (-> V),
                : [_] "0" (s0),
                  [two] "w" (schedule[i - 2]),
                  [three] "w" (schedule[i - 1]),
            );
        }
        const wk = schedule[i] +% @as(V, K[i * 4 ..][0..4].*);
        const old_a = a;
        a = asm volatile (
            \\.arch_extension crypto
            \\sha256h.4s %[out], %[e], %[wk]
            : [out] "=&w" (-> V),
            : [_] "0" (a),
              [e] "w" (e),
              [wk] "w" (wk),
        );
        e = asm volatile (
            \\.arch_extension crypto
            \\sha256h2.4s %[out], %[a], %[wk]
            : [out] "=&w" (-> V),
            : [_] "0" (e),
              [a] "w" (old_a),
              [wk] "w" (wk),
        );
    }
    state[0..4].* = a +% @as(V, state[0..4].*);
    state[4..8].* = e +% @as(V, state[4..8].*);
}

pub fn x86(state: *[8]u32, b: *const [64]u8) void {
    const input = words(b);
    var schedule: [16]V = undefined;
    for (0..4) |i| schedule[i] = input[4 * i ..][0..4].*;
    var abef: V = .{ state[5], state[4], state[1], state[0] };
    var cdgh: V = .{ state[7], state[6], state[3], state[2] };
    inline for (0..16) |i| {
        if (i >= 4) {
            const first = asm volatile ("sha256msg1 %[next], %[out]"
                : [out] "=&x" (-> V),
                : [_] "0" (schedule[i - 4]),
                  [next] "x" (schedule[i - 3]),
            );
            const cross = @shuffle(u32, schedule[i - 2], schedule[i - 1], @Vector(4, i32){ 1, 2, 3, -1 });
            schedule[i] = asm volatile ("sha256msg2 %[last], %[out]"
                : [out] "=&x" (-> V),
                : [_] "0" (first +% cross),
                  [last] "x" (schedule[i - 1]),
            );
        }
        const wk = schedule[i] +% @as(V, K[i * 4 ..][0..4].*);
        cdgh = asm volatile ("sha256rnds2 %[other], %[out]"
            : [out] "=&x" (-> V),
            : [_] "0" (cdgh),
              [other] "x" (abef),
              [_] "{xmm0}" (wk),
        );
        const high = @shuffle(u32, wk, wk, @Vector(4, i32){ 2, 3, 2, 3 });
        abef = asm volatile ("sha256rnds2 %[other], %[out]"
            : [out] "=&x" (-> V),
            : [_] "0" (abef),
              [other] "x" (cdgh),
              [_] "{xmm0}" (high),
        );
    }
    const v = [8]u32{ abef[3], abef[2], cdgh[3], cdgh[2], abef[1], abef[0], cdgh[1], cdgh[0] };
    for (state, v) |*s, value| s.* +%= value;
}

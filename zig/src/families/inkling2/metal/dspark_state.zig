//! Fixed eight-row DSpark scratch and a stream's context rings; the shared lane host sizes and books them.
const std = @import("std");
const mtl = @import("metal");

pub const hidden = 2304;
pub const query = 3072;
pub const kv = 1024;
pub const intermediate = 9216;
pub const rank = 256;
pub const vocabulary = 200058;
pub const block = 8;
pub const absorb_rows = 64;

const fields = [_][]const u8{
    "ids",    "resid", "next_resid", "x",    "y",     "q_raw", "k_raw", "v_raw", "q",     "kb",         "vb", "attn",
    "gate",   "up",    "act",        "base", "fused", "ck",    "cv",    "kappa", "state", "next_state", "z",  "lambda",
    "logits",
};

fn sizes() [fields.len]usize {
    const r = block;
    const a = absorb_rows;
    const b = 2;
    return .{
        r * 4,                r * hidden * b,     r * hidden * b,       r * hidden * b,       r * hidden * b,
        r * query * b,        r * kv * b,         r * kv * b,           r * query * b,        r * kv * b,
        r * kv * b,           r * query * b,      r * intermediate * b, r * intermediate * b, r * intermediate * b,
        (r - 1) * 201024 * b, 4 * a * hidden * b, a * kv * b,           a * kv * b,           (r - 1) * 3 * rank * 4,
        rank * b,             rank * b,           3 * rank * b,         rank * b,             vocabulary * b,
    };
}

pub fn scratchBytes() usize {
    var bytes: usize = 0;
    for (sizes()) |n| bytes += n;
    return bytes;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    resid: mtl.Buffer,
    next_resid: mtl.Buffer,
    x: mtl.Buffer,
    y: mtl.Buffer,
    q_raw: mtl.Buffer,
    k_raw: mtl.Buffer,
    v_raw: mtl.Buffer,
    q: mtl.Buffer,
    kb: mtl.Buffer,
    vb: mtl.Buffer,
    attn: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    base: mtl.Buffer,
    fused: mtl.Buffer,
    ck: mtl.Buffer,
    cv: mtl.Buffer,
    kappa: mtl.Buffer,
    state: mtl.Buffer,
    next_state: mtl.Buffer,
    z: mtl.Buffer,
    lambda: mtl.Buffer,
    logits: mtl.Buffer,

    pub fn init(device: mtl.Device) !Scratch {
        @setEvalBranchQuota(10_000);
        const ns = sizes();
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(@max(ns[i], 16), mtl.ResourceOptions.shared);
            made += 1;
        }
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).deinit();
        s.* = undefined;
    }
};
/// One stream's K and V rings of the drafter's three layers, and its newest target row.
pub const Lane = struct {
    k: [3]mtl.Buffer,
    v: [3]mtl.Buffer,
    newest: mtl.Buffer,

    pub fn init(device: mtl.Device, rings: [3]usize) !Lane {
        var l: Lane = .{ .k = undefined, .v = undefined, .newest = undefined };
        var made: usize = 0;
        errdefer {
            for (0..made) |i| {
                l.k[i].deinit();
                l.v[i].deinit();
            }
        }
        for (rings, 0..) |ring, i| {
            const bytes = ring * kv * 2;
            l.k[i] = try device.buffer(bytes, mtl.ResourceOptions.private);
            l.v[i] = device.buffer(bytes, mtl.ResourceOptions.private) catch |err| {
                l.k[i].deinit();
                return err;
            };
            made += 1;
        }
        l.newest = try device.buffer(hidden * 2, mtl.ResourceOptions.private);
        return l;
    }

    pub fn deinit(l: *Lane) void {
        l.newest.deinit();
        for (0..3) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        }
        l.* = undefined;
    }
};

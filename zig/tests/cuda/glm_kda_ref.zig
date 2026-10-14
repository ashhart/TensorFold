//! The KDA checks' helpers: manifest, fixture bytes, f64 references (kda_chunked's recurrent and chunked ones).

const std = @import("std");

pub const DK = 128;
const BT = 32; // rows of a sub-chunk

pub fn bf(u: u16) f64 {
    const f: f32 = @floatCast(@as(f32, @bitCast(@as(u32, u) << 16)));
    return f;
}

pub fn f32Of(u: u16) f32 {
    return @floatCast(@as(f32, @bitCast(@as(u32, u) << 16)));
}

pub const Manifest = struct {
    parsed: std.json.Parsed(std.json.Value),

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Manifest {
        const path = try std.fs.path.join(gpa, &.{ dir, "manifest.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        return .{ .parsed = parsed };
    }

    pub fn root(m: *const Manifest) std.json.ObjectMap {
        return m.parsed.value.object;
    }

    pub fn deinit(m: *Manifest) void {
        m.parsed.deinit();
    }

    pub fn kda(m: *const Manifest) ?std.json.ObjectMap {
        const v = m.parsed.value.object.get("kda") orelse return null;
        return v.object;
    }
};

/// One array file's raw bytes (little-endian); the caller frees them.
pub fn readFile(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 32));
    defer a.free(raw);
    const out = try a.alignedAlloc(u8, .@"8", raw.len);
    @memcpy(out, raw);
    return out;
}

pub fn f32s(bytes: []const u8) []const f32 {
    return @alignCast(std.mem.bytesAsSlice(f32, bytes));
}

/// A bf16 array's values, widened (exact in f32).
pub fn bf16s(bytes: []const u8) []const f32 {
    const n = bytes.len / 2;
    const src = std.mem.bytesAsSlice(u16, bytes);
    const o = std.heap.page_allocator.alloc(f32, n) catch unreachable;
    for (0..n) |i| o[i] = @floatCast(@as(f32, @bitCast(@as(u32, src[i]) << 16)));
    return o;
}

/// The gated RMSNorm the out kernel applies: norm_w * (y * rsqrt(mean(y^2) + eps)) * sigmoid(gate), f64.
pub fn gatedNorm(out: []f64, norm_w: []const f32, gate: []const f32, eps: f64, h: usize, rows: usize) void {
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var head: usize = 0;
        while (head < h) : (head += 1) {
            const base = (r * h + head) * DK;
            var ss: f64 = 0;
            for (0..DK) |vv| ss += out[base + vv] * out[base + vv];
            const rinv = 1.0 / @sqrt(ss / @as(f64, DK) + eps);
            for (0..DK) |vv| {
                const gt = 1.0 / (1.0 + @exp(-@as(f64, gate[base + vv])));
                out[base + vv] = @as(f64, norm_w[vv]) * out[base + vv] * rinv * gt;
            }
        }
    }
}

pub fn str(o: std.json.ObjectMap, key: []const u8) []const u8 {
    return o.get(key).?.string;
}

pub fn int(o: std.json.ObjectMap, key: []const u8) i64 {
    return o.get(key).?.integer;
}

pub fn float(o: std.json.ObjectMap, key: []const u8) f64 {
    return switch (o.get(key).?) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

/// recurrent_reference in f64: q k lg v fp32 [rows, H, 128], beta [rows, H], state [H, 128, 128] -> out, state f64.
pub fn serialRef(
    a: std.mem.Allocator,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    lg: []const f32,
    beta: []const f32,
    state_in: []const f32,
    gate: []const f32,
    norm_w: []const f32,
    eps: f32,
    h: usize,
    rows: usize,
    out: []f64,
    state: []f64,
) !void {
    const y = try a.alloc(f64, rows * h * DK);
    const S = try a.alloc(f64, h * DK * DK);
    defer a.free(S);
    defer a.free(y);
    for (S, state_in) |*x, s| x.* = s;
    const g = try a.alloc(f64, DK);
    defer a.free(g);
    const delta = try a.alloc(f64, DK);
    defer a.free(delta);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        for (0..h) |head| {
            const base = (r * h + head) * DK;
            for (0..DK) |c| g[c] = @exp(@as(f64, lg[base + c]));
            for (0..DK) |c| {
                for (0..DK) |vv| S[head * DK * DK + vv * DK + c] *= g[c];
            }
            for (0..DK) |vv| {
                var kv: f64 = 0;
                for (0..DK) |c| kv += S[head * DK * DK + vv * DK + c] * @as(f64, k[base + c]);
                delta[vv] = (@as(f64, v[base + vv]) - kv) * @as(f64, beta[r * h + head]);
            }
            for (0..DK) |c| {
                for (0..DK) |vv| S[head * DK * DK + vv * DK + c] += @as(f64, k[base + c]) * delta[vv];
            }
            for (0..DK) |vv| {
                var o: f64 = 0;
                for (0..DK) |c| o += S[head * DK * DK + vv * DK + c] * @as(f64, q[base + c]);
                y[base + vv] = o;
            }
        }
    }
    for (state, S) |*x, s| x.* = s;
    gatedNorm(y, norm_w, gate, @as(f64, eps), h, rows);
    @memcpy(out, y);
}

/// chunked_reference in f64: serialRef's inputs in sub-chunks at absolute pos; padded rows decay 1, no k, v or beta.
pub fn chunkedRef(
    a: std.mem.Allocator,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    lg: []const f32,
    beta: []const f32,
    state_in: []const f32,
    gate: []const f32,
    norm_w: []const f32,
    eps: f32,
    h: usize,
    rows: usize,
    pos: usize,
    out: []f64,
    state: []f64,
) !void {
    const off = pos % BT;
    const nc = (rows + off + BT - 1) / BT;

    const S = try a.alloc(f64, h * DK * DK);
    defer a.free(S);
    for (S, state_in) |*x, s| x.* = s;
    const y = try a.alloc(f64, rows * h * DK);
    defer a.free(y);

    const gx = try a.alloc(f64, BT * DK); // [t][c] = g of row t (1 outside)
    defer a.free(gx);
    const fwd = try a.alloc(f64, BT * DK); // [t][c] = E(0 -> t)
    defer a.free(fwd);
    const back = try a.alloc(f64, BT * DK); // [t][c] = E(t -> 31)
    defer a.free(back);
    const Vn = try a.alloc(f64, BT * DK);
    defer a.free(Vn);
    const L = try a.alloc(f64, BT * BT);
    defer a.free(L);
    const M = try a.alloc(f64, BT * BT);
    defer a.free(M);
    const T = try a.alloc(f64, BT * BT);
    defer a.free(T);
    @memset(L, 0);
    const e_run = try a.alloc(f64, DK); // E(t -> s) of the pair being accumulated
    defer a.free(e_run);
    @memset(M, 0);
    @memset(T, 0);
    const Y = try a.alloc(f64, BT * DK);
    defer a.free(Y);

    var c: usize = 0;
    while (c < nc) : (c += 1) {
        for (0..h) |head| {
            for (0..BT) |t| {
                const row_i = c * BT + t;
                const ok = row_i >= off and row_i < off + rows;
                const row = row_i -| off;
                for (0..DK) |cc| gx[t * DK + cc] = if (ok) @exp(@as(f64, lg[(row * h + head) * DK + cc])) else 1.0;
            }
            for (0..DK) |cc| {
                fwd[cc] = gx[cc];
                for (1..BT) |t| fwd[t * DK + cc] = fwd[(t - 1) * DK + cc] * gx[t * DK + cc];
                back[(BT - 1) * DK + cc] = 1.0;
                var t: usize = BT - 1;
                while (t > 0) : (t -= 1) back[(t - 1) * DK + cc] = back[t * DK + cc] * gx[t * DK + cc];
            }
            // L[s][t] (t < s) and M[s][t] (t <= s); E(t -> s) multiplied up from 1 per t, exactly the engine's order
            for (0..BT) |t| {
                for (0..DK) |cc| e_run[cc] = 1.0;
                var s: usize = t;
                while (s < BT) : (s += 1) {
                    if (s > t) {
                        for (0..DK) |cc| e_run[cc] *= gx[s * DK + cc];
                    }
                    var akk: f64 = 0;
                    var aqk: f64 = 0;
                    for (0..DK) |cc| {
                        const kt = rowAt(k, h, off, rows, c * BT + t, head, cc);
                        akk += rowAt(k, h, off, rows, c * BT + s, head, cc) * e_run[cc] * kt;
                        aqk += rowAt(q, h, off, rows, c * BT + s, head, cc) * e_run[cc] * kt;
                    }
                    const bs = rowBeta(beta, h, off, rows, c * BT + s, head);
                    if (s > t) L[s * BT + t] = bs * akk;
                    M[s * BT + t] = aqk;
                }
            }
            // T = (I + L)^-1 by forward substitution: T[s] = -(L[s] T) with T[s][s] = 1
            for (0..BT) |s| {
                for (0..BT) |t| T[s * BT + t] = if (s == t) 1.0 else 0.0;
            }
            for (1..BT) |s| {
                for (0..BT) |m| {
                    var acc: f64 = 0;
                    for (0..BT) |mm| acc += L[s * BT + mm] * T[mm * BT + m];
                    T[s * BT + m] = -acc;
                }
                T[s * BT + s] = 1.0;
            }
            // Y[t][vv] = beta_t (v_t[vv] - (k_t fwd_t) S^T)
            for (0..BT) |t| {
                const bt = rowBeta(beta, h, off, rows, c * BT + t, head);
                for (0..DK) |vv| {
                    var kv: f64 = 0;
                    for (0..DK) |cc| kv += rowAt(k, h, off, rows, c * BT + t, head, cc) * fwd[t * DK + cc] *
                        S[head * DK * DK + vv * DK + cc];
                    Y[t * DK + vv] = bt * (rowAt(v, h, off, rows, c * BT + t, head, vv) - kv);
                }
            }
            // Vn[t][vv] = sum_s T[t][s] Y[s][vv], then the rows: (q fwd) S^T + M V'
            for (0..BT) |t| for (0..DK) |vv| {
                var vn: f64 = 0;
                for (0..BT) |s| vn += T[t * BT + s] * Y[s * DK + vv];
                Vn[t * DK + vv] = vn;
            };
            for (0..BT) |t| {
                if (c * BT + t < off) continue;
                const gr = c * BT + t - off;
                if (gr >= rows) continue;
                const row = gr;
                for (0..DK) |vv| {
                    var o: f64 = 0;
                    for (0..DK) |cc| o += rowAt(q, h, off, rows, c * BT + t, head, cc) * fwd[t * DK + cc] *
                        S[head * DK * DK + vv * DK + cc];
                    for (0..BT) |s| o += M[t * BT + s] * Vn[s * DK + vv];
                    y[(row * h + head) * DK + vv] = o;
                }
            }
            // the state: S diag(E(0 -> 31)) + V'^T (k back)
            for (0..DK) |cc| {
                for (0..DK) |vv| S[head * DK * DK + vv * DK + cc] *= fwd[(BT - 1) * DK + cc];
            }
            for (0..BT) |t| {
                for (0..DK) |cc| {
                    const kb = rowAt(k, h, off, rows, c * BT + t, head, cc) * back[t * DK + cc];
                    for (0..DK) |vv| S[head * DK * DK + vv * DK + cc] += kb * Vn[t * DK + vv];
                }
            }
        }
    }
    for (state, S) |*x, s| x.* = s;
    gatedNorm(y, norm_w, gate, @as(f64, eps), h, rows);
    @memcpy(out, y);
}

fn rowAt(arr: []const f32, h: usize, off: usize, rows: usize, gr: usize, head: usize, c: usize) f64 {
    if (gr < off or gr >= off + rows) return 0;
    return arr[((gr - off) * h + head) * DK + c];
}

fn rowBeta(beta: []const f32, h: usize, off: usize, rows: usize, gr: usize, head: usize) f64 {
    if (gr < off or gr >= off + rows) return 0;
    return beta[(gr - off) * h + head];
}

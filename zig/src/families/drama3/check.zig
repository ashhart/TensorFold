//! drama3-generate's checks against the PyTorch oracle's dumps: prompt layers, teacher-forced frames and the codec.
const std = @import("std");
const core = @import("core");
const d3 = @import("drama3");
const gen = @import("generate.zig");

const W = d3.config.frame_width;
const H = d3.config.hidden;
const Args = gen.Args;

const Check = struct {
    gpa: std.mem.Allocator,
    e: *d3.Engine,
    ref: *core.safetensors.File,
    worst: f64 = 1.0,
    n: usize = 0,
    fails: usize = 0,
    // frame 0's heads (0: the candidate logits, 1 + i: codebook head i): both sides' top two
    tops: [10]?[2]Top2 = @splat(null),

    fn refF32(c: *Check, name: []const u8, rows: usize, width: usize) !?[]f32 {
        const t = c.ref.get(name) orelse return null;
        const n = t.numel();
        // the oracle's decode0 rows are [width]; prompt rows [rows, width]: compare the last `rows * width` values
        if (n < rows * width) return null;
        const out = try c.gpa.alloc(f32, rows * width);
        const skip = n - rows * width;
        switch (t.dtype) {
            .bf16 => {
                const v = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, t.bytes));
                for (out, 0..) |*o, j| o.* = @bitCast(@as(u32, v[skip + j]) << 16);
            },
            .f32 => {
                const v = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, t.bytes));
                for (out, 0..) |*o, j| o.* = v[skip + j];
            },
            else => return null,
        }
        return out;
    }

    fn probe(ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize, is_f32: bool) anyerror!void {
        const c: *Check = @ptrCast(@alignCast(ctx));
        const mine = try c.gpa.alloc(f32, rows * width);
        defer c.gpa.free(mine);
        if (is_f32) {
            try c.e.download(ptr, std.mem.sliceAsBytes(mine));
        } else {
            const raw = try c.gpa.alloc(u16, rows * width);
            defer c.gpa.free(raw);
            try c.e.download(ptr, std.mem.sliceAsBytes(raw));
            for (mine, raw) |*m, r| m.* = @bitCast(@as(u32, r) << 16);
        }
        if (std.mem.startsWith(u8, name, "dbg.")) {
            for ([_]usize{ 100, 140, 147, 149, 153 }) |r| if (r < rows) {
                var ss: f64 = 0;
                for (mine[r * width ..][0..width]) |x| ss += @as(f64, x) * x;
                std.debug.print("  {s} row {d} |x| {d:.3}\n", .{ name, r, @sqrt(ss) });
            };
            return;
        }
        const ref = (try c.refF32(name, rows, width)) orelse {
            std.debug.print("  {s:<36} (no oracle tensor)\n", .{name});
            return;
        };
        defer c.gpa.free(ref);
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        var maxd: f64 = 0;
        var maxr: f64 = 0;
        for (mine, ref) |x, y| {
            dot += @as(f64, x) * y;
            na += @as(f64, x) * x;
            nb += @as(f64, y) * y;
            maxd = @max(maxd, @abs(@as(f64, x) - y));
            maxr = @max(maxr, @abs(@as(f64, y)));
        }
        // the worst row's cosine too
        var worst_row: f64 = 1;
        for (0..rows) |r| {
            var d: f64 = 0;
            var a2: f64 = 0;
            var b2: f64 = 0;
            for (mine[r * width ..][0..width], ref[r * width ..][0..width]) |x, y| {
                d += @as(f64, x) * y;
                a2 += @as(f64, x) * x;
                b2 += @as(f64, y) * y;
            }
            if (a2 > 0 and b2 > 0) worst_row = @min(worst_row, d / @sqrt(a2 * b2));
        }
        const cos = if (na > 0 and nb > 0) dot / @sqrt(na * nb) else 1.0;
        if (std.mem.startsWith(u8, name, "tower_decode0.head") or std.mem.eql(u8, name, "tower_decode0.candidate_logits")) {
            const hi: usize = if (name[18] >= '0' and name[18] <= '9') 1 + (name[18] - '0') else 0;
            c.tops[hi] = .{ top2(mine), top2(ref) };
        }
        c.n += 1;
        c.worst = @min(c.worst, worst_row);
        var wr: usize = 0;
        var wv: f64 = 2;
        for (0..rows) |r| {
            var d: f64 = 0;
            var a2: f64 = 0;
            var b2: f64 = 0;
            for (mine[r * width ..][0..width], ref[r * width ..][0..width]) |x, y| {
                d += @as(f64, x) * y;
                a2 += @as(f64, x) * x;
                b2 += @as(f64, y) * y;
            }
            const cr = if (a2 > 0 and b2 > 0) d / @sqrt(a2 * b2) else 1.0;
            if (cr < wv) {
                wv = cr;
                wr = r;
            }
        }
        const ok = worst_row >= 0.999;
        if (!ok) c.fails += 1;
        std.debug.print("  {s:<36} rows {d:>4} cos {d:.6} worst-row {d:.6} (row {d}) max|d| {d:.4} (max|ref| {d:.2}) {s}\n", .{ name, rows, cos, worst_row, wr, maxd, maxr, if (ok) "ok" else "LOW" });
    }
};

pub fn check(gpa: std.mem.Allocator, io: std.Io, a: Args, dir: []const u8) !u8 {
    const vpath = try std.fs.path.join(gpa, &.{ dir, "vectors.safetensors" });
    var ref = try core.safetensors.File.open(gpa, io, vpath);
    defer ref.close(io);
    const pr = ref.get("prompt_rows") orelse return error.NoPromptRows;
    const n = pr.dim(0);
    if (pr.dim(1) != W or pr.dtype != .i64) return error.BadPromptRows;
    const rows = try gpa.alloc([W]i32, n);
    defer gpa.free(rows);
    const v = std.mem.bytesAsSlice(i64, @as([]align(1) const u8, pr.bytes));
    for (rows, 0..) |*r, j| for (r, 0..) |*x, c| {
        x.* = @intCast(v[j * W + c]);
    };
    var opts = a.opts;
    opts.max_rows = @max(opts.max_rows, n);
    const e = try d3.Engine.create(gpa, io, a.model, opts);
    defer e.destroy();
    var c: Check = .{ .gpa = gpa, .e = e, .ref = &ref };
    const probe: d3.engine.Probe = .{ .ctx = &c, .f = Check.probe };
    std.debug.print("drama3 check: {d} prompt rows from {s}\n", .{ n, vpath });
    try e.prefill(rows, probe);
    if (ref.get("backbone_final_norm")) |hf| if (hf.dtype == .bf16 and hf.dim(0) == n) {
        std.debug.print("  -- the tower prefill again from the oracle's backbone_final_norm rows:\n", .{});
        try e.towerOnly(n, hf.bytes, probe);
        try e.prefill(rows, null); // restore my own state for the decode comparison
    };
    // frame 0 with the probe (tower_decode0.*), then greedy frames against the oracle's
    const smp: d3.Sampling = .{ .temperature = 0, .top_k = 1 };
    const oref = ref.get("frames_all") orelse ref.get("frames") orelse return error.NoFrames;
    const nref = oref.dim(0);
    const ov = std.mem.bytesAsSlice(i64, @as([]align(1) const u8, oref.bytes));
    const cand_ref = ref.get("decode.candidate_logits");
    const nc = e.w.n_cand;
    const mine_logits = try gpa.alloc(f32, nc);
    defer gpa.free(mine_logits);
    var same: usize = 0;
    var diverged = false;
    const want = @min(a.check_frames, nref + 1);
    var f: usize = 0;
    var got: [W]i32 = undefined;
    while (f < want) : (f += 1) {
        // my candidate logits for this frame (the step recomputes them; read them back after it)
        try e.frameStep(smp, false, if (f == 0) probe else null);
        try e.download(e.cand_logits, std.mem.sliceAsBytes(mine_logits));
        try e.download(e.frames_out + f * W * 4, std.mem.asBytes(&got));
        if (f >= nref) {
            std.debug.print("  frame {d}: mine {d} (the oracle stopped at {d} frames: eos expected)\n", .{ f, got[0], nref });
            if (got[0] == e.c.ids.eos) same += 1;
            break;
        }
        var match = true;
        for (0..W) |j| match = match and got[j] == ov[f * W + j];
        if (match) {
            same += 1;
            continue;
        }
        diverged = true;
        std.debug.print("  frame {d} differs: mine [", .{f});
        for (got) |x| std.debug.print(" {d}", .{x});
        std.debug.print(" ] oracle [", .{});
        for (0..W) |j| std.debug.print(" {d}", .{ov[f * W + j]});
        std.debug.print(" ]\n", .{});
        var col: usize = 0;
        while (col < W and got[col] == ov[f * W + col]) col += 1;
        const head: usize = if (col <= 1) 0 else col - 1;
        if (f == 0) if (c.tops[head]) |t| {
            std.debug.print("    first difference: column {d} ({s} {d}); mine top {d} ({d:.4}) second {d} ({d:.4}) margin {d:.4}; oracle top {d} ({d:.4}) second {d} ({d:.4}) margin {d:.4}\n", .{ col, if (head == 0) "candidate logits" else "codebook head", if (head == 0) 0 else head - 1, t[0].i1, t[0].v1, t[0].i2, t[0].v2, t[0].v1 - t[0].v2, t[1].i1, t[1].v1, t[1].i2, t[1].v2, t[1].v1 - t[1].v2 });
        };
        // the main token's margin on both sides (next frame's candidate logits)
        const mm = top2(mine_logits);
        std.debug.print("    my candidate logits: top {d} ({d:.4}) second {d} ({d:.4})\n", .{ mm.i1, mm.v1, mm.i2, mm.v2 });
        if (cand_ref) |cr| if (f < cr.dim(0)) {
            const rv = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, cr.bytes))[f * nc ..][0..nc];
            const rcopy = try gpa.alloc(f32, nc);
            defer gpa.free(rcopy);
            @memcpy(rcopy, rv);
            const rm = top2(rcopy);
            std.debug.print("    oracle candidate logits: top {d} ({d:.4}) second {d} ({d:.4}); mine at the oracle's top: {d:.4}\n", .{ rm.i1, rm.v1, rm.i2, rm.v2, mine_logits[rm.i1] });
        };
        break;
    }
    _ = &diverged;
    // teacher forcing: the oracle's frames fed back, comparing greedy picks and decode.backbone_final_norm rows
    var tf_agree: [W]usize = @splat(0);
    var tf_frames: usize = 0;
    var tf_worst: f64 = 1;
    {
        try e.prefill(rows, null);
        const dev = try e.scratch(2 * W * 4);
        const hidden_ref = ref.get("decode.backbone_final_norm");
        const hbuf = try gpa.alloc(u16, H);
        defer gpa.free(hbuf);
        const nf = @min(nref, a.check_frames * 5);
        var mg: Margins = .{ .gpa = gpa, .e = e };
        defer gpa.free(mg.buf);
        const mg_probe: d3.engine.Probe = .{ .ctx = &mg, .f = Margins.probe };
        for (0..nf) |fi| {
            if (hidden_ref) |hr| if (fi < hr.dim(0)) {
                try e.download(e.hlast, std.mem.sliceAsBytes(hbuf));
                const rv = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, hr.bytes))[fi * H ..][0..H];
                var d: f64 = 0;
                var a2: f64 = 0;
                var b2: f64 = 0;
                for (hbuf, rv) |x16, y16| {
                    const x: f64 = @as(f32, @bitCast(@as(u32, x16) << 16));
                    const y: f64 = @as(f32, @bitCast(@as(u32, y16) << 16));
                    d += x * y;
                    a2 += x * x;
                    b2 += y * y;
                }
                tf_worst = @min(tf_worst, d / @sqrt(a2 * b2));
            };
            var fr: [W]i32 = undefined;
            for (0..W) |j| fr[j] = @intCast(ov[fi * W + j]);
            try e.put(dev, std.mem.asBytes(&fr));
            mg.frame = fi;
            mg.want = fr;
            try e.frameForced(.{ .frame = dev, .picks = dev + W * 4 }, mg_probe);
            var picks: [W]i32 = undefined;
            try e.download(dev + W * 4, std.mem.asBytes(&picks));
            for (0..W) |j| tf_agree[j] += @intFromBool(picks[j] == fr[j]);
            tf_frames += 1;
        }
        std.debug.print("  teacher-forced {d} frames: my greedy picks equal the oracle's: main {d}", .{ tf_frames, tf_agree[0] });
        for (2..W) |j| std.debug.print(" c{d} {d}", .{ j - 1, tf_agree[j] });
        std.debug.print("; decode-row final-norm hidden worst cos {d:.6} over the oracle's dumped rows\n", .{tf_worst});
        std.debug.print("  teacher-forced differences: {d}, my largest margin over the oracle's pick {d:.4}\n", .{ mg.diffs, mg.worst });
    }
    const pass = c.fails == 0 and (same >= @min(20, want));
    var tf_total: usize = 0;
    for (tf_agree) |x| tf_total += x;
    std.debug.print("RESULT drama3 check {s}: {d} probes, {d} below cos 0.999 (worst row {d:.6}); greedy frames identical {d} of {d} (oracle made {d}); teacher-forced picks {d:.2}% of {d}\n", .{ if (pass) "PASS" else "FAIL", c.n, c.fails, c.worst, same, want, nref, 100.0 * @as(f64, @floatFromInt(tf_total - tf_agree[1])) / @as(f64, @floatFromInt(@max(1, tf_frames * (W - 1)))), tf_frames * (W - 1) });
    return if (pass) 0 else 1;
}

/// Every teacher-forced pick that differs from the oracle's, with my logit margin over the oracle's pick.
const Margins = struct {
    gpa: std.mem.Allocator,
    e: *d3.Engine,
    buf: []f32 = &.{},
    frame: usize = 0,
    want: [W]i32 = undefined,
    diffs: usize = 0,
    worst: f32 = 0,

    fn probe(ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize, is_f32: bool) anyerror!void {
        const m: *Margins = @ptrCast(@alignCast(ctx));
        if (!is_f32 or rows != 1) return;
        var col: usize = undefined;
        var pick: usize = undefined;
        if (std.mem.endsWith(u8, name, "candidate_logits")) {
            col = 0;
            const t = m.want[0];
            pick = if (t == m.e.c.ids.eos) 0 else @intCast(t - m.e.c.ids.sem_begin + 1);
        } else if (std.mem.indexOf(u8, name, ".head")) |at| {
            const hi = name[at + 5] - '0';
            col = 2 + hi;
            pick = @intCast(m.want[col]);
        } else return;
        if (m.buf.len < width) {
            m.gpa.free(m.buf);
            m.buf = try m.gpa.alloc(f32, width);
        }
        const x = m.buf[0..width];
        try m.e.download(ptr, std.mem.sliceAsBytes(x));
        const t2 = top2(x);
        if (t2.i1 == pick) return;
        const margin = t2.v1 - x[pick];
        m.diffs += 1;
        m.worst = @max(m.worst, margin);
        std.debug.print("    tf frame {d} column {d}: mine {d} ({d:.4}), oracle {d} ({d:.4} here), margin {d:.4}\n", .{ m.frame, col, t2.i1, t2.v1, pick, x[pick], margin });
    }
};

const Top2 = struct { i1: usize, v1: f32, i2: usize, v2: f32 };

fn top2(x: []const f32) Top2 {
    var r: Top2 = .{ .i1 = 0, .v1 = -std.math.inf(f32), .i2 = 0, .v2 = -std.math.inf(f32) };
    for (x, 0..) |v, j| {
        if (v > r.v1) {
            r.i2 = r.i1;
            r.v2 = r.v1;
            r.i1 = j;
            r.v1 = v;
        } else if (v > r.v2) {
            r.i2 = j;
            r.v2 = v;
        }
    }
    return r;
}

// ------------------------------------------------------------------------------------------------ codec check

fn refTensor(gpa: std.mem.Allocator, f: *const core.safetensors.File, name: []const u8) !?[]f32 {
    const t = f.get(name) orelse return null;
    if (t.dtype != .f32) return error.BadRef;
    const out = try gpa.alloc(f32, t.numel());
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

fn snrDb(ref: []const f32, got: []const f32) f64 {
    var sig: f64 = 0;
    var err: f64 = 0;
    for (ref, got) |r, g| {
        sig += @as(f64, r) * r;
        err += (@as(f64, r) - g) * (@as(f64, r) - g);
    }
    return 10.0 * std.math.log10(sig / @max(err, 1e-30));
}

/// --codec-check: the streamed decode's SNR against the PyTorch codec's waveform, and a clip's codes against its codes.
pub fn codecCheck(gpa: std.mem.Allocator, io: std.Io, a: Args, ref_path: []const u8) !u8 {
    var ref = try core.safetensors.File.open(gpa, io, ref_path);
    defer ref.close(io);
    const t0 = d3.engine.nowNs(io);
    const c = try d3.tts.CodecOnly.open(gpa, io, a.codec orelse return error.CodecRequired);
    defer c.close();
    std.debug.print("codec: {d:.2} GB of weights in {d:.1} s\n", .{ @as(f64, @floatFromInt(c.weightBytes())) / 1e9, @as(f64, @floatFromInt(d3.engine.nowNs(io) - t0)) / 1e9 });
    var pass = true;
    if (a.frames) |fp| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, fp, gpa, .limited(1 << 28));
        const nf = bytes.len / 44;
        const fb = try c.buffer(bytes.len);
        try c.upload(fb, bytes);
        const ab = try c.buffer(nf * 2048 * 4);
        for (0..2) |rep| {
            try c.reset();
            try c.sync();
            const s0 = d3.engine.nowNs(io);
            const first = @min(a.first_chunk, nf);
            try c.decode(fb, first, ab);
            try c.sync();
            const s1 = d3.engine.nowNs(io);
            try c.decode(fb + first * 44, nf - first, ab + first * 2048 * 4);
            try c.sync();
            const s2 = d3.engine.nowNs(io);
            const audio = try gpa.alloc(f32, nf * 2048);
            defer gpa.free(audio);
            try c.download(ab, std.mem.sliceAsBytes(audio));
            const secs = @as(f64, @floatFromInt(nf * 2048)) / 44100.0;
            std.debug.print("decode run {d}: {d} frames ({d:.2} s): first chunk of {d} in {d:.1} ms, all in {d:.1} ms = {d:.0}x real time\n", .{ rep, nf, secs, first, @as(f64, @floatFromInt(s1 - s0)) / 1e6, @as(f64, @floatFromInt(s2 - s0)) / 1e6, secs / (@as(f64, @floatFromInt(s2 - s0)) / 1e9) });
            if (rep == 1) if (try refTensor(gpa, &ref, "dec.audio")) |r| {
                defer gpa.free(r);
                const n = @min(r.len, audio.len);
                const snr = snrDb(r[0..n], audio[0..n]);
                std.debug.print("  waveform vs the PyTorch codec (fp32, no TF32): SNR {d:.2} dB over {d} samples\n", .{ snr, n });
                pass = pass and snr >= 35;
                const wavb = try d3.wav.encode16(gpa, audio, 44100);
                defer gpa.free(wavb);
                try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "codec_check.wav", .data = wavb });
            };
        }
    }
    if (a.ref_audio) |wp| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, wp, gpa, .limited(1 << 30));
        const au = try d3.wav.parse(gpa, bytes);
        const x44 = try d3.wav.resample(gpa, au.samples, au.rate, 44100);
        if (try refTensor(gpa, &ref, "enc.audio44k")) |r| {
            const n = @min(r.len, x44.len);
            std.debug.print("  resampled clip ({d} Hz -> 44.1 kHz, {d} vs {d} samples): SNR {d:.2} dB vs torchaudio\n", .{ au.rate, x44.len, r.len, snrDb(r[0..n], x44[0..n]) });
        }
        const P = struct {
            gpa: std.mem.Allocator,
            ref: *core.safetensors.File,
            c: *d3.tts.CodecOnly,
            fn f(pc: *anyopaque, name: []const u8, ptr: u64, rows: usize, cols: usize) anyerror!void {
                const p: *@This() = @ptrCast(@alignCast(pc));
                const r = (try refTensor(p.gpa, p.ref, name)) orelse return;
                const mine = try p.gpa.alloc(f32, rows * cols);
                try p.c.download(ptr, std.mem.sliceAsBytes(mine));
                const n = @min(r.len, mine.len);
                std.debug.print("  {s}: {d} vs {d} values, SNR {d:.2} dB\n", .{ name, mine.len, r.len, snrDb(r[0..n], mine[0..n]) });
            }
        };
        var pctx: P = .{ .gpa = gpa, .ref = &ref, .c = c };
        c.setProbe(.{ .ctx = &pctx, .f = P.f });
        const e0 = d3.engine.nowNs(io);
        const codes = try c.encode(x44);
        c.setProbe(null);
        const e1 = d3.engine.nowNs(io);
        std.debug.print("encode: {d} frames in {d:.1} ms\n", .{ codes.len, @as(f64, @floatFromInt(e1 - e0)) / 1e6 });
        if (ref.get("enc.codes")) |rc| {
            const rv = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, rc.bytes));
            const nf = @min(codes.len, rc.dim(0));
            var same: [10]usize = @splat(0);
            for (0..nf) |t| for (0..10) |j| {
                same[j] += @intFromBool(codes[t][j] == rv[t * 10 + j]);
            };
            var tot: usize = 0;
            std.debug.print("  codes vs the PyTorch codec ({d} vs {d} frames): per codebook", .{ codes.len, rc.dim(0) });
            for (same) |x| {
                tot += x;
                std.debug.print(" {d}", .{x});
            }
            const pct = 100.0 * @as(f64, @floatFromInt(tot)) / @as(f64, @floatFromInt(nf * 10));
            std.debug.print(" of {d}: {d:.2}%\n", .{ nf, pct });
            pass = pass and pct >= 97;
        }
    }
    std.debug.print("RESULT drama3 codec-check {s}\n", .{if (pass) "PASS" else "FAIL"});
    return if (pass) 0 else 1;
}

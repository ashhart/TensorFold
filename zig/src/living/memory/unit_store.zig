//! Living-memory unit tensor serialization and parsing: per-unit tensors, unit JSON, whiteners and metadata.
const std = @import("std");
const whitener = @import("whitener.zig");
const gate = @import("gate.zig");
const store = @import("store.zig");
const types = @import("types.zig");
const Bank = @import("bank.zig").Bank;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const ArchSpec = types.ArchSpec;
const KvLayerData = types.KvLayerData;
const RecLayerData = types.RecLayerData;
const Unit = types.Unit;

/// Serializes one unit's metadata and tensors into a self-contained safetensors slice for the delta journal.
pub fn serializeSingleUnit(self: *const Bank, gpa: Allocator, u: *const Unit) ![]u8 {
    const arch = self.backend.arch;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const rank: u16 = if (u.rec.len > 0) u.rec[0].rank else 0;
    const has_conv: bool = u.rec.len > 0 and u.rec[0].conv.len == arch.conv_elems;
    const ujson = try std.fmt.allocPrint(
        a,
        "{{\"uid\":{d},\"pid\":{d},\"tok\":{d},\"nl\":{d},\"nbl\":{d},\"ns\":{d},\"rk\":{d},\"cv\":{s},\"sm\":{s},\"st\":{s},\"dg\":{s},\"nsemb\":{d},\"ep\":{d},\"sup\":{d},\"rf\":{d}}}",
        .{
            u.unit_id,
            u.parent_id,
            u.n_tokens,
            u.gate.n_lex,
            u.gate.bodyLexCount(),
            u.gate.n_sem,
            rank,
            if (has_conv) "true" else "false",
            if (u.gate.is_small_fact) "true" else "false",
            if (u.gate.is_standing) "true" else "false",
            if (u.gate.is_digest) "true" else "false",
            @as(u32, @bitCast(u.gate.neg_sem_max)), // its bits: the value routing compares, exactly
            u.gate.epoch,
            u.gate.superseded_by,
            u.gate.reinforcements,
        },
    );
    const meta = [_]store.MetaKV{
        .{ .key = "format", .value = store.format_tag },
        .{ .key = "unit", .value = ujson },
    };
    var tensors: std.ArrayList(store.TensorSpec) = .empty;
    try appendUnitTensors(a, &tensors, arch, u, 0);
    return store.serializeToBytes(gpa, &meta, tensors.items);
}

pub fn appendUnitTensors(a: Allocator, tensors: *std.ArrayList(store.TensorSpec), arch: ArchSpec, u: *const Unit, ui: usize) !void {
    // every tensor as computed: prototypes and weights F32, K and V BF16, recurrent factors F32
    const f32s = [_]struct { []const u8, []const f32, usize }{
        .{ "gl", u.gate.lex_protos, u.gate.n_lex },
        .{ "gl.w", u.gate.lex_weights, 0 },
        .{ "gs", u.gate.sem_protos, u.gate.n_sem },
        .{ "gs.w", u.gate.sem_weights, 0 },
    };
    for (f32s) |t| if (t[1].len > 0) {
        const shape = if (t[2] > 0) try a.dupe(usize, &.{ t[2], arch.hidden }) else try a.dupe(usize, &.{t[1].len});
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.{s}", .{ ui, t[0] }), .dtype = .f32, .shape = shape, .bytes = std.mem.sliceAsBytes(t[1]) });
    };
    for (u.kv, 0..) |kv, l| {
        const sh = try a.dupe(usize, &.{ arch.kv_heads, u.n_tokens, arch.head_dim });
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.kv.{d}.k", .{ ui, l }), .dtype = .bf16, .shape = sh, .bytes = std.mem.sliceAsBytes(kv.k) });
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.kv.{d}.v", .{ ui, l }), .dtype = .bf16, .shape = sh, .bytes = std.mem.sliceAsBytes(kv.v) });
    }
    if (u.rec.len != arch.num_rec) return;
    for (u.rec, 0..) |rl, r| {
        const rk: usize = rl.rank;
        if (rk == 0) continue;
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.rc.{d}.u", .{ ui, r }), .dtype = .f32, .shape = try a.dupe(usize, &.{ arch.rec_heads, arch.rec_rows, rk }), .bytes = std.mem.sliceAsBytes(rl.u) });
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.rc.{d}.v", .{ ui, r }), .dtype = .f32, .shape = try a.dupe(usize, &.{ arch.rec_heads, rk, arch.rec_cols }), .bytes = std.mem.sliceAsBytes(rl.v) });
        if (rl.conv.len == arch.conv_elems) {
            try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.rc.{d}.cv", .{ ui, r }), .dtype = .bf16, .shape = try a.dupe(usize, &.{arch.conv_elems}), .bytes = std.mem.sliceAsBytes(rl.conv) });
        }
    }
}

pub fn loadUnitFromObject(gpa: Allocator, lf: *const store.LoadedFile, arch: ArchSpec, item: std.json.Value, ui: usize) !Unit {
    if (item != .object) return error.BadLivingWeights;
    const o = item.object;
    const uid: u32 = @intCast((o.get("uid") orelse return error.BadLivingWeights).integer);
    const pid: u32 = @intCast((o.get("pid") orelse return error.BadLivingWeights).integer);
    const n_tok: u32 = @intCast((o.get("tok") orelse return error.BadLivingWeights).integer);
    const n_lex: usize = @intCast((o.get("nl") orelse return error.BadLivingWeights).integer);
    const n_body_lex: usize = if (o.get("nbl")) |v| @intCast(v.integer) else n_lex;
    const n_sem: usize = @intCast((o.get("ns") orelse return error.BadLivingWeights).integer);
    const rk: u16 = @intCast((o.get("rk") orelse return error.BadLivingWeights).integer);
    const has_conv: bool = if (o.get("cv")) |v| v.bool else (rk > 0);
    const is_sm = (o.get("sm") orelse return error.BadLivingWeights).bool;
    const is_st = if (o.get("st")) |v| v.bool else false;
    const is_dg = if (o.get("dg")) |v| v.bool else false;
    const epoch: u64 = if (o.get("ep")) |v| @intCast(v.integer) else 0;
    const sup: u32 = if (o.get("sup")) |v| @intCast(v.integer) else 0;
    const rf: u16 = if (o.get("rf")) |v| @intCast(v.integer) else 1;
    const nsem_bits = o.get("nsemb") orelse return error.BadLivingWeights;
    if (nsem_bits != .integer or nsem_bits.integer < 0 or nsem_bits.integer > std.math.maxInt(u32)) return error.BadLivingWeights;
    const neg_sem: f32 = @bitCast(@as(u32, @intCast(nsem_bits.integer)));

    var name_buf: [96]u8 = undefined;
    const gl_v = if (n_lex > 0) try lf.copyF32(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.gl", .{ui}), n_lex * arch.hidden) else try gpa.alloc(f32, 0);
    errdefer gpa.free(gl_v);
    const gl_w = if (n_lex > 0) try lf.copyF32(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.gl.w", .{ui}), n_lex) else try gpa.alloc(f32, 0);
    errdefer gpa.free(gl_w);
    const gs_v = if (n_sem > 0) try lf.copyF32(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.gs", .{ui}), n_sem * arch.hidden) else try gpa.alloc(f32, 0);
    errdefer gpa.free(gs_v);
    const gs_w = if (n_sem > 0) try lf.copyF32(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.gs.w", .{ui}), n_sem) else try gpa.alloc(f32, 0);
    errdefer gpa.free(gs_w);

    const kv = try gpa.alloc(KvLayerData, arch.num_attn);
    var kv_done: usize = 0;
    errdefer {
        for (kv[0..kv_done]) |*k| k.deinit(gpa);
        gpa.free(kv);
    }
    const kv_rows = arch.kv_heads * n_tok;
    for (0..arch.num_attn) |l| {
        const k = try lf.copyBf16(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.kv.{d}.k", .{ ui, l }), kv_rows * arch.head_dim);
        errdefer gpa.free(k);
        kv[l] = .{ .k = k, .v = try lf.copyBf16(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.kv.{d}.v", .{ ui, l }), kv_rows * arch.head_dim) };
        kv_done += 1;
    }

    const rec: []RecLayerData = if (rk == 0 or arch.num_rec == 0)
        try gpa.alloc(RecLayerData, 0)
    else blk: {
        const arr = try gpa.alloc(RecLayerData, arch.num_rec);
        var rc_done: usize = 0;
        errdefer {
            for (arr[0..rc_done]) |*r| r.deinit(gpa);
            gpa.free(arr);
        }
        const rk_u: usize = rk;
        for (0..arch.num_rec) |r| {
            const fu = try lf.copyF32(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.rc.{d}.u", .{ ui, r }), arch.rec_heads * arch.rec_rows * rk_u);
            errdefer gpa.free(fu);
            const fv = try lf.copyF32(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.rc.{d}.v", .{ ui, r }), arch.rec_heads * rk_u * arch.rec_cols);
            errdefer gpa.free(fv);
            const cv = if (has_conv)
                try lf.copyBf16(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.rc.{d}.cv", .{ ui, r }), arch.conv_elems)
            else
                try gpa.alloc(u16, 0);
            errdefer gpa.free(cv);
            arr[r] = .{ .rank = rk, .u = fu, .v = fv, .conv = cv };
            rc_done += 1;
        }
        break :blk arr;
    };
    errdefer {
        for (rec) |*r| r.deinit(gpa);
        gpa.free(rec);
    }

    var ug = gate.UnitGate{
        .dim = arch.hidden,
        .lex_protos = gl_v,
        .lex_weights = gl_w,
        .n_lex = n_lex,
        .n_body_lex = n_body_lex,
        .sem_protos = gs_v,
        .sem_weights = gs_w,
        .n_sem = n_sem,
        .neg_sem_max = neg_sem,
        .is_small_fact = is_sm,
        .is_standing = is_st,
        .is_digest = is_dg,
        .epoch = epoch,
        .superseded_by = sup,
        .reinforcements = rf,
    };
    // its four arrays stay this function's cleanups' to free: a failure frees only the summaries the gate made
    ug.computeSummaries(gpa) catch |e| {
        if (ug.lex_summary.len > 0) gpa.free(ug.lex_summary);
        if (ug.sem_summary.len > 0) gpa.free(ug.sem_summary);
        return e;
    };

    return .{
        .unit_id = uid,
        .parent_id = pid,
        .n_tokens = n_tok,
        .gate = ug,
        .kv = kv,
        .rec = rec,
    };
}

/// Deletes `path`; already absent is done, any other failure is the caller's error.
pub fn removeIfPresent(io: Io, path: []const u8) !void {
    Io.Dir.cwd().deleteFile(io, path) catch |e| if (e != error.FileNotFound) return e;
}

/// A saved whitener read whole; on failure nothing it read is kept, on success the caller owns every array.
pub fn loadWhitener(lf: *const store.LoadedFile, gpa: Allocator, comptime name: []const u8, k: usize, n_cal: usize, floor: f32, hidden: usize) !whitener.LayerWhitener {
    const cal = try lf.copyF32(gpa, name ++ ".cal", n_cal * hidden);
    errdefer gpa.free(cal);
    const mean = try lf.copyF32(gpa, name ++ ".mean", hidden);
    errdefer gpa.free(mean);
    const basis = try lf.copyF32(gpa, name ++ ".basis", k * hidden);
    return .{ .dim = hidden, .k_pcs = k, .n_cal = n_cal, .mean = mean, .basis = basis, .cal_tokens = cal, .norm_floor = floor };
}

pub fn parseUsizeMeta(lf: *const store.LoadedFile, key: []const u8) !usize {
    const s = lf.getMeta(key) orelse return error.MissingMetadata;
    return std.fmt.parseInt(usize, s, 10) catch return error.BadLivingWeights;
}

/// An F32 saved as its bit pattern, so a reload reads back exactly the value the bank computed.
pub fn parseBitsMeta(lf: *const store.LoadedFile, key: []const u8) !f32 {
    const s = lf.getMeta(key) orelse return error.MissingMetadata;
    return @bitCast(std.fmt.parseInt(u32, s, 10) catch return error.BadLivingWeights);
}

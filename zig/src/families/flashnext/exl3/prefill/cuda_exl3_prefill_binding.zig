const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const parts = @import("../../cuda_batch_state.zig");
const node_query = @import("../kv/cuda_exl3_state_binding.zig");

pub const Kind = enum { prep, scores, chunks, plan, remap };
pub const Functions = struct { prep: usize, scores: usize, chunks: usize, ahead: usize, plan: usize, remap: usize };
pub const Request = struct {
    functions: Functions,
    rows: usize,
    sub: usize,
    pos0: usize,
    positions: u64,
    cap: usize,
    origin: usize,
    page_rows: usize,
    shared: bool,
    /// the indexer scores ran on the 8-row, 256-block tile (four index heads), not the one-row kernel
    tile: bool = false,
    generation: u64,
    keys: u64,
    values: u64,

    pub fn validate(self: Request) !void {
        if (self.rows == 0 or self.sub == 0 or self.sub % parts.att_rows != 0 or
            self.rows > 65536 or self.positions == 0 or self.page_rows == 0 or self.page_rows > parts.att_rows or
            self.origin % wt.index_ratio != 0 or (self.shared and self.generation == 0) or self.origin > self.pos0 or self.pos0 > self.cap or self.rows > self.cap - self.pos0 or
            self.cap > std.math.maxInt(c_int) or self.keys == 0 or self.values == 0) return error.UnsupportedPrefillBinding;
        comptime var functions = std.meta.fieldNames(Functions);
        inline for (functions, 0..) |field, index| {
            if (@field(self.functions, field) == 0) return error.UnsupportedPrefillBinding;
            inline for (functions[0..index]) |previous| {
                if (@field(self.functions, field) == @field(self.functions, previous)) return error.UnsupportedPrefillBinding;
            }
        }
    }

    pub fn end(self: Request, block: usize) usize {
        return self.pos0 + @min(self.rows, (block + 1) * parts.att_rows);
    }

    pub fn sparse(self: Request, block: usize) bool {
        return self.end(block) / wt.index_ratio > wt.index_top;
    }

    pub fn compatible(self: Request, target: Request) bool {
        target.validate() catch return false;
        var before = self;
        var after = target;
        before.pos0 = 0;
        after.pos0 = 0;
        before.origin = 0;
        after.origin = 0;
        before.generation = 0;
        after.generation = 0;
        if (!std.meta.eql(before, after)) return false;
        for (0..divUp(self.rows, parts.att_rows)) |block| {
            if (self.sparse(block) != target.sparse(block)) return false;
        }
        return true;
    }

    pub fn chunks(self: Request, block: usize) u32 {
        return @intCast(divUp(@min(self.end(block), wt.dense_keys), 512));
    }

    pub fn scores(self: Request, block: usize) u32 {
        return @intCast(divUp(@min(parts.scoreStride(self.cap), divUp(self.end(block), wt.index_ratio)), if (self.tile) 256 else 64));
    }
};

fn divUp(value: usize, divisor: usize) usize {
    return (value + divisor - 1) / divisor;
}

pub fn schema(kind: Kind, shared: bool) []const u8 {
    return switch (kind) {
        .prep => &.{ 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4 },
        .scores => &.{ 8, 8, 8, 8, 4, 4, 4, 4 },
        .chunks => &.{ 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4 },
        .plan => if (shared)
            &.{ 8, 8, 8, 4, 4, 4, 4, 4, 8, 8, 8, 4, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 4, 4 }
        else
            &.{ 8, 8, 8, 4, 4, 4, 4, 4, 8, 8, 8, 4, 8, 8, 8, 4, 8, 8, 8, 8 },
        .remap => &.{ 8, 8, 8, 4, 4, 4, 8, 4, 8 },
    };
}

pub const Record = struct {
    node: cuda.graph.Node,
    params: cuda.abi.KernelNodeParams,
    kind: Kind,
    block: usize = 0,
    values: [25]u64 = @splat(0),

    pub fn read(node: cuda.graph.Node, params: cuda.abi.KernelNodeParams, kind: Kind, request: Request) !Record {
        if (params.func == null or params.extra != null or params.params == null) return error.UnsupportedPrefillBinding;
        var record: Record = .{ .node = node, .params = params, .kind = kind };
        record.params.params = null;
        for (schema(kind, request.shared), 0..) |size, index| {
            const source = params.params.?[index] orelse return error.UnsupportedPrefillBinding;
            @memcpy(std.mem.asBytes(&record.values[index])[0..size], @as([*]const u8, @ptrCast(source))[0..size]);
        }
        const position_index: ?usize = switch (kind) {
            .prep => 1,
            .scores => 2,
            .chunks => 3,
            else => null,
        };
        if (position_index) |index| {
            const position = record.values[index];
            if (position < request.positions or (position - request.positions) % 16 != 0) return error.UnsupportedPrefillBinding;
            record.block = @intCast((position - request.positions) / 16);
            if (record.block >= divUp(request.rows, parts.att_rows)) return error.UnsupportedPrefillBinding;
        }
        const block_rows = @min(parts.att_rows, request.rows - record.block * parts.att_rows);
        switch (kind) {
            .prep => {
                const offset = record.block * parts.att_rows;
                if (offset % request.sub != 0 or params.grid_x != @min(request.sub, request.rows - offset) or
                    params.grid_y != wt.q_heads + wt.kv_heads + wt.index_heads + 1 or params.grid_z != 1 or params.block_x != 256 or
                    record.values[7] != request.keys -% (request.origin * row_bytes) or
                    record.values[8] != request.values -% (request.origin * row_bytes)) return error.UnsupportedPrefillBinding;
            },
            .scores => {
                if (!request.sparse(record.block) or params.grid_x != (if (request.tile) divUp(block_rows, 8) else block_rows) or params.grid_y != request.scores(record.block) or params.grid_z != 1 or params.block_x != 256 or
                    record.values[4] != parts.scoreStride(request.cap)) return error.UnsupportedPrefillBinding;
            },
            .chunks => {
                if (params.grid_z != request.chunks(record.block) or params.grid_y != wt.kv_heads or params.block_x != 128 or
                    params.grid_x == 0 or params.grid_x > @min(request.page_rows, block_rows)) return error.UnsupportedPrefillBinding;
            },
            .plan => if (record.values[7] != request.origin or (request.shared and record.values[22] != request.generation) or record.values[3] == 0 or record.values[3] > request.page_rows or params.grid_x != 1 or params.block_x != 1024) return error.UnsupportedPrefillBinding,
            .remap => if (record.values[4] != request.origin or params.grid_x == 0 or params.grid_x > request.page_rows or params.block_x != 256) return error.UnsupportedPrefillBinding,
        }
        if (params.block_y != 1 or params.block_z != 1) return error.UnsupportedPrefillBinding;
        return record;
    }

    pub fn changed(self: Record, before: Request, target: Request) bool {
        return switch (self.kind) {
            .prep, .remap => before.origin != target.origin,
            .plan => before.origin != target.origin or before.generation != target.generation,
            .scores => before.scores(self.block) != target.scores(self.block),
            .chunks => before.chunks(self.block) != target.chunks(self.block),
        };
    }

    pub fn rebound(self: Record, target: Request) Record {
        var record = self;
        switch (record.kind) {
            .prep => {
                record.values[7] = target.keys -% (target.origin * row_bytes);
                record.values[8] = target.values -% (target.origin * row_bytes);
            },
            .scores => record.params.grid_y = target.scores(record.block),
            .chunks => record.params.grid_z = target.chunks(record.block),
            .plan => {
                record.values[7] = target.origin;
                if (target.shared) record.values[22] = target.generation;
            },
            .remap => record.values[4] = target.origin,
        }
        return record;
    }
};

const row_bytes = wt.kv_heads * wt.head_dim * 2;

pub fn requestFor(pass: anytype, linear: bool, layer: usize, rows: usize, sub: usize, pos0: usize, positions: u64) ?Request {
    if (linear or pass.tri.set != null or pass.seq.media != null or pass.seq.kv8 != null or pass.lanes.len != 0) return null;
    const paging = @import("../kv/cuda_exl3_kvpage.zig");
    if (paging.trace != null or @import("../model/cuda_exl3_heads.zig").isSplit(&pass.m.layers[layer])) return null;
    const pg = pass.seq.paged orelse return null;
    const tile = wt.index_heads == 4;
    const glue = pass.tri.own orelse return null;
    const index = wt.attnIndex(layer);
    const plan = if (pg.cfg.shared_lanes == 0) pg.k.plan else (pg.shared_arena.?.kernels orelse return null).plan;
    const remap = if (pg.cfg.shared_lanes == 0) pg.k.remap else (pg.shared_arena.?.kernels orelse return null).remap;
    const request: Request = .{
        .functions = .{ .prep = @intFromPtr(glue.f.prep.handle), .scores = @intFromPtr((if (tile) glue.f.scores_tile else glue.f.scores).handle), .chunks = @intFromPtr(glue.f.chunks.handle), .ahead = @intFromPtr(glue.f.chunks_ahead.handle), .plan = @intFromPtr(plan.handle), .remap = @intFromPtr(remap.handle) },
        .rows = rows,
        .sub = sub,
        .pos0 = pos0,
        .positions = positions,
        .cap = pass.seq.cap,
        .origin = pg.win0[index],
        .page_rows = paging.pagingRows(pg.cfg, pg.win0[index], parts.att_rows) catch return null,
        .shared = pg.cfg.shared_lanes != 0,
        .tile = tile,
        .generation = if (pg.cfg.shared_lanes != 0) pg.shared_sequence.generation else 0,
        .keys = pg.devRows(index, false),
        .values = pg.devRows(index, true),
    };
    request.validate() catch return null;
    return request;
}

pub const Binding = struct {
    graph: cuda.graph.Graph,
    request: Request,
    records: []Record,
    poisoned: bool = false,

    pub fn create(allocator: std.mem.Allocator, graph: cuda.graph.Graph, request: Request) !*Binding {
        return createWithNodeType(allocator, graph, request, node_query.nodeType(graph.d));
    }

    pub fn createWithNodeType(allocator: std.mem.Allocator, graph: cuda.graph.Graph, request: Request, node_type: ?node_query.NodeType) !*Binding {
        try request.validate();
        const query = node_type orelse return error.UnsupportedPrefillBinding;
        const count = try graph.nodeCount();
        const nodes = try allocator.alloc(cuda.graph.Node, count);
        defer allocator.free(nodes);
        const actual = try graph.nodes(nodes);
        if (actual.len != count) return error.UnsupportedPrefillBinding;
        var records: std.array_list.Managed(Record) = .init(allocator);
        defer records.deinit();
        for (actual) |node| {
            var kind_id: c_uint = undefined;
            try graph.d.check(query(node, &kind_id), "cuGraphNodeGetType(prefill binding)");
            if (kind_id == 4) return error.UnsupportedPrefillBinding;
            if (kind_id != 0) continue;
            var params: cuda.abi.KernelNodeParams = undefined;
            try graph.d.check(graph.d.api.cuGraphKernelNodeGetParams_v2(node, &params), "cuGraphKernelNodeGetParams(prefill binding)");
            const function = @intFromPtr(params.func);
            const functions = request.functions;
            const kind: Kind = if (function == functions.prep) .prep else if (function == functions.scores) .scores else if (function == functions.chunks or function == functions.ahead) .chunks else if (function == functions.plan) .plan else if (function == functions.remap) .remap else continue;
            try records.append(try Record.read(node, params, kind, request));
        }
        try validateRecords(request, records.items);
        const binding = try allocator.create(Binding);
        errdefer allocator.destroy(binding);
        binding.* = .{ .graph = graph, .request = request, .records = try records.toOwnedSlice() };
        return binding;
    }

    pub fn deinit(self: *Binding, allocator: std.mem.Allocator) void {
        self.graph.deinit();
        allocator.free(self.records);
        allocator.destroy(self);
    }

    pub fn apply(self: *Binding, backend: anytype, target: Request) !void {
        if (self.poisoned or !self.request.compatible(target)) return error.UnsupportedPrefillBinding;
        errdefer self.poisoned = true;
        for (self.records) |*record| {
            if (!record.changed(self.request, target)) continue;
            const next = record.rebound(target);
            try backend.set(next, target.shared);
            record.* = next;
        }
        self.request = target;
    }
};

pub fn validateRecords(request: Request, records: []const Record) !void {
    var preps: [256]usize = @splat(0);
    var scores: [256]usize = @splat(0);
    var chunks: [256]usize = @splat(0);
    var chunk_rows: [256]usize = @splat(0);
    var plans: usize = 0;
    var remaps: usize = 0;
    var plan_rows: usize = 0;
    var remap_rows: usize = 0;
    for (records) |record| switch (record.kind) {
        .prep => preps[record.block] += 1,
        .scores => scores[record.block] += 1,
        .chunks => {
            chunks[record.block] += 1;
            chunk_rows[record.block] += record.params.grid_x;
        },
        .plan => {
            plans += 1;
            plan_rows += @intCast(record.values[3]);
        },
        .remap => {
            remaps += 1;
            remap_rows += record.params.grid_x;
        },
    };
    var expected_pages: usize = 0;
    var expected_rows: usize = 0;
    for (0..divUp(request.rows, parts.att_rows)) |block| {
        const rows = @min(parts.att_rows, request.rows - block * parts.att_rows);
        const pages = divUp(rows, request.page_rows);
        const sparse = request.sparse(block);
        if (preps[block] != @intFromBool((block * parts.att_rows) % request.sub == 0) or
            scores[block] != @intFromBool(sparse) or chunks[block] != pages or chunk_rows[block] != rows) return error.UnsupportedPrefillBinding;
        if (sparse) {
            expected_pages += pages;
            expected_rows += rows;
        }
    }
    if (plans != expected_pages or remaps != expected_pages or plan_rows != expected_rows or remap_rows != expected_rows) return error.UnsupportedPrefillBinding;
}

pub const Backend = struct {
    exec: cuda.graph.Exec,

    pub fn set(self: Backend, record: Record, shared: bool) !void {
        var args: cuda.Args = .{};
        for (schema(record.kind, shared), 0..) |size, index| {
            if (size == 8) args.add(record.values[index]) else args.add(@as(u32, @truncate(record.values[index])));
        }
        const params = record.params;
        try self.exec.setKernel(record.node, .{ .d = self.exec.d, .handle = params.func }, .{
            .grid = .{ .x = params.grid_x, .y = params.grid_y, .z = params.grid_z },
            .block = .{ .x = params.block_x, .y = params.block_y, .z = params.block_z },
            .shared = params.shared_bytes,
        }, &args);
    }
};

test {
    _ = @import("cuda_exl3_prefill_binding_test.zig");
}

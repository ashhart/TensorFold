//! LFM2's attention on prefill_attention.cu's flash kernels: the trunk and its branches, and the vision tower's tiles.
const cuda = @import("cuda");

/// tree_kernel<64, 8, 4, 8>: eight warps as four query heads by two 16-row tiles.
const Tree = struct {
    const sym = "_ZN20tf_prefill_attention11tree_kernelILi64ELi8ELi4ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_PKiiiiiif";
    const dim = 64;
    const heads_per_block = 4;
    const rows_per_block = 32;
    const smem: u32 = 32768; // eight 32-key slots of 64 dims
};

/// full_kernel<72, 8, 8>: eight warps as eight 16-row tiles of one head.
const Full = struct {
    const sym = "_ZN20tf_prefill_attention11full_kernelILi72ELi8ELi8EEEvPK13__nv_bfloat16PS1_iif";
    const dim = 72;
    const rows_per_block = 128;
    const smem: u32 = 65536; // eight 32-key slots of 72 dims, rows padded to 128
};

const threads = 256;

pub const Attention = struct {
    mod: cuda.Module,
    tree_kernel: cuda.Function,
    full_kernel: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Attention {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.prefill_attention);
        errdefer mod.unload();
        const a: Attention = .{ .mod = mod, .tree_kernel = try mod.function(Tree.sym), .full_kernel = try mod.function(Full.sym) };
        try a.full_kernel.allowDynamicShared(Full.smem);
        return a;
    }

    pub fn unload(a: *Attention) void {
        a.mod.unload();
    }

    /// Causal GQA over one sequence: q [rows, H, D], k/v [rows, KV, D] bf16 -> out [rows, H, D] bf16.
    pub fn causal(a: *const Attention, s: cuda.Stream, q: u64, k: u64, v: u64, out: u64, rows: usize, heads: usize, kv_heads: usize, dim: usize, scale: f32) !void {
        return a.tree(s, q, k, v, out, rows, 0, 0, rows, heads, kv_heads, dim, scale);
    }

    /// As causal over packed rows: a trunk, then branches (rows trunk + cu[b]..) over it and their own earlier rows.
    pub fn tree(a: *const Attention, s: cuda.Stream, q: u64, k: u64, v: u64, out: u64, trunk: usize, cu: u64, branches: usize, rows: usize, heads: usize, kv_heads: usize, dim: usize, scale: f32) !void {
        const group = heads / kv_heads;
        if (dim != Tree.dim or group * kv_heads != heads or group % Tree.heads_per_block != 0) return error.UnsupportedShape;
        if (rows == 0) return;
        var args: cuda.Args = .{};
        for ([_]u64{ q, k, v, out, cu }) |p| args.add(p);
        for ([_]usize{ trunk, branches, heads, kv_heads, group }) |n| args.add(@as(c_int, @intCast(n)));
        args.add(scale);
        // the trunk's row tiles, then the branches': their rows' worth plus at most one partial tile each
        const per = Tree.rows_per_block;
        const blocks = (trunk + per - 1) / per + (rows - trunk + branches * (per - 1)) / per;
        try go(a.tree_kernel, s, blocks, heads / Tree.heads_per_block, Tree.smem, &args);
    }

    /// Every row over every key, a head at a time: q, k, v heads of qkv [rows, 3 H D] bf16 -> out [rows, H D] bf16.
    pub fn full(a: *const Attention, s: cuda.Stream, qkv: u64, out: u64, rows: usize, heads: usize, dim: usize, scale: f32) !void {
        if (dim != Full.dim) return error.UnsupportedShape;
        if (rows == 0) return;
        var args: cuda.Args = .{};
        args.add(qkv);
        args.add(out);
        for ([_]usize{ rows, heads }) |n| args.add(@as(c_int, @intCast(n)));
        args.add(scale);
        try go(a.full_kernel, s, (rows + Full.rows_per_block - 1) / Full.rows_per_block, heads, Full.smem, &args);
    }
};

/// `row_blocks` blocks of rows by `head_blocks` blocks of heads.
fn go(f: cuda.Function, s: cuda.Stream, row_blocks: usize, head_blocks: usize, smem: u32, args: *cuda.Args) !void {
    const grid: cuda.launch.Dim3 = .{ .x = @intCast(row_blocks), .y = @intCast(head_blocks) };
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads }, .shared = smem }, s, args);
}

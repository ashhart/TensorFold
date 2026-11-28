/// Scratch bytes for a slice-reduction plan: fp32 partials shaped [sk][m][n].
pub fn required(m: usize, n: usize, sk: u32, reduce: bool) usize {
    return if (reduce) @as(usize, sk) * m * n * 4 else 0;
}

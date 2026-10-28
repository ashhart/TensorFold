// A 64-key tile's rows of kv head hk as a tensor over its page: rows to L, at most to the page's end.
template <int D>
inline tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> nemo_page_tensor(const device bfloat16_t* base, device const uint* table,
                                                                                   ulong page_bytes, int kt, int L, uint hk) {
  const int rows = min(TF_PAGE_TOKENS - kt % TF_PAGE_TOKENS, L - kt);
  device bfloat* p = (device bfloat*)((const device uchar*)base + nemo_paged_row(table, uint(kt), page_bytes, hk, D));
  return tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(p, dextents<int32_t, 2>(D, rows), array<int32_t, 2>({1, D}));
}

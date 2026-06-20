from __future__ import annotations

import unittest

from smarttensor.weight_pager import PagedWeightCache, WeightPageKey, row_page_specs


class PagedWeightCacheTests(unittest.TestCase):
    def test_lru_evicts_oldest_unprotected_page(self) -> None:
        cache = PagedWeightCache(10)
        k1 = WeightPageKey("a", 0, 1)
        k2 = WeightPageKey("b", 0, 1)
        k3 = WeightPageKey("c", 0, 1)

        cache.put(k1, "a0", nbytes=4)
        cache.put(k2, "b0", nbytes=4)
        self.assertEqual(cache.get_or_load(k1, nbytes=4, loader=lambda: "miss"), "a0")
        cache.put(k3, "c0", nbytes=4)

        self.assertTrue(cache.contains(k1))
        self.assertFalse(cache.contains(k2))
        self.assertTrue(cache.contains(k3))
        self.assertEqual(cache.stats.evictions, 1)
        self.assertEqual(cache.resident_bytes, 8)

    def test_active_lease_blocks_eviction_until_released(self) -> None:
        cache = PagedWeightCache(4)
        k1 = WeightPageKey("a", 0, 1)
        k2 = WeightPageKey("b", 0, 1)
        cache.put(k1, "a0", nbytes=4)

        with cache.lease(k1, nbytes=4, loader=lambda: "unused"):
            with self.assertRaises(MemoryError):
                cache.put(k2, "b0", nbytes=4)
            self.assertTrue(cache.contains(k1))

        cache.put(k2, "b0", nbytes=4)
        self.assertFalse(cache.contains(k1))
        self.assertTrue(cache.contains(k2))

    def test_active_lease_blocks_same_key_replacement(self) -> None:
        cache = PagedWeightCache(4)
        key = WeightPageKey("a", 0, 1)
        cache.put(key, "original", nbytes=4)

        with cache.lease(key, nbytes=4, loader=lambda: "unused"):
            with self.assertRaises(MemoryError):
                cache.put(key, "replacement", nbytes=4)

        self.assertEqual(
            cache.get_or_load(key, nbytes=4, loader=lambda: "miss"),
            "original",
        )

    def test_failed_same_key_replacement_keeps_original_page_and_accounting(self) -> None:
        cache = PagedWeightCache(10)
        pinned = WeightPageKey("p", 0, 1)
        key = WeightPageKey("x", 0, 1)
        cache.put(pinned, "pinned", nbytes=6, pin=True)
        cache.put(key, "old", nbytes=4)

        with self.assertRaises(MemoryError):
            cache.put(key, "new", nbytes=5)

        self.assertTrue(cache.contains(key))
        self.assertEqual(
            cache.get_or_load(key, nbytes=4, loader=lambda: "miss"),
            "old",
        )
        self.assertEqual(cache.resident_bytes, 10)
        self.assertEqual(cache.stats.bytes_evicted, 0)

    def test_pin_protects_page_until_unpinned(self) -> None:
        cache = PagedWeightCache(8)
        k1 = WeightPageKey("a", 0, 1)
        k2 = WeightPageKey("b", 0, 1)
        k3 = WeightPageKey("c", 0, 1)
        k4 = WeightPageKey("d", 0, 1)

        cache.put(k1, "a0", nbytes=4, pin=True)
        cache.put(k2, "b0", nbytes=4)
        cache.put(k3, "c0", nbytes=4)
        self.assertTrue(cache.contains(k1))
        self.assertFalse(cache.contains(k2))

        cache.unpin(k1)
        cache.put(k4, "d0", nbytes=4)
        self.assertFalse(cache.contains(k1))
        self.assertTrue(cache.contains(k3))
        self.assertTrue(cache.contains(k4))

    def test_frequency_policy_never_evicts_protected_pages(self) -> None:
        cache = PagedWeightCache(12, eviction_policy="frequency")
        pinned = WeightPageKey("pinned", 0, 1)
        leased = WeightPageKey("leased", 0, 1)
        cold = WeightPageKey("cold", 0, 1)
        incoming = WeightPageKey("incoming", 0, 1)

        cache.put(pinned, "pinned", nbytes=4, pin=True)
        cache.put(leased, "leased", nbytes=4)
        cache.put(cold, "cold", nbytes=4)

        with cache.lease(leased, nbytes=4, loader=lambda: "unused"):
            cache.put(incoming, "incoming", nbytes=4)

            self.assertTrue(cache.contains(pinned))
            self.assertTrue(cache.contains(leased))
            self.assertFalse(cache.contains(cold))
            self.assertTrue(cache.contains(incoming))
            self.assertEqual(cache.resident_bytes, 12)

    def test_frequency_policy_retains_hot_pages_better_than_lru_scan(self) -> None:
        hot_keys = tuple(WeightPageKey(f"hot.{index}", 0, 1) for index in range(2))
        scan_keys = tuple(WeightPageKey(f"scan.{index}", 0, 1) for index in range(6))

        def access(cache: PagedWeightCache, key: WeightPageKey) -> None:
            cache.get_or_load(key, nbytes=1, loader=lambda: key.tensor_name)

        def run_pattern(cache: PagedWeightCache) -> PagedWeightCache:
            for _ in range(3):
                for key in hot_keys:
                    access(cache, key)

            for _ in range(4):
                for key in scan_keys:
                    access(cache, key)
                for key in hot_keys:
                    access(cache, key)

            for key in scan_keys:
                access(cache, key)
            return cache

        lru_cache = run_pattern(PagedWeightCache(3))
        frequency_cache = run_pattern(
            PagedWeightCache(3, eviction_policy="frequency")
        )

        self.assertGreater(frequency_cache.stats.hits, lru_cache.stats.hits)
        self.assertTrue(all(frequency_cache.contains(key) for key in hot_keys))
        self.assertFalse(any(lru_cache.contains(key) for key in hot_keys))
        self.assertGreater(frequency_cache.stats.admission_rejections, 0)
        self.assertGreater(frequency_cache.stats.bytes_read, frequency_cache.stats.bytes_loaded)

    def test_two_queue_policy_keeps_a_probationary_scan_window(self) -> None:
        cache = PagedWeightCache(3, eviction_policy="two_queue")
        hot_keys = tuple(WeightPageKey(f"hot.{index}", 0, 1) for index in range(2))
        scan_keys = tuple(WeightPageKey(f"scan.{index}", 0, 1) for index in range(4))

        def access(key: WeightPageKey) -> None:
            cache.get_or_load(key, nbytes=1, loader=lambda: key.tensor_name)

        for _ in range(2):
            for key in hot_keys:
                access(key)

        for key in scan_keys:
            access(key)

        self.assertTrue(all(cache.contains(key) for key in hot_keys))
        self.assertTrue(cache.contains(scan_keys[-1]))
        self.assertFalse(any(cache.contains(key) for key in scan_keys[:-1]))
        self.assertGreater(cache.stats.evictions, 0)
        self.assertEqual(cache.stats.admission_rejections, 0)

    def test_two_queue_policy_rejects_cold_page_when_only_hot_pages_fit(self) -> None:
        cache = PagedWeightCache(2, eviction_policy="two_queue")
        hot_keys = tuple(WeightPageKey(f"hot.{index}", 0, 1) for index in range(2))
        cold_key = WeightPageKey("cold", 0, 1)

        for _ in range(2):
            for key in hot_keys:
                cache.get_or_load(key, nbytes=1, loader=lambda: key.tensor_name)

        cache.get_or_load(cold_key, nbytes=1, loader=lambda: cold_key.tensor_name)

        self.assertTrue(all(cache.contains(key) for key in hot_keys))
        self.assertFalse(cache.contains(cold_key))
        self.assertEqual(cache.stats.admission_rejections, 1)

    def test_row_page_specs_group_requested_rows(self) -> None:
        specs = row_page_specs(
            "experts.weight",
            shape=(5, 3, 2),
            tensor_nbytes=5 * 3 * 2 * 4,
            indices=(0, 1, 4),
            rows_per_page=2,
        )

        self.assertEqual(
            [spec.key for spec in specs],
            [
                WeightPageKey("experts.weight", 0, 2),
                WeightPageKey("experts.weight", 4, 5),
            ],
        )
        self.assertEqual([spec.nbytes for spec in specs], [48, 24])
        self.assertEqual([spec.row_indices for spec in specs], [(0, 1), (4,)])

    def test_row_page_specs_allows_empty_request(self) -> None:
        specs = row_page_specs(
            "experts.weight",
            shape=(5, 3, 2),
            tensor_nbytes=5 * 3 * 2 * 4,
            indices=(),
            rows_per_page=2,
        )

        self.assertEqual(specs, ())

    def test_row_page_specs_reject_out_of_bounds_rows(self) -> None:
        with self.assertRaises(IndexError):
            row_page_specs(
                "experts.weight",
                shape=(2, 3),
                tensor_nbytes=24,
                indices=(2,),
            )


if __name__ == "__main__":
    unittest.main()

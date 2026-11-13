from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from tests.fixtures import RasterCase
import stac_utils as stac


class StacTests(RasterCase):
    def setUp(self):
        super().setUp()
        stac.clear_search_cache()
        self.client = MagicMock()
        self.item = self.scene({"red": 1000})

    def search(self, **kwargs):
        options = dict(
            bbox=[0, 0, 1, 1],
            start_date="2025-01-01",
            end_date="2025-02-01",
            collection="sentinel-2-l2a",
            return_report=True,
            use_cache=False,
        )
        options.update(kwargs)
        with patch.object(stac, "get_stac_client", return_value=self.client):
            return stac.search_stac(**options)

    def test_default_paginates_without_cap(self):
        items = [self.item.clone() for _ in range(120)]
        for i, item in enumerate(items):
            item.id = str(i)
        self.client.search.return_value.items.return_value = iter(items)
        result = self.search()
        self.assertEqual(len(result.items), 120)
        self.assertTrue(result.complete)
        self.assertIsNone(self.client.search.call_args.kwargs["max_items"])

    def test_limit_detected_by_one_extra_result(self):
        second = self.item.clone()
        second.id = "second"
        self.client.search.return_value.items.return_value = iter([self.item, second])
        result = self.search(max_items=1)
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(self.client.search.call_args.kwargs["max_items"], 2)

    def test_exact_limit_is_complete(self):
        self.client.search.return_value.items.return_value = iter([self.item])
        self.assertTrue(self.search(max_items=1).complete)

    def test_zero_cloud_inclusive_and_100_unfiltered(self):
        self.client.search.return_value.items.return_value = []
        self.search(max_cloud_cover=0)
        self.assertEqual(self.client.search.call_args.kwargs["query"], {"eo:cloud_cover": {"lte": 0}})
        self.search(max_cloud_cover=100)
        self.assertEqual(self.client.search.call_args.kwargs["query"], {})

    def test_cache_returns_independent_items_and_expires(self):
        self.client.search.return_value.items.return_value = [self.item]
        first = self.search(use_cache=True)
        first.items[0].id = "mutated"
        cached = self.search(use_cache=True)
        self.assertTrue(cached.cached)
        self.assertNotEqual(cached.items[0].id, "mutated")
        self.assertEqual(self.client.search.call_count, 1)
        with patch.object(stac, "CACHE_TTL_SECONDS", 0):
            self.search(use_cache=True)
        self.assertEqual(self.client.search.call_count, 2)

    def test_search_failure_not_cached(self):
        self.client.search.side_effect = TimeoutError("timeout")
        with self.assertRaises(stac.CatalogSearchError):
            self.search(use_cache=True)
        self.assertEqual(len(stac._CACHE), 0)

    def test_configuration_changes_fingerprint(self):
        original = stac.search_fingerprint([0, 0, 1, 1], "2025-01-01", "2025-02-01", "sentinel-2-l2a", 20)
        changes = [
            ([0, 0, 2, 2], "2025-01-01", "2025-02-01", "sentinel-2-l2a", 20),
            ([0, 0, 1, 1], "2025-01-02", "2025-02-01", "sentinel-2-l2a", 20),
            ([0, 0, 1, 1], "2025-01-01", "2025-02-01", "sentinel-1-grd", 20),
            ([0, 0, 1, 1], "2025-01-01", "2025-02-01", "sentinel-2-l2a", 30),
        ]
        for args in changes:
            self.assertNotEqual(original, stac.search_fingerprint(*args))

    def test_timeout_and_transient_retries_configured(self):
        with (
            patch.object(stac, "_CLIENTS", SimpleNamespace()),
            patch.object(stac.pystac_client.Client, "open") as opened,
        ):
            stac.get_stac_client()
            io = opened.call_args.kwargs["stac_io"]
            self.assertEqual(io.timeout, (10, 45))
            retry = io.session.get_adapter("https://").max_retries
            self.assertIn(429, retry.status_forcelist)
            self.assertIn("POST", retry.allowed_methods)

    def test_invalid_inputs_fail_before_network(self):
        for changes in [dict(bbox=[1, 0, 0, 1]), dict(start_date="2026-01-01"), dict(max_items=0)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.search(**changes)
        self.client.search.assert_not_called()

    def test_cache_byte_budget_and_cached_truncation_warning(self):
        self.client.search.return_value.items.return_value = [self.item]
        with patch.object(stac, "CACHE_MAX_BYTES", 1):
            self.search(use_cache=True)
        self.assertEqual(len(stac._CACHE), 0)
        second = self.item.clone()
        second.id = "second"
        self.client.search.return_value.items.return_value = [self.item, second]
        self.search(max_items=1, use_cache=True)
        with (
            patch.object(stac, "get_stac_client", return_value=self.client),
            self.assertWarns(RuntimeWarning),
        ):
            stac.search_stac(
                [0, 0, 1, 1], "2025-01-01", "2025-02-01", "sentinel-2-l2a", max_items=1, use_cache=True
            )

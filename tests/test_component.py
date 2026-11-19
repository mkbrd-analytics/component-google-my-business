"""
Created on 12. 11. 2018

@author: esner
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from freezegun import freeze_time
from keboola.component.exceptions import UserException

from component import Component
from google_my_business import (
    AVAILABLE_DAILY_METRICS,
    GoogleMyBusiness,
    GoogleMyBusinessException,
    GoogleMyBusinessRequestError,
)


class TestComponent(unittest.TestCase):
    # set global time to 2010-10-10 - affects functions like datetime.now()
    @freeze_time("2010-10-10")
    # set KBC_DATADIR env to non-existing dir
    @mock.patch.dict(os.environ, {"KBC_DATADIR": "./non-existing-dir"})
    def test_run_no_cfg_fails(self):
        with self.assertRaises(ValueError):
            comp = Component()
            comp.run()


class TestGetRequestErrorHandling(unittest.TestCase):
    """Regression tests for the defensive fix around unexpected HTTP statuses.

    A 503 (or any other unexpected non-200 that the existing retry logic cannot
    recover from) used to escape ``get_request`` as a bare ``Exception``, which the
    component boundary did not catch, so the job died with an opaque internal error
    (exit 2). It is now raised as ``GoogleMyBusinessRequestError`` and surfaced by the
    component as a clear ``UserException`` (exit 1) with the same message.
    """

    @staticmethod
    def _make_client():
        return GoogleMyBusiness(access_token="tok", data_folder_path=tempfile.gettempdir())

    @mock.patch("time.sleep")  # neutralise the backoff waits so the test is fast
    def test_unexpected_status_raises_request_error_after_retries(self, _sleep):
        gmb = self._make_client()
        resp = mock.Mock(status_code=503, text="Service Unavailable")
        with mock.patch.object(gmb.session, "get", return_value=resp) as mock_get:
            with self.assertRaises(GoogleMyBusinessRequestError) as ctx:
                gmb.get_request("https://example.test/resource")

        # Same message as before the fix - only the exception *type* changed.
        self.assertEqual(str(ctx.exception), "Request failed with status code 503")
        # Inner backoff retry budget (max_tries=7) is preserved.
        self.assertEqual(mock_get.call_count, 7)

    def test_success_and_handled_statuses_are_unchanged(self):
        """Happy path and the explicitly-handled statuses must behave byte-for-byte as before."""
        gmb = self._make_client()

        ok = mock.Mock(status_code=200)
        with mock.patch.object(gmb.session, "get", return_value=ok):
            self.assertEqual(gmb.get_request("https://example.test/ok"), (200, ok))

        for handled in (400, 401, 403, 404, 500, 501):
            resp = mock.Mock(status_code=handled)
            with mock.patch.object(gmb.session, "get", return_value=resp):
                self.assertEqual(gmb.get_request("https://example.test/handled"), (handled, resp))

    def test_request_error_bypasses_endpoint_retry_decorators(self):
        """The new exception must not be swept up by the per-endpoint backoff decorators.

        ``list_locations`` / ``list_media`` retry on ``GoogleMyBusinessException``. The
        terminal request error must therefore be an ``Exception`` (so ``get_request``'s
        own ``backoff.on_exception(..., Exception, ...)`` still retries it) but NOT a
        ``GoogleMyBusinessException`` (so the outer decorators keep ignoring it exactly
        as they ignored the previous bare ``Exception``). This is what keeps the retry
        behaviour identical.
        """
        self.assertTrue(issubclass(GoogleMyBusinessRequestError, Exception))
        self.assertFalse(issubclass(GoogleMyBusinessRequestError, GoogleMyBusinessException))

    def test_component_run_converts_request_error_to_user_exception(self):
        """End-to-end: a terminal request error now exits 1 (UserException), not exit 2."""
        with tempfile.TemporaryDirectory() as datadir:
            os.makedirs(os.path.join(datadir, "in"))
            os.makedirs(os.path.join(datadir, "out", "tables"))
            with open(os.path.join(datadir, "in", "state.json"), "w") as f:
                json.dump({}, f)
            config = {
                "parameters": {
                    "#api_token": "x",
                    "endpoints": ["reviews"],
                    "accounts": {"label": "acc", "value": "accounts/1"},
                    "request_range": {"start_date": "7 days ago", "end_date": "today"},
                },
                "authorization": {"oauth_api": {"credentials": {}}},
                "action": "run",
            }
            with open(os.path.join(datadir, "config.json"), "w") as f:
                json.dump(config, f)

            with mock.patch.dict(os.environ, {"KBC_DATADIR": datadir}):
                comp = Component()
                with (
                    mock.patch.object(Component, "get_oauth_token", return_value="tok"),
                    mock.patch.object(
                        GoogleMyBusiness,
                        "process",
                        side_effect=GoogleMyBusinessRequestError("Request failed with status code 503"),
                    ),
                ):
                    with self.assertRaises(UserException) as ctx:
                        comp.run()

        self.assertIn("Request failed with status code 503", str(ctx.exception))


class TestFetchMultiDailyMetrics(unittest.TestCase):
    """``list_daily_metrics`` fetches every metric in one request.

    It calls fetchMultiDailyMetricsTimeSeries instead of getDailyMetricsTimeSeries once
    per metric. The return shape and the metric names must be unchanged, because the
    ``metric`` column downstream holds the API's enum names verbatim.
    """

    @staticmethod
    def _make_client():
        return GoogleMyBusiness(
            access_token="tok",
            data_folder_path=tempfile.gettempdir(),
            start_timestamp="2026-03-01T00:00:00.000000Z",
            end_timestamp="2026-03-02T00:00:00.000000Z",
        )

    @staticmethod
    def _series(metric, dated_values, sub_entity_type=None):
        series = {"dailyMetric": metric, "timeSeries": {"datedValues": dated_values}}
        if sub_entity_type is not None:
            series["dailySubEntityType"] = sub_entity_type
        return series

    def _run(self, payload, status=200):
        gmb = self._make_client()
        resp = mock.Mock(status_code=status, text="")
        resp.json.return_value = payload
        with mock.patch.object(gmb, "get_request", return_value=(status, resp)) as mock_request:
            result = gmb.list_daily_metrics("locations/12345")
        return result, mock_request

    def test_single_request_carries_every_metric_and_the_date_range(self):
        result, mock_request = self._run({"multiDailyMetricTimeSeries": []})

        # One request for the whole location, not one per metric.
        self.assertEqual(mock_request.call_count, 1)

        params = mock_request.call_args.kwargs["params"]
        self.assertEqual(params["dailyMetrics"], AVAILABLE_DAILY_METRICS)
        self.assertNotIn("dailyMetric", params)
        self.assertEqual(
            (params["dailyRange.startDate.year"], params["dailyRange.startDate.month"]),
            (2026, 3),
        )
        self.assertEqual(params["dailyRange.endDate.day"], 2)
        self.assertEqual(mock_request.call_args.kwargs["url"].split(":")[-1], "fetchMultiDailyMetricsTimeSeries")
        self.assertEqual(result, {})

    def test_parses_multiple_metrics_into_the_legacy_shape(self):
        payload = {
            "multiDailyMetricTimeSeries": [
                {
                    "dailyMetricTimeSeries": [
                        self._series(
                            "CALL_CLICKS",
                            [
                                {"date": {"year": 2026, "month": 3, "day": 1}, "value": "7"},
                                {"date": {"year": 2026, "month": 3, "day": 2}, "value": "3"},
                            ],
                        ),
                        self._series(
                            "WEBSITE_CLICKS",
                            [{"date": {"year": 2026, "month": 3, "day": 1}, "value": "11"}],
                        ),
                    ]
                }
            ]
        }
        result, _ = self._run(payload)

        self.assertEqual(
            result,
            {
                "2026-03-01": {"CALL_CLICKS": 7, "WEBSITE_CLICKS": 11},
                "2026-03-02": {"CALL_CLICKS": 3},
            },
        )
        # Metric keys stay the API enum names, so the stored `metric` column is unchanged.
        self.assertIn("CALL_CLICKS", result["2026-03-01"])

    def test_absent_value_counts_as_zero(self):
        payload = {
            "multiDailyMetricTimeSeries": [
                {
                    "dailyMetricTimeSeries": [
                        self._series("BUSINESS_BOOKINGS", [{"date": {"year": 2026, "month": 3, "day": 1}}])
                    ]
                }
            ]
        }
        result, _ = self._run(payload)
        self.assertEqual(result, {"2026-03-01": {"BUSINESS_BOOKINGS": 0}})

    def test_sub_entity_breakdown_does_not_overwrite_the_location_total(self):
        """A breakdown series carries the same metric split by sub-entity.

        We never request breakdowns, but if one arrives it must not clobber the total.
        """
        payload = {
            "multiDailyMetricTimeSeries": [
                {
                    "dailyMetricTimeSeries": [
                        self._series(
                            "CALL_CLICKS",
                            [{"date": {"year": 2026, "month": 3, "day": 1}, "value": "7"}],
                        ),
                        self._series(
                            "CALL_CLICKS",
                            [{"date": {"year": 2026, "month": 3, "day": 1}, "value": "2"}],
                            sub_entity_type={"dayOfWeek": "MONDAY"},
                        ),
                    ]
                }
            ]
        }
        result, _ = self._run(payload)
        self.assertEqual(result, {"2026-03-01": {"CALL_CLICKS": 7}})

    def test_forbidden_location_returns_empty_without_raising(self):
        result, _ = self._run({}, status=403)
        self.assertEqual(result, {})


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

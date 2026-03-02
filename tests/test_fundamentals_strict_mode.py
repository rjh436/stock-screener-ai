import os
import unittest

from data.fundamentals import _allow_estimated_available_date, _strict_fundamental_mode


class FundamentalStrictModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            "FUNDAMENTAL_STRICT_MODE": os.environ.get("FUNDAMENTAL_STRICT_MODE"),
            "APEX_STRICT_FUNDAMENTALS": os.environ.get("APEX_STRICT_FUNDAMENTALS"),
            "FUNDAMENTAL_ALLOW_ESTIMATED_AVAILABLE_DATE": os.environ.get(
                "FUNDAMENTAL_ALLOW_ESTIMATED_AVAILABLE_DATE"
            ),
        }

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_strict_mode_defaults_on(self) -> None:
        os.environ.pop("FUNDAMENTAL_STRICT_MODE", None)
        os.environ.pop("APEX_STRICT_FUNDAMENTALS", None)
        self.assertTrue(_strict_fundamental_mode())

    def test_strict_mode_can_be_disabled(self) -> None:
        os.environ["FUNDAMENTAL_STRICT_MODE"] = "0"
        self.assertFalse(_strict_fundamental_mode())

    def test_estimated_available_date_default_off(self) -> None:
        os.environ.pop("FUNDAMENTAL_ALLOW_ESTIMATED_AVAILABLE_DATE", None)
        self.assertFalse(_allow_estimated_available_date())


if __name__ == "__main__":
    unittest.main()

import unittest

import core
from core.live import storage


class OutputLayoutTest(unittest.TestCase):
    def test_backtest_and_live_outputs_use_sibling_roots(self):
        for product in core.config.available_products():
            with self.subTest(product=product):
                config = core.config.load_config(product)
                self.assertEqual(config.report.output_root, "output/backtest")

        live_root = storage.output_dir("kc50etf").parent
        self.assertEqual(live_root.name, "live")
        self.assertEqual(live_root.parent.name, "output")


if __name__ == "__main__":
    unittest.main()

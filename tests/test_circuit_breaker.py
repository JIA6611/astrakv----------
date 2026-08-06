import tempfile
import unittest
from pathlib import Path

from astrakv.runtime.circuit_breaker import CircuitBreaker, CircuitBreakerPolicy


class CircuitBreakerTests(unittest.TestCase):
    def test_failure_opens_and_cooldown_requires_explicit_health_restoration(self):
        breaker = CircuitBreaker(CircuitBreakerPolicy(failure_threshold=2, timeout_threshold=2, pressure_threshold=2, cooldown_ns=10))
        breaker.record_failure(now_ns=1)
        self.assertTrue(breaker.allow_dispatch(now_ns=2))
        breaker.record_failure(now_ns=3)
        self.assertEqual(breaker.state, "open")
        self.assertFalse(breaker.allow_dispatch(now_ns=12))
        self.assertEqual(breaker.state, "open")
        self.assertFalse(breaker.allow_dispatch(now_ns=13))
        self.assertEqual(breaker.state, "half_open")
        self.assertFalse(breaker.allow_dispatch(now_ns=13))
        breaker.restore_health(now_ns=13)
        self.assertEqual(breaker.state, "closed")
        self.assertTrue(breaker.allow_dispatch(now_ns=14))

    def test_timeout_and_pressure_open_and_state_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "breaker.json"
            breaker = CircuitBreaker(CircuitBreakerPolicy(failure_threshold=3, timeout_threshold=1, pressure_threshold=2, cooldown_ns=10), state_path=path)
            breaker.record_timeout(now_ns=1)
            self.assertEqual(breaker.state, "open")
            restored = CircuitBreaker(CircuitBreakerPolicy(failure_threshold=3, timeout_threshold=1, pressure_threshold=2, cooldown_ns=10), state_path=path)
            self.assertEqual(restored.state, "open")
            restored.restore_health(now_ns=20)
            restored.record_pressure(now_ns=21)
            restored.record_pressure(now_ns=22)
            self.assertEqual(restored.state, "open")


if __name__ == "__main__":
    unittest.main()

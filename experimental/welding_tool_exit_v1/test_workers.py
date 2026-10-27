import unittest
from unittest.mock import patch
from run_geometry import initialize_worker


class WorkerTests(unittest.TestCase):
    def test_worker_runs_without_optional_threadpool_package(self):
        with patch.dict('sys.modules', {'threadpoolctl': None}):
            try:
                initialize_worker()
            except ModuleNotFoundError:
                self.fail('Optional BLAS helper prevents geometry worker from starting')


if __name__ == '__main__': unittest.main()

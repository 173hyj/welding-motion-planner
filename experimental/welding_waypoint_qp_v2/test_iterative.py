import unittest
try:
    from iterative import optimize_iterations
except ImportError:
    optimize_iterations=None


class IterativeTests(unittest.TestCase):
    def test_multiple_updates_relinearize_and_stop_at_success(self):
        self.assertIsNotNone(optimize_iterations,'iterative controller missing')
        visited=[]
        def step(x):
            visited.append(x)
            return dict(solved=True,proposals=[x+1.])
        def evaluate(x):return dict(score=x,passed=x>=3.)
        result=optimize_iterations(0.,step,evaluate,max_iterations=12)
        self.assertTrue(result['passed'])
        self.assertEqual(result['iterations'],3)
        self.assertEqual(visited,[0.,1.,2.])

    def test_regression_is_rejected_and_stagnation_stops(self):
        self.assertIsNotNone(optimize_iterations,'iterative controller missing')
        result=optimize_iterations(2.,lambda x:dict(solved=True,proposals=[x-1.]),
                                   lambda x:dict(score=x,passed=False),max_iterations=12)
        self.assertEqual(result['state'],2.)
        self.assertFalse(result['passed'])
        self.assertEqual(result['iterations'],1)
        self.assertEqual(result['reason'],'no_improving_step')


if __name__=='__main__':unittest.main()

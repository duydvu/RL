import numpy as np

def SMA(data: np.array, degree: int):
    data = np.concatenate(([data[0]] * (degree - 1), data, [0]))
    slice_sum = sum(data[:degree])
    result = []
    for i in range(1, len(data) - degree + 1):
        result.append(slice_sum)
        slice_sum += data[i + degree - 1] - data[i - 1]
    return np.array(result) / degree


if __name__ == "__main__":
    import unittest
    from numpy import testing

    class TestSMA(unittest.TestCase):
        def test_result(self):
            arr = np.array([1, 2, 3, 4, 5])
            res = SMA(arr, 2)
            testing.assert_array_equal(res, [1, 1.5, 2.5, 3.5, 4.5])

        def test_result_2(self):
            arr = np.array([1, 2, 3, 4, 5])
            res = SMA(arr, 3)
            testing.assert_almost_equal(res, [1, 1.33333333, 2, 3, 4])

    unittest.main()

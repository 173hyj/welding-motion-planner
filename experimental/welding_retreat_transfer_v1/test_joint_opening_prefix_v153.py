import joint_search_v153 as joint


def test_bad_common_opening_prefix_rejects_before_direction_search():
    checked=[]
    def probe(driver):
        checked.append(driver)
        return driver != -.5
    assert not joint.opening_prefix_valid(probe,10.)
    assert checked == [-.5]


def test_opening_prefix_uses_same_half_mm_grid_as_final_online_checker():
    checked=[]
    assert joint.opening_prefix_valid(lambda d:checked.append(d) is None,10.)
    assert checked == [-.5*i for i in range(1,21)]

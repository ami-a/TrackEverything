"""Temporal smoothing methods and the calculator that drives them."""
import numpy as np
import pytest

from TrackEverything import (
    StatisticalCalculator,
    StatMethods,
    StatParams,
    cumulative_moving_average,
    exponential_moving_average,
    finite_moving_average,
    no_average,
)


def feed(calc, scores):
    """Push a sequence of score vectors through a calculator."""
    out = None
    for score in scores:
        out = calc.update(np.asarray(score, dtype=float))
    return out


class TestStatMethodsEnum:
    """StatMethods used to have zero members: bare functions in an Enum body
    become methods rather than members."""

    def test_enum_has_four_members(self):
        assert len(list(StatMethods)) == 4

    def test_members_are_callable(self):
        params = StatParams()
        params.initialize(2)
        params.insert_score(np.array([0.3, 0.7]))
        assert StatMethods.Non(params) == pytest.approx([0.3, 0.7])

    def test_members_are_distinct_from_raw_functions(self):
        assert StatMethods.EMA is not exponential_moving_average
        assert StatMethods.EMA.value.func is exponential_moving_average

    def test_lookup_by_name_works(self):
        assert StatMethods["CMA"] is StatMethods.CMA


class TestNoAverage:
    def test_returns_the_latest_point(self):
        calc = StatisticalCalculator(method=no_average, class_num=2)
        assert feed(calc, [[0.1, 0.9], [0.8, 0.2]]) == pytest.approx([0.8, 0.2])


class TestCumulativeMovingAverage:
    def test_matches_numpy_mean(self):
        rng = np.random.default_rng(0)
        points = rng.random((10, 2))
        calc = StatisticalCalculator(method=cumulative_moving_average, class_num=2)
        result = feed(calc, points)
        assert result == pytest.approx(points.mean(axis=0))

    def test_single_point_is_itself(self):
        calc = StatisticalCalculator(method=cumulative_moving_average, class_num=2)
        assert feed(calc, [[0.4, 0.6]]) == pytest.approx([0.4, 0.6])


class TestFiniteMovingAverage:
    def test_after_warmup_equals_rolling_mean(self):
        """The three-branch warm-up at the top of the function is the risky part."""
        params = StatParams(avg_n=4)
        calc = StatisticalCalculator(
            parameters=params, method=finite_moving_average, class_num=1
        )
        points = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        result = feed(calc, points)
        assert result == pytest.approx([np.mean([3.0, 4.0, 5.0, 6.0])])

    def test_during_warmup_divides_by_window(self):
        params = StatParams(avg_n=4)
        calc = StatisticalCalculator(
            parameters=params, method=finite_moving_average, class_num=1
        )
        assert feed(calc, [[4.0]]) == pytest.approx([1.0])

    def test_window_never_exceeds_avg_n(self):
        params = StatParams(avg_n=3)
        calc = StatisticalCalculator(
            parameters=params, method=finite_moving_average, class_num=1
        )
        feed(calc, [[1.0]] * 10)
        assert len(calc.parameters.pre_score_list) <= 3


class TestExponentialMovingAverage:
    def test_matches_closed_form(self):
        beta = 0.9
        points = [[1.0], [2.0], [3.0], [4.0]]
        params = StatParams(beta=beta)
        calc = StatisticalCalculator(
            parameters=params, method=exponential_moving_average, class_num=1
        )
        result = feed(calc, points)

        expected = 0.0
        for point in points:
            expected = beta * expected + (1 - beta) * point[0]
        assert result == pytest.approx([expected])

    def test_converges_towards_a_constant_signal(self):
        params = StatParams(beta=0.5)
        calc = StatisticalCalculator(
            parameters=params, method=exponential_moving_average, class_num=1
        )
        result = feed(calc, [[1.0]] * 30)
        assert result == pytest.approx([1.0], abs=1e-6)


class TestClassCountAdaptation:
    """class_num is only a guess until the first real score arrives."""

    def test_one_class_score_against_default_two(self):
        """The detection-only path produces a single class."""
        calc = StatisticalCalculator(method=no_average)  # class_num defaults to 2
        assert feed(calc, [[1.0]]) == pytest.approx([1.0])

    def test_three_class_score_against_default_two(self):
        calc = StatisticalCalculator(method=no_average)
        assert feed(calc, [[0.1, 0.2, 0.7]]) == pytest.approx([0.1, 0.2, 0.7])

    def test_scalar_score_is_accepted(self):
        calc = StatisticalCalculator(method=no_average)
        assert calc.update(np.float64(0.5)) == pytest.approx([0.5])


class TestClassEffect:
    """class_effect was documented and advertised but never read."""

    def test_zero_effect_class_is_ignored(self):
        params = StatParams(class_effect=np.array([0.0, 1.0]))
        calc = StatisticalCalculator(
            parameters=params, method=no_average, class_num=2
        )
        assert feed(calc, [[0.9, 0.4]]) == pytest.approx([0.0, 0.4])

    def test_partial_effect_scales_the_contribution(self):
        params = StatParams(class_effect=np.array([0.5, 1.0]))
        calc = StatisticalCalculator(
            parameters=params, method=no_average, class_num=2
        )
        assert feed(calc, [[0.8, 0.8]]) == pytest.approx([0.4, 0.8])

    def test_default_effect_is_neutral(self):
        calc = StatisticalCalculator(method=no_average, class_num=2)
        assert feed(calc, [[0.3, 0.7]]) == pytest.approx([0.3, 0.7])

    def test_explicit_mismatched_effect_raises_clearly(self):
        params = StatParams(class_effect=np.array([1.0, 1.0]))
        calc = StatisticalCalculator(
            parameters=params, method=no_average, class_num=2
        )
        with pytest.raises(ValueError, match="class_effect has 2 entries"):
            calc.update(np.array([0.1, 0.2, 0.7]))


class TestCopyIsolation:
    """Each tracker gets its own calculator via __copy__."""

    def test_copied_calculator_has_independent_state(self):
        original = StatisticalCalculator(method=cumulative_moving_average, class_num=2)
        feed(original, [[1.0, 0.0]])
        clone = original.__copy__()
        feed(clone, [[0.0, 1.0]])
        assert original.get_score() == pytest.approx([1.0, 0.0])

    def test_copy_preserves_accumulated_score(self):
        original = StatisticalCalculator(method=no_average, class_num=2)
        feed(original, [[0.25, 0.75]])
        clone = original.__copy__()
        assert clone.get_score() == pytest.approx([0.25, 0.75])

    def test_copy_deep_copies_the_window_list(self):
        params = StatParams(avg_n=3)
        original = StatisticalCalculator(
            parameters=params, method=finite_moving_average, class_num=1
        )
        feed(original, [[1.0], [2.0]])
        clone = original.__copy__()
        clone.update(np.array([3.0]))
        assert len(original.parameters.pre_score_list) == 2

    def test_copy_keeps_the_method(self):
        original = StatisticalCalculator(method=StatMethods.EMA, class_num=2)
        assert original.__copy__().method is StatMethods.EMA


class TestStatParamsCopy:
    def test_copies_are_independent(self):
        params = StatParams(class_effect=np.array([1.0, 1.0]))
        params.initialize(2)
        clone = params.__copy__()
        clone.class_effect[0] = 0.0
        assert params.class_effect[0] == pytest.approx(1.0)

    def test_none_class_effect_survives_copy(self):
        assert StatParams().__copy__().class_effect is None

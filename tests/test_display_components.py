from keisei.training.display_components import (
    MultiMetricSparkline,
    RollingAverageCalculator,
    Sparkline,
)


def test_multi_metric_sparkline_render():
    spark = MultiMetricSparkline(width=5, metrics=["A", "B"])
    for i in range(3):
        spark.add_data_point("A", i)
        spark.add_data_point("B", i * 2)
    panel_text = spark.render_with_trendlines()
    assert "A:" in panel_text.plain
    assert "B:" in panel_text.plain


def test_rolling_average_calculator():
    calc = RollingAverageCalculator(window_size=3)
    calc.add_value(1)
    calc.add_value(2)
    avg = calc.add_value(3)
    assert avg == 2
    assert calc.get_trend_direction() == "↑"


def test_sparkline_bounded_generation():
    spark = Sparkline(width=5)
    values = [10, 20, 30, 40, 50]
    bounded = spark.generate(values, range_min=0, range_max=100)
    assert len(bounded) == 5

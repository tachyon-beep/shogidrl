from keisei.training.display_components import MultiMetricSparkline, RollingAverageCalculator


def test_multi_metric_sparkline_render():
    spark = MultiMetricSparkline(width=5, metrics=["A", "B"])
    for i in range(3):
        spark.add_data_point("A", i)
        spark.add_data_point("B", i * 2)
    panel = spark.render_with_trendlines()
    assert "A:" in panel.renderable.plain
    assert "B:" in panel.renderable.plain


def test_rolling_average_calculator():
    calc = RollingAverageCalculator(window_size=3)
    calc.add_value(1)
    calc.add_value(2)
    avg = calc.add_value(3)
    assert avg == 2
    assert calc.get_trend_direction() == "↑"

# TUI Display Configuration - Quick Reference

This page summarises the options in `DisplayConfig` that control the advanced training dashboard. These settings can be placed under the `display:` key in your YAML configuration or overridden via CLI using dotted keys.

## Key Options

- `enable_board_display` – Toggle the ASCII board panel.
- `enable_trend_visualization` – Show sparkline charts for metrics.
- `enable_elo_ratings` – Display the Elo rating panel.
- `enable_enhanced_layout` – Use the three‑column dashboard layout.
- `board_unicode_pieces` – Use Japanese piece symbols when possible.
- `sparkline_width` – Width of the trend charts in characters.
- `trend_history_length` – Number of history points retained for trends.
- `elo_initial_rating` – Starting Elo value for both sides.
- `elo_k_factor` – Update factor for rating adjustments.
- `dashboard_height_ratio` – Vertical space reserved for the dashboard.
- `progress_bar_height` – Height of the progress bar section.

Example CLI override:

```bash
python train.py --override display.enable_enhanced_layout=false
```

# Performance Notes (To Be Collected)

Use `keisei/utils/profiling.py` to time rendering operations.  The table below
captures average frame times measured on a development machine using the default
configuration.

| Layout | Avg Frame Time | FPS |
|-------|---------------|-----|
| Compact | ~170 ms | ~5.8 |
| Enhanced | ~185 ms | ~5.4 |

No significant memory increase was observed (<5 MB).  These results indicate the
enhanced layout has a negligible performance penalty.

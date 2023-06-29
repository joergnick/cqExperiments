import pstats
p = pstats.Stats('output')
p.sort_stats('cumulative').print_stats(100)

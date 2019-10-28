#36.3236585,127.5269414
'''
import folium
map_osm = folium.Map(location=[36.3236585,127.5269414])

map_osm
'''
from pyecharts import Pie, Bar, Line, Overlap

attr = ['A','B','C','D','E','F']
v1 = [10, 20, 30, 40, 50, 60]
v2 = [38, 28, 58, 48, 78, 68]
pie = Pie("pie chart", title_pos="center", width=600)
pie.add("A", attr, v1, center=[25, 50], is_random=True, radius=[30, 75], rosetype='radius')
pie.add("B", attr, v2, center=[75, 50], is_randome=True, radius=[30, 75], rosetype='area', is_legend_show=False,
       is_label_show=True)
pie.render("pie.html")

attr = ['A','B','C','D','E','F']
v1 = [10, 20, 30, 40, 50, 60]
v2 = [38, 28, 58, 48, 78, 68]
bar = Bar("Line Bar")
bar.add("bar", attr, v1)
line = Line()
line.add("line", attr, v2)

overlap = Overlap()
overlap.add(bar)
overlap.add(line)

overlap.render("bar_line.html")
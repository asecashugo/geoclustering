## Geoclustering

Picture this: you have a **massive greenfield** to farm for electricity, be it for your offtaker or or a [P2X project](https://orsted.com/en/what-we-do/renewable-energy-solutions/power-to-x). You need to get the most of it while balancing cost.

You will want to:
- Place your assets efficiently
- Connect them in the optimal way

Only then will you have a foundation to perform detail engineering.

This demo does exactly that using [geopandas](https://geopandas.org/en/stable/index.html), [scikit-learn](https://scikit-learn.org/stable/): [clustering](https://scikit-learn.org/stable/modules/clustering.html), [scipy](https://scipy.org/): [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation), [Dijkstra's algorythm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm).
[plotly](https://plotly.com/) for map plotting, [pytest](https://docs.pytest.org/en/stable/) for automated testing in Github Actions

Input:

__What a beautiful, windy, sunny creek! I wonder if we could put it to some use instead of building horrendous luxury villas..._

![base 1](https://github.com/user-attachments/assets/5e6d457a-f8bb-4c3b-b082-0bda49cef02c)


Base points [geojson](https://en.wikipedia.org/wiki/GeoJSON):

__Here we go, these wind turbines will do fine_

![base 2](https://github.com/user-attachments/assets/72773c16-2b58-474f-8bf0-602dff90d415)

Clusters:

__Neat groups_

![base 3](https://github.com/user-attachments/assets/bc469fda-16d8-4443-814b-caf8d4785a0f)

Connections:

__Just like trees branching out_

![base 4](https://github.com/user-attachments/assets/e8f7d59a-0e3d-4220-b923-f716933e03fa)


# NeuroSwarm

NeuroSwarm is a function approximator that fits an n-dimensional grid of hyperplanes. It trades off the generalization power of a large neural network with locally-valid single-layered linear networks. Its properties are comparable to a [tile coder](https://webdocs.cs.ualberta.ca/~sutton/book/8/node6.html).

# Dependencies

* numpy
* matplotlib (to run the example)

# Usage

```python
from neuroswarm import nnswarm

# grid dimensions
dims = [8, 10, 6, 10]

# value limits of each dimension (min, max)
lims = [(3.0, 7.5), (-4.4, 4.2), (9.6, 12.7), (0.0, 1.0)]

# create swarm with learning rate 0.2 and momentum 0.8 (default 0.1 and 0.7071 respectively)
S = nnswarm(dims, lims, 0.2, 0.8)

# training iteration with value 5.5 at location (3.3, -2.1, 11.1, 0.7)
S[3.3, -2.1, 11.1, 0.7] = 5.5

# get approximated value at (3.3, -2.1, 11.1, 0.7)
print S[3.3, -2.1, 11.1, 0.7]
```

# Examples
<p align="center">
  <img src="https://raw.githubusercontent.com/MeepMoop/neuroswarm/master/examples/nnswarm_sincos.png"><br>
  8x8 NeuroSwarm approximating f(x, y) = sin(x) + cos(y) + <i>N</i>(0, 0.1)<br><br>
  <img src="https://raw.githubusercontent.com/MeepMoop/neuroswarm/master/examples/nnswarm_gaussian.png"><br>
  9x9 NeuroSwarm approximating f(x, y) = exp(-(x ** 2 + y ** 2)) + <i>N</i>(0, 0.1)
</p>

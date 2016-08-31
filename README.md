# NeuroSwarm

NeuroSwarm is a function approximator that fits an n-dimensional grid of hyperplanes. It trades off the generalization power of a large neural network with locally-valid single-layered linear networks. Its properties are comparable to a [tile coder](https://webdocs.cs.ualberta.ca/~sutton/book/8/node6.html).

# Dependencies

* numpy
* matplotlib (to run the example)

# Examples
<center>
  <img src="https://raw.githubusercontent.com/MeepMoop/neuroswarm/master/examples/nnswarm_sincos.png"><br>
  8x8 NeuroSwarm approximating f(x, y) = sin(x) + cos(y) + N(0, 0.1)<br><br>
  <img src="https://raw.githubusercontent.com/MeepMoop/neuroswarm/master/examples/nnswarm_gaussian.png"><br>
  9x9 NeuroSwarm approximating f(x, y) = exp(-(x ** 2 + y **2)) + N(0, 0.1)
</center>

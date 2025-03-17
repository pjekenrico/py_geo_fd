# py_geo_fd

Repository dedicated to the virtual deployment of flow diverters following: [Virtual flow diverter deployment and embedding for hemodynamic simulations](https://doi.org/10.1016/j.compbiomed.2024.109023)

## Install

```
pip install -e .
```

## Run

Once properly installed as described above, you can import the `Stent` class:

```
from py_geo_fd import Stent
```

The class is constructed with a `json` file containing the required parameters for all of the involved model parameters. An example file can be found in the `demo` folder and can be used as followed:

```
# Build stent using the config file
stent = Stent("stent_config.json")

# Save the centerline after smoothing. This helps bebugging code and checking that the smoothing is not too large/small.
stent.save_centerline("centerline.vtp")

# Save the stent envelope including the local porosity and wire angles.
stent.save_envelope("stent_envelope")

# Save the stent as a collection of 1D lines as vtp (ideal for embedding)
stent.save_stent_to_vtp("stent.vtp")

# Save the stent as a tetrahaedra mesh (ideal for visualization). It will output sets of positive and negative winding wires for potential embedding.
# The outputs are in the vtu format. No need to specifying that in the path.
stent.save_stent("stent")
```

## Funding
European Union (ERC, CURE, 101045042). The views and opinions expressed are, however, those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
<p align="center">
  <img src="https://cure-erc.github.io/imgs/LOGO-ERC.jpg" alt="Logo ERC" width="175">
  <img src="https://cure-erc.github.io/imgs/CURE_LOGO.png" alt="CURE Logo" width="300">
</p>

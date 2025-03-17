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

1) Build stent using the config file
```
stent = Stent("stent_config.json")
```

2) Save the centerline after smoothing. This helps bebugging code and checking that the smoothing is not too large/small.
```
stent.save_centerline("centerline.vtp")
```

3) Save the stent envelope including the local porosity and wire angles.
```
stent.save_envelope("stent_envelope")
```

4) Save the stent as a collection of 1D lines as vtp (ideal for embedding).
```
stent.save_stent_to_vtp("stent.vtp")
```

5) Save the stent as a tetrahaedra mesh (ideal for visualization). It will output sets of positive and negative winding wires for potential embedding. The outputs are in the vtu format. No need to specifying that in the path.
```
stent.save_stent("stent")
```

To build a stent one requires a surface file readable by [meshio](https://github.com/nschloe/meshio) build from triangles. Other element types will be ignored. With the surface file, you need to compute the vessel centerline along which the flow diverter should be deployed. This can be done using [vmtk](http://www.vmtk.org/). A convenient interface for retrieving it is [3DSlicer](https://www.slicer.org/) which includes an interactive window for clipping and displacing the precomputed curves. Centerlines can be read by `py_geo_fd` in both `json` and `vtp` formats.

## Funding
European Union (ERC, CURE, 101045042). The views and opinions expressed are, however, those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
<p align="center">
  <img src="https://cure-erc.github.io/imgs/CURE_LOGO.png" alt="CURE Logo" width="325">
  <img src="https://cure-erc.github.io/imgs/LOGO-ERC.jpg" alt="Logo ERC" width="175">
</p>

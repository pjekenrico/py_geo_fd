from __future__ import annotations

import json
import numpy as np


class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that puts small lists on single lines."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""

        if isinstance(o, (list, tuple)):
            if self._is_single_line_list(o):
                return "[" + ", ".join(json.dumps(el) for el in o) + "]"
            else:
                self.indentation_level += 1
                output = [self.indent_str + self.encode(el) for el in o]
                self.indentation_level -= 1
                return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"

        elif isinstance(o, dict):
            self.indentation_level += 1
            output = [
                self.indent_str + f"{json.dumps(k)}: {self.encode(v)}"
                for k, v in o.items()
            ]
            self.indentation_level -= 1
            return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"

        else:
            return json.dumps(o)

    def _is_single_line_list(self, o):
        if isinstance(o, (list, tuple)):
            return (
                not any(isinstance(el, (list, tuple, dict)) for el in o)
                and len(o) <= 4
                and len(str(o)) - 2 <= 60
            )

    @property
    def indent_str(self) -> str:
        return " " * self.indentation_level * self.indent

    def iterencode(self, o, **kwargs):
        """Required to also work with `json.dump`."""
        return self.encode(o)


class ctrl(object):
    def __init__(
        self,
        centerline_path: str,
        x_smoothing: float,
        dx_smoothing: float,
        s_start: float,
        *args,
        **kwargs,
    ) -> None:
        self.path = centerline_path
        self.x_s = float(x_smoothing)
        self.dx_s = float(dx_smoothing)
        self.s_start = float(s_start)


class st(object):
    def __init__(self, geo_params, numeric_params, *args, **kwargs) -> None:
        self.geom = geom(**geo_params)
        self.num = num(**numeric_params)


class geom(object):
    def __init__(
        self,
        lw: float,
        w: float,
        d_nom: float,
        Nw: float,
        wire_radius: float = 0.015,
        *args,
        **kwargs,
    ) -> None:
        self.lw = float(lw)
        self.w = float(w)
        self.d_nom = float(d_nom)
        self.Nw = int(Nw)
        self.wire_radius = float(wire_radius)


class num(object):
    def __init__(
        self, wire_resolution: int = 10, n_segments: int = 200, *args, **kwargs
    ) -> None:
        self.wire_resolution = int(wire_resolution)
        self.n_segments = int(n_segments)


class wall(object):
    def __init__(
        self,
        wall_path: str,
        smoothing: float = 0.0,
        n_axial_segments: int = 100,
        n_radial_segments: int = 32,
        margin: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        self.path = wall_path
        self.smoothing = int(smoothing)
        self.n_axial_segments = int(n_axial_segments)
        self.n_radial_segments = int(n_radial_segments)
        self.margin = float(margin)


class Stent_Config(object):
    """Class to bundle all of the parameters used for the stent generation. Examples for the json file to fill the parameters are given in the demo folder."""

    def __init__(
        self,
        centerline_params,
        stent_params,
        wall_params,
        n_cpu: int = 8,
        *args,
        **kwargs,
    ) -> None:
        self.ctrl = ctrl(**centerline_params)
        self.st = st(**stent_params)
        self.wall = wall(**wall_params)
        self.n_cpu = int(n_cpu)
        return


def load_config(stent_dict: str) -> Stent_Config:
    try:
        with open(stent_dict, "r") as f:
            stent_dict = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: The provided file {stent_dict} is", "not a valid JSON document.")
        exit(1)

    return Stent_Config(**stent_dict)


def compute_w_lw():
    path = "/".join(__file__.split("/")[:-1]) + "/stent_data.json"

    try:
        with open(path, "r") as f:
            stent_dict = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: The stent data file is", "not found.")
        exit(1)

    for stent in stent_dict.keys():
        stent_data = stent_dict[stent]

        Nw = int(stent_data["Nw"])
        d = np.array(stent_data["d"])
        L = np.array(stent_data["L"])

        ratios = (L[:-1] ** 2 * d[1:] ** 2 - L[1:] ** 2 * d[:-1] ** 2) / (
            L[:-1] ** 2 - L[1:] ** 2
        )
        w = np.mean(np.pi / Nw * np.sqrt(ratios))
        lw = np.mean(L / np.sqrt(1 - (np.pi * d / (Nw * w)) ** 2))

        stent_data.update({"wire_radius": 0.015})
        stent_data.update({"w": w})
        stent_data.update({"lw": lw})

    with open(path, "w") as f:
        json.dump(stent_dict, f, cls=CompactJSONEncoder, indent=2)
        f.write("\n")


compute_w_lw()

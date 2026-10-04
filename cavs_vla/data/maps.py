"""Map backends that return world-frame polylines for a radius query.

Every backend yields dicts:
    {"pts": (n, 2) float64 world coordinates,
     "kind": "lane" | "connector" | "crosswalk" | "boundary",
     "id": str, "roadblock_id": str | None, "speed_limit": float (m/s, nan if unknown)}

Backends
--------
devkit : nuplan-devkit map API (official; handles the GPKG -> UTM projection).
gpkg   : direct read of nuPlan ``map.gpkg`` with pyogrio, reprojected to the CRS
         stored in the GPKG ``meta`` layer (same transform the devkit applies).
json   : a JSON file {location: [polyline dicts]} (synthetic self-test data, CARLA exports).
none   : no map (ablation only; the policy then has no lane information).
"""
from __future__ import annotations

import abc
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

from ..geometry_np import PolylineIndex

_CACHE: Dict[str, "MapSource"] = {}


class MapSource(abc.ABC):
    backend = "base"

    @abc.abstractmethod
    def query(self, location: str, x: float, y: float, radius: float) -> List[dict]:
        """World-frame polylines whose distance to (x, y) is at most ``radius``."""

    def location_ok(self, location: str) -> bool:
        return True


class NoMap(MapSource):
    backend = "none"

    def query(self, location: str, x: float, y: float, radius: float) -> List[dict]:
        return []


class JsonMap(MapSource):
    backend = "json"

    def __init__(self, path: str) -> None:
        with open(path, "r") as f:
            raw = json.load(f)
        self.index: Dict[str, PolylineIndex] = {}
        for loc, items in raw.items():
            polys = []
            for it in items:
                polys.append(dict(pts=np.asarray(it["pts"], dtype=np.float64), kind=it["kind"], id=str(it["id"]),
                                  roadblock_id=None if it.get("roadblock_id") is None else str(it["roadblock_id"]),
                                  speed_limit=float(it.get("speed_limit", float("nan")))))
            self.index[loc] = PolylineIndex(polys)

    def location_ok(self, location: str) -> bool:
        return location in self.index

    def query(self, location: str, x: float, y: float, radius: float) -> List[dict]:
        if location not in self.index:
            raise KeyError(f"Location '{location}' not present in JSON map (have {sorted(self.index)})")
        return [p for p, _ in self.index[location].query(x, y, radius)]


class DevkitMap(MapSource):
    backend = "devkit"

    def __init__(self, map_root: str, map_version: str) -> None:
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api  # noqa: F401
        self.map_root = map_root
        self.map_version = map_version
        self._apis: Dict[str, object] = {}

    def _api(self, location: str):
        if location not in self._apis:
            from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
            self._apis[location] = get_maps_api(self.map_root, self.map_version, location)
        return self._apis[location]

    def query(self, location: str, x: float, y: float, radius: float) -> List[dict]:
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        api = self._api(location)
        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.CROSSWALK]
        objs = api.get_proximal_map_objects(Point2D(x, y), radius, layers)
        out: List[dict] = []
        seen_boundaries = set()
        for lane in objs[SemanticMapLayer.LANE]:
            pts = np.array([[s.x, s.y] for s in lane.baseline_path.discrete_path], dtype=np.float64)
            out.append(dict(pts=pts, kind="lane", id=str(lane.id), roadblock_id=str(lane.get_roadblock_id()),
                            speed_limit=_float_or_nan(lane.speed_limit_mps)))
            for bnd in (lane.left_boundary, lane.right_boundary):
                if bnd.id in seen_boundaries:
                    continue
                seen_boundaries.add(bnd.id)
                bpts = np.array([[s.x, s.y] for s in bnd.discrete_path], dtype=np.float64)
                out.append(dict(pts=bpts, kind="boundary", id=f"b{bnd.id}", roadblock_id=None,
                                speed_limit=float("nan")))
        for con in objs[SemanticMapLayer.LANE_CONNECTOR]:
            pts = np.array([[s.x, s.y] for s in con.baseline_path.discrete_path], dtype=np.float64)
            out.append(dict(pts=pts, kind="connector", id=str(con.id), roadblock_id=str(con.get_roadblock_id()),
                            speed_limit=_float_or_nan(con.speed_limit_mps)))
        for cw in objs[SemanticMapLayer.CROSSWALK]:
            pts = np.asarray(cw.polygon.exterior.coords, dtype=np.float64)[:, :2]
            out.append(dict(pts=pts, kind="crosswalk", id=f"cw{cw.id}", roadblock_id=None,
                            speed_limit=float("nan")))
        return out


def _float_or_nan(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


class GpkgMap(MapSource):
    """Reads nuPlan map.gpkg directly. Requires pyogrio + geopandas (pip install pyogrio geopandas)."""

    backend = "gpkg"

    def __init__(self, map_root: str) -> None:
        import geopandas  # noqa: F401
        import pyogrio  # noqa: F401
        self.map_root = map_root
        self.index: Dict[str, PolylineIndex] = {}

    def _gpkg_path(self, location: str) -> str:
        cands = sorted(glob.glob(os.path.join(self.map_root, location, "*", "map.gpkg")))
        if not cands:
            raise FileNotFoundError(f"No map.gpkg for location '{location}' under {self.map_root}")
        return cands[-1]

    @staticmethod
    def _first_col(df, names: List[str]) -> Optional[str]:
        for n in names:
            if n in df.columns:
                return n
        return None

    def _load(self, location: str) -> PolylineIndex:
        import pyogrio

        path = self._gpkg_path(location)
        meta = pyogrio.read_dataframe(path, layer="meta")
        crs_row = meta.loc[meta["key"] == "projectedCoordSystem", "value"]
        if len(crs_row) == 0:
            raise ValueError(f"{path}: 'meta' layer has no projectedCoordSystem entry")
        crs = crs_row.iloc[0]

        def layer(name: str):
            df = pyogrio.read_dataframe(path, layer=name, fid_as_index=True)
            df.index = df.index.map(str)
            return df.to_crs(crs)

        polys: List[dict] = []
        lanes = layer("lanes_polygons")
        connectors = layer("lane_connectors")
        baselines = layer("baseline_paths")
        lane_rb_col = self._first_col(lanes, ["lane_group_fid"])
        con_rb_col = self._first_col(connectors, ["lane_group_connector_fid"])
        lane_spd_col = self._first_col(lanes, ["speed_limit_mps"])
        con_spd_col = self._first_col(connectors, ["speed_limit_mps"])
        bl_lane_col = self._first_col(baselines, ["lane_fid"])
        bl_con_col = self._first_col(baselines, ["lane_connector_fid"])
        if bl_lane_col is None and bl_con_col is None:
            raise ValueError(f"{path}: baseline_paths has neither lane_fid nor lane_connector_fid columns")
        for _, row in baselines.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            pts = np.asarray(geom.coords, dtype=np.float64)[:, :2]
            lane_id = _str_or_none(row[bl_lane_col]) if bl_lane_col else None
            con_id = _str_or_none(row[bl_con_col]) if bl_con_col else None
            if lane_id is not None and lane_id in lanes.index:
                rb = _str_or_none(lanes.at[lane_id, lane_rb_col]) if lane_rb_col else None
                spd = _float_or_nan(lanes.at[lane_id, lane_spd_col]) if lane_spd_col else float("nan")
                polys.append(dict(pts=pts, kind="lane", id=lane_id, roadblock_id=rb, speed_limit=spd))
            elif con_id is not None and con_id in connectors.index:
                rb = _str_or_none(connectors.at[con_id, con_rb_col]) if con_rb_col else None
                spd = _float_or_nan(connectors.at[con_id, con_spd_col]) if con_spd_col else float("nan")
                polys.append(dict(pts=pts, kind="connector", id=con_id, roadblock_id=rb, speed_limit=spd))
        try:
            crosswalks = layer("crosswalks")
            for fid, row in crosswalks.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                ext = geom.exterior if hasattr(geom, "exterior") else geom.geoms[0].exterior
                polys.append(dict(pts=np.asarray(ext.coords, dtype=np.float64)[:, :2], kind="crosswalk",
                                  id=f"cw{fid}", roadblock_id=None, speed_limit=float("nan")))
        except Exception as exc:  # layer missing in a map version -> no crosswalks, recorded in the manifest
            polys.append(dict(pts=np.zeros((1, 2)), kind="__warning__", id=f"crosswalks:{exc}",
                              roadblock_id=None, speed_limit=float("nan")))
        try:
            boundaries = layer("boundaries")
            for fid, row in boundaries.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                pts = np.asarray(geom.coords, dtype=np.float64)[:, :2]
                polys.append(dict(pts=pts, kind="boundary", id=f"b{fid}", roadblock_id=None,
                                  speed_limit=float("nan")))
        except Exception as exc:
            polys.append(dict(pts=np.zeros((1, 2)), kind="__warning__", id=f"boundaries:{exc}",
                              roadblock_id=None, speed_limit=float("nan")))
        return PolylineIndex([p for p in polys if p["kind"] != "__warning__"])

    def query(self, location: str, x: float, y: float, radius: float) -> List[dict]:
        if location not in self.index:
            self.index[location] = self._load(location)
        return [p for p, _ in self.index[location].query(x, y, radius)]


def _str_or_none(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    s = str(v)
    if s.endswith(".0"):
        s = s[:-2]
    return s if s and s.lower() != "nan" else None


def make_map_source(backend: str, nuplan_map_root: str = "", map_version: str = "nuplan-maps-v1.0",
                    json_path: str = "") -> MapSource:
    key = f"{backend}|{nuplan_map_root}|{map_version}|{json_path}"
    if key in _CACHE:
        return _CACHE[key]
    if backend == "none":
        src: MapSource = NoMap()
    elif backend == "json":
        if not json_path:
            raise ValueError("map_backend=json needs data.json_map_path")
        src = JsonMap(json_path)
    elif backend == "devkit":
        src = DevkitMap(nuplan_map_root, map_version)
    elif backend == "gpkg":
        src = GpkgMap(nuplan_map_root)
    elif backend == "auto":
        if json_path:
            src = JsonMap(json_path)
        elif not nuplan_map_root:
            raise ValueError("map_backend=auto: set data.nuplan_map_root (nuPlan maps are required for a "
                             "leak-free map input) or data.map_backend=none for the no-map ablation")
        else:
            try:
                src = DevkitMap(nuplan_map_root, map_version)
            except ImportError:
                try:
                    src = GpkgMap(nuplan_map_root)
                except ImportError as exc:
                    raise ImportError("Neither nuplan-devkit nor pyogrio+geopandas is installed; install one "
                                      "of them to read nuPlan maps") from exc
    else:
        raise ValueError(f"Unknown map backend {backend}")
    _CACHE[key] = src
    return src

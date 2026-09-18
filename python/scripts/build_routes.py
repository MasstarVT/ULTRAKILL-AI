"""Regenerate the room-trunk route files from the shipped ULTRAKILL scene bundles.

Layer 2 of the route signal (`docs/superpowers/specs/2026-09-17-route-fallback-and-boss-levels-design.md`):
on a level whose live door gates fail `GateProgress`'s own guard, Python reads a committed per-level
JSON ladder of ROOM rungs instead. This script is the only thing that writes those files.

    python scripts/build_routes.py                     # regenerate every shipped level
    python scripts/build_routes.py --validate           # + the acceptance report of spec 7.5
    python scripts/build_routes.py --levels 0-5 7-1     # one or more levels
    python scripts/build_routes.py --dry-run            # measure, write nothing

Read-only with respect to the game: it opens the scene bundles under `<game>/ULTRAKILL_Data/
StreamingAssets/aa/StandaloneWindows64` and nothing else. It never launches the game, never opens a
port and is single-threaded, so it is safe to run beside a live training run. ~4 min for all 33
levels; the voxel standability pass (R4) is 2-21 s per level and dominates.

THE SIGNAL (spec 2.1, 2.6)
    nodes   the level's rooms, which ULTRAKILL names `<N>[suffix] - <Name>` or `<Branch><N> - <Name>`,
            and whose own transform sits at the doorway the player enters them through.
    order   the room numbering inside a NUMBERING SCOPE, scopes chained end to end. The authored
            activation wiring (`Door.activatedRooms`, `CheckPoint.rooms`, `ActivateNextWave.toActivate`,
            ...) is used to link and splice scopes and to prove liveness, but deliberately NOT to
            produce the order: an adjacency BFS reports 1-1's spawn field as one hop from the goal,
            because the final door opens onto it.
    ladder  `hops = last_rung - rung`, the same shape the live door gates publish, so `GateProgress`
            and observation slots 448-455 are reused unchanged.

THE GUARD PIPELINE, in the spec's order (4.3): I6 -> T -> R4 -> I2 -> R1 -> I3 -> R2.
    I6  optional areas are not the route: a `secret`/`bonus` name, an `S` ordinal suffix, `P Door`
        or `Prime`.
    T   trunk only: a parallel set (>= 2 suffixed rooms at one ordinal in one scope, or the whole
        branch-prefix region when the level carries more than one branch prefix) is dropped.
        `GateProgress` carries one `best_hops` and cannot express "both required".
    R4  reachability: a rung with no standable voxel cell inside `_is_reached`'s own cylinder
        (8 m horizontal, 6 m vertical) is dropped. Without it 5-2 can never target its own exit.
    I2  separation: no two rungs within 16 m = 2 x `gate_reach_m`, or one arrival marks several.
    R1  at least 3 rungs.  I3  at most 24 rungs.  R2  tour ratio <= 1.35, on the rungs that SHIP.

The guards say a ladder is well formed; they do not say it is the SAME ladder. `EXPECTED_SHIPPED`
pins which levels ship and `EXPECTED_SHAPE` pins the shape of each one, because this script rewrites
the committed files in place and a rerun after a game update must not re-aim a level in silence.

Derived from the measured offline work of 2026-09-17: the chain assembly from `checkpoint-chain/
ladder.py`, the emitter from `judge/final_table.py`, the revised guards from `spec-rev/recompute.py`,
the standability probe from `spec-rev/reach_verify.py` over `geometry-geodesic/ukgeom.py` +
`voxel.py`, and the leg-witness measure from `spec-rev/legs.py`. The bundle and SerializedFile
readers are the pure-python ones those scripts share; they are inlined here so that regenerating the
data needs nothing but this file, the repo and the game install.
"""

from __future__ import annotations

import argparse
import collections
import io
import itertools
import json
import lzma
import math
import os
import re
import statistics
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultrakill_ai.procmem import cap_blas_threads, refuse_if_commit_high  # noqa: E402

cap_blas_threads()  # before numpy: OpenBLAS reserves ~785 MB of commit for thread buffers at load
# This one parses every shipped scene bundle and is the heaviest offline script in the repo. It is safe
# beside a live run only while the box HAS room; at 85% commit it is the thing that tips it over.
refuse_if_commit_high("build_routes.py")

import numpy as np  # noqa: E402

from ultrakill_ai.campaign import CAMPAIGN_LEVELS, CAMPAIGN_LEVELS_SHIPPED, safe_name  # noqa: E402

# ---------------------------------------------------------------------------- constants

VERSION = 2
SOURCE = "room-trunk offline v2"

SEP = 16.0              # I2: 2 x GateProgress.gate_reach_m
MIN_RUNGS = 3           # R1
MAX_RUNGS = 24          # I3
MAX_TOUR = 1.35         # R2

REACH_H = 8.0           # R4: GateProgress._is_reached, open=false -> margin 1.0
REACH_V = 6.0
PROBE_PAD = 14.0        # half-size of the voxel box around a rung; > reach, to report the true nearest
PLAYER_H = 3.5          # CapsuleCollider m_Height on FirstRoom/Player
EYE_OFFSET = 1.75       # NewMovement puts the player transform at the capsule centre
SLOPE_DEG = 55.0        # steepest surface the player can stand on
VOX_CS, VOX_CH = 0.5, 0.25
SAMPLE_SPACING = 0.25

CP_NEAR = 60.0          # `checkpoints_within_60m`
LEG_NEAR = 40.0         # `legs_witnessed`
PIT_MAX_M = 70.0        # spec 7.1 test 9: a rung named `Pit` may only be hops 0, and near the pit

# ---- the manual override file -------------------------------------------------------------------
#
# `rung_overrides.json`, beside the routes themselves (NOT `route_*.json`, which is the glob every
# reader of the route files uses). A room's CENTROID is the right rung almost
# everywhere, but it is not a promise that the whole 8 m x 6 m reach cylinder around it stays inside
# that room: on `Level 0-3` the centroid of `2 - Side Hallway - Floor 1` sat 1.5 m past the main
# room's far wall, so the cylinder covered the WALL FACE on the main-room side down to y 4 and the
# rung was credited from the air against it, without the player ever entering the hallway (680 of 801
# live credit steps airborne, and a scripted hold-forward-and-jump run that never crossed the wall
# credited it 31 times). That is a property of the level's geometry, not of the pipeline, so there is
# nothing in the generator to fix -- and a plain hand-edit of the emitted JSON would be silently
# undone the next time spec risk 3 says to rerun this script after a game update.
#
# So the fix lives here, in data the generator applies itself. Three rules make it safe:
#
#   1. it is applied LAST, after every guard (I6 -> T -> R4 -> I2 -> R1 -> I3) and BEFORE the
#      diagnostics (R2's tour ratio, `legs_witnessed`, `checkpoints_within_60m`,
#      `last_rung_to_exit_m`), so what ships is measured on the positions that ship -- the property
#      `test_tour_ratio_passes_and_is_reproduced_from_the_shipped_positions` exists to hold;
#   2. it moves a rung, and never adds, removes or reorders one. `hops` is assigned after this from
#      the row order, so a total order cannot be broken by an override;
#   3. every entry carries `was`, the position the generator itself produced when the override was
#      written. If the generator no longer produces that position the rooms have MOVED, the
#      hand-picked point is no longer known to be inside the room, and the override is REFUSED with
#      a loud note that `--validate` turns into a failure. A stale override is worse than none.
OVERRIDES_NAME = "rung_overrides.json"  # NOT route_*.json: that glob is the route files themselves
OVERRIDE_WAS_TOL_M = 1.0  # rule 3: how far the generator's own position may drift and still match `was`

# Levels whose gate ladder layer 1 ACCEPTS but which the collapse detector calls collapsed at the
# spawn, and whose trunk the S1 measurement showed reproduces a sensible walkable order. Their file
# ships beside the ladder and is read only by a run with `prefer_route_when_collapsed: true` (the
# lead's ruling, default false) -- `GateProgress._gates` gives a healthy level's file no way in at
# all. An explicit allow-list, not a guard result: of the other four collapsed levels 1-1 waits for
# S3 to stamp its skull locks, and 1-2 (tour 1.426), 2-3 (tour 1.393) and 8-1 (exit room dropped by
# guard T, last leg 1513 m) are not trunks anyone should hand an agent.
COLLAPSED_SHIP = ("0-3", "4-3")

# The 14 levels the spec ships: the 12 with no gate ladder, plus the two collapsed ones above.
# `--validate` fails when the emitted set differs, because a file appearing or vanishing is a
# coverage change and must not be silent (spec 7.1 test 10, risk 1).
EXPECTED_SHIPPED = ("0-3", "0-5", "1-4", "2-4", "4-2", "4-3", "4-4", "5-2",
                    "7-1", "7-2", "7-3", "7-4", "8-3", "8-4")

# The SHAPE of each shipped ladder, not just the set. Spec risk 3 tells the operator to rerun this
# script after every game update, and `write_docs` rewrites the committed files in place -- so
# without this table a patch that moves one room, or a `_filter_rooms` heuristic that drops one,
# re-aims the agent at a different sequence of places while every structural invariant still holds
# and `--validate` still prints PASS. The set check cannot see it: the same 12 files ship. A change
# here has to be a deliberate edit with a measured reason, the way the two rows below that differ
# from spec 10 were (`docs/level-survey.md` section 9).
#
#   level -> (rungs, med gap m, max gap m, tour, legs witnessed, checkpoints within 60 m,
#             last rung -> pit m)
#
# Measured 2026-09-17 against the shipped bundles. It is spec 10's own table, with the two rows the
# survey records as deviations:
#   8-3 ships 16 rungs, not 13. The design measured it on revision 1's file, which the budget cap
#       had already cut to 24 rungs -- dropping `2 - Shifting Hallway`, `3 - Shifting Arena` and
#       `10 - Split Color Door` -- before the trunk collapse ran. Spec 4.3 moves I3 last, so after T
#       the cap binds on nothing and those three trunk rooms survive: med gap 258 -> 167 m, tour
#       1.247 -> 1.239. The 814 m step and the 1 690 m final leg are unchanged.
#   5-2 keeps `1A - Opening + 1B - Second Rock` as one rung. Both sit at exactly (0, -10, 300), and
#       a group inside one `_is_reached` cylinder is not an order the agent can take either way, so
#       I2 collapses it rather than T dropping it. 7 rungs, which is what the design says.
EXPECTED_SHAPE = {
    # The two collapsed-ladder levels (COLLAPSED_SHIP), measured 2026-09-17. 0-3's trunk is the one the S1
    # survey printed room by room: 11 rungs from `1 - Main Room - Floor 1` to the merged
    # `10B - Second Encounter + 11 - Boss Arena - Floor 2`, in the order the level is actually walked
    # (the gate ladder's own hops along it run 2,2,3,4,5,6,4,4,3,2,0 -- which is why it collapses).
    "0-3": (11, 64.9, 117.6, 0.944, "9/11", "4/4", 119.7),
    "4-3": (8, 64.8, 249.8, 1.000, "6/8", "3/3", 218.5),
    "0-5": (5, 46.5, 137.2, 1.000, "3/5", "1/1", 211.1),
    "1-4": (4, 81.2, 117.0, 1.000, "2/4", "2/2", 149.5),
    "2-4": (4, 425.0, 430.2, 1.000, "1/4", "1/2", 111.2),
    "4-2": (6, 171.5, 268.1, 1.000, "4/6", "2/4", 162.9),
    "4-4": (8, 119.1, 1099.4, 1.000, "4/8", "2/3", 539.9),
    "5-2": (7, 70.9, 407.1, 1.000, "6/7", "3/4", 372.8),
    "7-1": (10, 80.0, 477.9, 1.000, "5/10", "2/5", 127.9),
    "7-2": (15, 76.3, 384.3, 1.255, "10/15", "5/6", 168.3),
    "7-3": (12, 103.8, 184.3, 1.174, "8/12", "4/4", 224.9),
    "7-4": (4, 291.0, 916.9, 1.000, "3/4", "5/7", 62.2),
    "8-3": (16, 167.0, 813.6, 1.239, "12/16", "8/13", 1689.6),
    "8-4": (4, 64.5, 306.5, 1.000, "0/4", "0/1", 62.2),
}
SHAPE_FIELDS = ("rungs", "med gap m", "max gap m", "tour",
                "legs witnessed", "cp within 60 m", "last rung -> pit m")
SHAPE_FMT = ("%d", "%.1f", "%.1f", "%.3f", "%s", "%s", "%.1f")
SHAPE_TOL_M = 0.05      # every distance in the table is already rounded to 0.1 m
SHAPE_TOL_TOUR = 5e-4

MONOSCRIPT_BUNDLE = "monoscript_monoscripts.bundle"
MONO_CAB = "b552895809b12015978369875364470f"


def levels_in_order() -> list[str]:
    """The 33 shipped campaign levels, in mission order, as short names ("0-1"). 9-1 and 9-2 are in
    `CAMPAIGN_LEVELS` but ship no bundle in this build."""
    return [s[len("Level "):] for s in CAMPAIGN_LEVELS if s in CAMPAIGN_LEVELS_SHIPPED]


def game_dir() -> Path:
    """The game install, from `mod/GamePaths.props` (gitignored, per machine)."""
    env = os.environ.get("ULTRAKILL_DIR")
    if env:
        return Path(env)
    for name in ("GamePaths.props", "GamePaths.props.example"):
        p = ROOT.parent / "mod" / name
        if not p.exists():
            continue
        m = re.search(r"<UltrakillDir>(.*?)</UltrakillDir>", p.read_text(encoding="utf8"))
        if m and m.group(1).strip():
            return Path(m.group(1).strip())
    raise SystemExit("cannot find the game: copy mod/GamePaths.props.example to GamePaths.props, "
                     "or set ULTRAKILL_DIR")


def bundles_dir() -> Path:
    return game_dir() / "ULTRAKILL_Data" / "StreamingAssets" / "aa" / "StandaloneWindows64"


def dist3(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def pos_key(p) -> str:
    """`CampaignPatches.Key()`: the position rounded to whole metres. Unity's `Mathf.RoundToInt` is
    round-half-to-even, which is what python's own `round` does."""
    return "%d,%d,%d" % (int(round(p[0])), int(round(p[1])), int(round(p[2])))


def seg_dist(p, a, b) -> float:
    """Distance from a point to the segment a-b."""
    d = [b[i] - a[i] for i in range(3)]
    ln = sum(x * x for x in d)
    t = 0.0 if ln == 0 else max(0.0, min(1.0, sum((p[i] - a[i]) * d[i] for i in range(3)) / ln))
    return dist3(p, [a[i] + t * d[i] for i in range(3)])


# ============================================================================ 1. UnityFS bundle
# Minimal read-only UnityFS reader, pure python (LZ4 block + LZMA1 raw).

def _lz4_block(src: bytes, usize: int) -> bytes:
    dst = bytearray(usize)
    si = di = 0
    n = len(src)
    while si < n:
        tok = src[si]
        si += 1
        ll = tok >> 4
        if ll == 15:
            while True:
                b = src[si]
                si += 1
                ll += b
                if b != 255:
                    break
        if ll:
            dst[di:di + ll] = src[si:si + ll]
            si += ll
            di += ll
        if si >= n:
            break
        off = src[si] | (src[si + 1] << 8)
        si += 2
        ml = tok & 15
        if ml == 15:
            while True:
                b = src[si]
                si += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        start = di - off
        if off >= ml:
            dst[di:di + ml] = dst[start:start + ml]
            di += ml
        else:                                        # overlapping copy: repeat the pattern
            pat = bytes(dst[start:di])
            dst[di:di + ml] = (pat * (ml // off + 1))[:ml]
            di += ml
    if di != usize:
        raise ValueError("lz4 block short: %d of %d" % (di, usize))
    return bytes(dst)


def _decomp(data: bytes, usize: int, ctype: int) -> bytes:
    if ctype == 0:
        return data
    if ctype in (2, 3):
        return _lz4_block(data, usize)
    if ctype == 1:
        props = data[:5]
        lc = props[0] % 9
        r = props[0] // 9
        lp, pb = r % 5, r // 5
        ds = struct.unpack("<I", props[1:5])[0]
        d = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[
            {"id": lzma.FILTER_LZMA1, "lc": lc, "lp": lp, "pb": pb, "dict_size": ds}])
        return d.decompress(data[5:], usize)
    raise ValueError("unknown bundle compression %r" % ctype)


def _cstr(f) -> str:
    out = bytearray()
    while True:
        c = f.read(1)
        if not c or c == b"\0":
            return out.decode("utf8", "replace")
        out += c


def unpack_bundle(path) -> dict[str, bytes]:
    """{node_path: bytes} for one .bundle."""
    with open(path, "rb") as f:
        sig = _cstr(f)
        if sig != "UnityFS":
            raise ValueError("not a UnityFS bundle: %r" % sig)
        ver = struct.unpack(">I", f.read(4))[0]
        _cstr(f)
        _cstr(f)
        size, cbs, ubs, flags = struct.unpack(">qIII", f.read(20))
        if ver >= 7:
            f.seek((f.tell() + 15) & ~15)
        if flags & 0x80:
            pos = f.tell()
            f.seek(size - cbs)
            bi = f.read(cbs)
            f.seek(pos)
        else:
            bi = f.read(cbs)
        b = io.BytesIO(_decomp(bi, ubs, flags & 0x3F))
        b.read(16)
        nblocks = struct.unpack(">i", b.read(4))[0]
        blocks = [struct.unpack(">IIH", b.read(10)) for _ in range(nblocks)]
        nnodes = struct.unpack(">i", b.read(4))[0]
        nodes = []
        for _ in range(nnodes):
            off, sz, fl = struct.unpack(">qqI", b.read(20))
            nodes.append((off, sz, fl, _cstr(b)))
        if flags & 0x200:
            f.seek((f.tell() + 15) & ~15)
        data = b"".join(_decomp(f.read(cs), us, fl & 0x3F) for us, cs, fl in blocks)
    return {name: data[off:off + sz] for off, sz, _fl, name in nodes}


# ============================================================================ 2. SerializedFile
# Minimal SerializedFile (format >= 22) reader with type-tree object decoding.

_COMMON_STRINGS = [
    "AABB", "AnimationClip", "AnimationCurve", "AnimationState", "Array", "Base", "BitField", "bitset",
    "bool", "char", "ColorRGBA", "Component", "data", "deque", "double", "dynamic_array",
    "FastPropertyName", "first", "float", "Font", "GameObject", "Generic Mono", "GradientNEW", "GUID",
    "GUIStyle", "int", "list", "long long", "map", "Matrix4x4f", "MdFour", "MonoBehaviour", "MonoScript",
    "m_ByteSize", "m_Curve", "m_EditorClassIdentifier", "m_EditorHideFlags", "m_Enabled",
    "m_ExtensionPtr", "m_GameObject", "m_Index", "m_IsArray", "m_IsStatic", "m_MetaFlag", "m_Name",
    "m_ObjectHideFlags", "m_PrefabInternal", "m_PrefabParentObject", "m_Script", "m_StaticEditorFlags",
    "m_Type", "m_Version", "Object", "pair", "PPtr<Component>", "PPtr<GameObject>", "PPtr<Material>",
    "PPtr<MonoBehaviour>", "PPtr<MonoScript>", "PPtr<Object>", "PPtr<Prefab>", "PPtr<Sprite>",
    "PPtr<TextAsset>", "PPtr<Texture>", "PPtr<Texture2D>", "PPtr<Transform>", "Prefab", "Quaternionf",
    "Rectf", "RectInt", "RectOffset", "second", "set", "short", "size", "SInt16", "SInt32", "SInt64",
    "SInt8", "staticvector", "string", "TextAsset", "TextMesh", "Texture", "Texture2D", "Transform",
    "TypelessData", "UInt16", "UInt32", "UInt64", "UInt8", "unsigned int", "unsigned long long",
    "unsigned short", "vector", "Vector2f", "Vector3f", "Vector4f", "m_ScriptingClassIdentifier",
    "Gradient", "Type*", "int2_storage", "int3_storage", "BoundsInt", "m_CorrespondingSourceObject",
    "m_PrefabInstance", "m_PrefabAsset", "FileSize", "Hash128", "RenderingLayerMask"]
COMMON = b"\0".join(s.encode() for s in _COMMON_STRINGS) + b"\0"
assert COMMON.index(b"m_GameObject\0") == 374 and COMMON.index(b"string\0") == 840
assert COMMON.index(b"\0vector\0") == 980 and COMMON.index(b"Vector3f\0") == 997
assert COMMON.index(b"Hash128\0") == 1161


def _zs(buf: bytes, off: int) -> str:
    return buf[off:buf.index(b"\0", off)].decode("utf8", "replace")


class _TypeNode:
    __slots__ = ("level", "type", "name", "size", "flag", "children")

    def __init__(self, level, type_, name, size, flag):
        self.level, self.type, self.name, self.size, self.flag = level, type_, name, size, flag
        self.children = []


class SFile:
    _PRIM = {"SInt8": "b", "UInt8": "B", "char": "B", "bool": "?", "SInt16": "h", "short": "h",
             "UInt16": "H", "unsigned short": "H", "SInt32": "i", "int": "i", "UInt32": "I",
             "unsigned int": "I", "Type*": "I", "SInt64": "q", "long long": "q", "UInt64": "Q",
             "unsigned long long": "Q", "FileSize": "Q", "float": "f", "double": "d"}

    def __init__(self, data: bytes):
        self.d = d = data
        _ms, _fs, ver, _do = struct.unpack(">IIII", d[:16])
        self.big = d[16]
        p = 20
        if ver < 22:
            raise ValueError("SerializedFile version %d not supported" % ver)
        _ms, _fs, do, _ = struct.unpack(">IqqQ", d[p:p + 28])
        p += 28
        self.ver, self.data_off = ver, do
        self.e = e = ">" if self.big else "<"
        self.unity = _zs(d, p)
        p = d.index(b"\0", p) + 1
        self.platform, = struct.unpack(e + "i", d[p:p + 4])
        p += 4
        self.has_tt = d[p]
        p += 1
        ntypes, = struct.unpack(e + "i", d[p:p + 4])
        p += 4
        self.types = []
        for _ in range(ntypes):
            p, t = self._read_type(p, False)
            self.types.append(t)
        nobj, = struct.unpack(e + "i", d[p:p + 4])
        p += 4
        self.objects = {}
        for _ in range(nobj):
            p = (p + 3) & ~3
            pid, start, size, tid = struct.unpack(e + "qqIi", d[p:p + 24])
            p += 24
            self.objects[pid] = (start + do, size, tid)
        nscr, = struct.unpack(e + "i", d[p:p + 4])
        p += 4
        self.scripts = []
        for _ in range(nscr):
            fi, = struct.unpack(e + "i", d[p:p + 4])
            p = ((p + 4) + 3) & ~3
            li, = struct.unpack(e + "q", d[p:p + 8])
            p += 8
            self.scripts.append((fi, li))
        next_, = struct.unpack(e + "i", d[p:p + 4])
        p += 4
        self.externals = []
        for _ in range(next_):
            p = d.index(b"\0", p) + 1
            p += 16 + 4
            self.externals.append(_zs(d, p))
            p = d.index(b"\0", p) + 1

    def _read_type(self, p, is_ref):
        d, e = self.d, self.e
        cid, = struct.unpack(e + "i", d[p:p + 4])
        p += 4
        p += 1                                        # stripped
        sti, = struct.unpack(e + "h", d[p:p + 2])
        p += 2
        if (is_ref and sti >= 0) or cid < 0 or cid == 114:
            p += 16
        p += 16
        root = None
        if self.has_tt:
            nn, sb = struct.unpack(e + "ii", d[p:p + 8])
            p += 8
            raw = d[p:p + nn * 32]
            p += nn * 32
            sbuf = d[p:p + sb]
            p += sb
            stack = []
            for i in range(nn):
                _v, lvl, _tf, to, no, bs, _idx, mf, _rh = struct.unpack(
                    e + "HBBIIiiiQ", raw[i * 32:i * 32 + 32])
                ts = _zs(COMMON, to & 0x7FFFFFFF) if to & 0x80000000 else _zs(sbuf, to)
                ns = _zs(COMMON, no & 0x7FFFFFFF) if no & 0x80000000 else _zs(sbuf, no)
                n = _TypeNode(lvl, ts, ns, bs, mf)
                if lvl == 0:
                    root, stack = n, [n]
                else:
                    stack = stack[:lvl]
                    stack[-1].children.append(n)
                    stack.append(n)
            if is_ref:
                for _ in range(3):
                    p = d.index(b"\0", p) + 1
            else:
                cnt, = struct.unpack(e + "i", d[p:p + 4])
                p += 4 + 4 * cnt
        return p, {"class_id": cid, "script_idx": sti, "tree": root}

    def read(self, pid, max_array=200000):
        start, _size, tid = self.objects[pid]
        self._p = start
        return self._val(self.types[tid]["tree"], max_array)

    def _align(self):
        self._p = (self._p + 3) & ~3

    def _val(self, n, max_array):
        d, e = self.d, self.e
        t = n.type
        fmt = self._PRIM.get(t)
        if fmt:
            sz = struct.calcsize(fmt)
            v, = struct.unpack(e + fmt, d[self._p:self._p + sz])
            self._p += sz
        elif t == "string":
            ln, = struct.unpack(e + "i", d[self._p:self._p + 4])
            self._p += 4
            v = d[self._p:self._p + ln].decode("utf8", "replace")
            self._p += ln
            self._align()
            return v
        elif t == "TypelessData":
            ln, = struct.unpack(e + "i", d[self._p:self._p + 4])
            self._p += 4
            v = ("<bytes>", ln, self._p)
            self._p += ln
        elif n.children and n.children[0].type == "Array":
            arr = n.children[0]
            ln, = struct.unpack(e + "i", d[self._p:self._p + 4])
            self._p += 4
            elem = arr.children[1]
            efmt = self._PRIM.get(elem.type)
            if efmt and not (elem.flag & 0x4000):
                sz = struct.calcsize(efmt)
                if ln > max_array:
                    v = ("<array>", elem.type, ln, self._p)
                else:
                    v = list(struct.unpack(e + str(ln) + efmt, d[self._p:self._p + sz * ln]))
                self._p += sz * ln
            else:
                v = [self._val(elem, max_array) for _ in range(ln)]
            if arr.flag & 0x4000:
                self._align()
        elif t == "Array":
            raise RuntimeError("bare array")
        else:
            v = {c.name: self._val(c, max_array) for c in n.children}
        if n.flag & 0x4000:
            self._align()
        return v


# ============================================================================ 3. Scene model

def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _qrot(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    cx = y * vz - z * vy + w * vx
    cy = z * vx - x * vz + w * vy
    cz = x * vy - y * vx + w * vz
    return (vx + 2 * (y * cz - z * cy), vy + 2 * (z * cx - x * cz), vz + 2 * (x * cy - y * cx))


_SCRIPT_MAP: dict[int, tuple[str, str, str]] | None = None


def script_map() -> dict[int, tuple[str, str, str]]:
    """MonoScript m_PathID -> (namespace, class, assembly), from `monoscript_monoscripts.bundle`.

    Every scene's MonoBehaviour names its script by a PPtr into that one shared file, so this map is
    what turns class_id 114 objects into `Door`, `CheckPoint`, `FinalPit` and the rest. Built once per
    process and held in memory; nothing is cached on disk.
    """
    global _SCRIPT_MAP
    if _SCRIPT_MAP is not None:
        return _SCRIPT_MAP
    nodes = unpack_bundle(bundles_dir() / MONOSCRIPT_BUNDLE)
    node = next((k for k in nodes if MONO_CAB in k and "." not in k), None)
    if node is None:
        raise SystemExit("%s does not hold CAB-%s" % (MONOSCRIPT_BUNDLE, MONO_CAB))
    mono = SFile(nodes[node])
    out = {}
    for pid, (_st, _sz, tid) in mono.objects.items():
        if mono.types[tid]["class_id"] != 115:
            continue
        o = mono.read(pid)
        out[pid] = (o.get("m_Namespace", ""), o["m_ClassName"], o.get("m_AssemblyName", ""))
    _SCRIPT_MAP = out
    return out


class Scene:
    """GameObject hierarchy, world transforms, active state and script names for one scene file."""

    def __init__(self, data: bytes, scripts: dict):
        self.sf = sf = SFile(data)
        self.scripts = scripts
        ext = [i for i, e in enumerate(sf.externals) if MONO_CAB in e]
        self.mono_ext = (ext[0] + 1) if ext else -1
        self.by_class = {}
        for pid, (_st, _sz, tid) in sf.objects.items():
            self.by_class.setdefault(sf.types[tid]["class_id"], []).append(pid)
        self.go, self.tf, self.comp_go = {}, {}, {}
        for pid in self.by_class.get(1, []):
            o = sf.read(pid)
            self.go[pid] = {"name": o["m_Name"], "active": o["m_IsActive"], "layer": o["m_Layer"],
                            "tag": o["m_Tag"],
                            "comps": [c["component"]["m_PathID"] for c in o["m_Component"]]}
            for c in self.go[pid]["comps"]:
                self.comp_go[c] = pid
        for cid in (4, 224):                          # Transform, RectTransform
            for pid in self.by_class.get(cid, []):
                o = sf.read(pid)
                self.tf[pid] = {"go": o["m_GameObject"]["m_PathID"],
                                "father": o["m_Father"]["m_PathID"],
                                "children": [c["m_PathID"] for c in o["m_Children"]],
                                "pos": tuple(o["m_LocalPosition"].values()),
                                "rot": tuple(o["m_LocalRotation"].values()),
                                "scale": tuple(o["m_LocalScale"].values())}
        self.go_tf = {t["go"]: pid for pid, t in self.tf.items()}
        self._world = {}
        self.script_of = {}
        for pid in self.by_class.get(114, []):
            st = sf.objects[pid][0]
            fid, spid = struct.unpack("<iq", sf.d[st + 16:st + 28])
            if fid == self.mono_ext and spid in scripts:
                self.script_of[pid] = scripts[spid][1]
            else:
                self.script_of[pid] = "ext%d:%d" % (fid, spid)
        self._by_script = collections.defaultdict(list)
        for pid, name in self.script_of.items():
            self._by_script[name].append(pid)
        self._desc = {}

    def world(self, tfid):
        if tfid in self._world:
            return self._world[tfid]
        t = self.tf[tfid]
        if t["father"] == 0 or t["father"] not in self.tf:
            w = (t["pos"], t["rot"], t["scale"])
        else:
            pp, pr, ps = self.world(t["father"])
            lp = (t["pos"][0] * ps[0], t["pos"][1] * ps[1], t["pos"][2] * ps[2])
            rp = _qrot(pr, lp)
            w = ((pp[0] + rp[0], pp[1] + rp[1], pp[2] + rp[2]), _qmul(pr, t["rot"]),
                 (ps[0] * t["scale"][0], ps[1] * t["scale"][1], ps[2] * t["scale"][2]))
        self._world[tfid] = w
        return w

    def go_pos(self, goid):
        return self.world(self.go_tf[goid])[0]

    def active_in_hierarchy(self, goid):
        t = self.go_tf.get(goid)
        while t and t in self.tf:
            if not self.go[self.tf[t]["go"]]["active"]:
                return False
            t = self.tf[t]["father"]
        return True

    def comps(self, script):
        return list(self._by_script.get(script, ()))

    def read(self, pid):
        return self.sf.read(pid)

    def go_of(self, comp_pid):
        return self.comp_go.get(comp_pid)

    def go_scripts(self, goid):
        out = []
        for c in self.go[goid]["comps"]:
            if c in self.script_of:
                out.append(self.script_of[c])
            elif c in self.sf.objects:
                out.append("#%d" % self.sf.types[self.sf.objects[c][2]]["class_id"])
        return out

    def descendants(self, goid):
        got = self._desc.get(goid)
        if got is not None:
            return got
        out, stack = [], [self.go_tf[goid]]
        while stack:
            t = stack.pop()
            for c in self.tf[t]["children"]:
                if c in self.tf:
                    out.append(self.tf[c]["go"])
                    stack.append(c)
        self._desc[goid] = out
        return out

    def ref_go(self, pptr):
        """The GameObject path-id behind a PPtr to a GameObject or component (same file only)."""
        if not pptr or pptr.get("m_PathID", 0) == 0 or pptr.get("m_FileID", 0) != 0:
            return None
        pid = pptr["m_PathID"]
        return pid if pid in self.go else self.comp_go.get(pid)


# ============================================================================ 4. the room trunk
# Chain assembly, reproduced from the measured offline work. The one deliberate change against it is
# the `ActivateNextWave` exclusion in `_scan_rooms` (spec 4.3 I2): wave containers match the room
# regex, and on 7-1 `1 - Wave 1` / `1 - Wave 2` sorted as ordinal 1 and dragged `2 - Left Arena` out
# of place, shipping the order `1, 3, 4, 5, 2, 1, 2, 3, 4, 5`. Measured campaign-wide: exactly two
# objects are excluded by this rule and both are on 7-1.

RX_ORD = re.compile(r"^(\d+)([A-Za-z][A-Za-z0-9]?)?\s*-\s+(\D.*)$")
RX_BR = re.compile(r"^([A-Z])(\d{1,2})?(?:-([A-Z0-9]{1,2}))?\s*-\s+(\D.*)$")

BAD_ANC = re.compile(r"(HUD|Canvas|Hellmap|EventSystem|Panel|Intermission|Options|Tutorial Screen)", re.I)
BAD_LABEL = re.compile(r"\b(OLD|Unused|Deprecated|Template|Backup)\b", re.I)
BAD_GROUP = re.compile(r"(Track|Rail|Waypoint|Path Nodes|Checkpoints?$|NonStuff|Nonstuff)", re.I)

SECTION_WORDS = [("first", 1), ("second", 2), ("third", 3), ("fourth", 4), ("fifth", 5),
                 ("sixth", 6), ("seventh", 7), ("eighth", 8), ("ninth", 9), ("tenth", 10)]
MIN_DESC = 8

# ItemType, decompiled/ItemType.cs
ITEM_TYPES = {0: "None", 1: "SkullBlue", 2: "SkullRed", 3: "SkullGreen", 4: "Readable", 5: "Torch",
              6: "Soap", 7: "CustomKey1", 8: "CustomKey2", 9: "CustomKey3", 10: "Breakable"}


def parse_room_name(nm):
    """(branch, ordinal, suffix, label) for `<N>[suffix] - <Name>` / `<Branch><N> - <Name>`, else None."""
    m = RX_ORD.match(nm)
    if m:
        return ("", int(m.group(1)), (m.group(2) or ""), m.group(3).strip())
    m = RX_BR.match(nm)
    if m:
        return (m.group(1), int(m.group(2)) if m.group(2) else 0, (m.group(3) or ""), m.group(4).strip())
    return None


def suffix_rank(suf):
    if not suf:
        return 0
    v = 0
    for c in suf[:2]:
        v = v * 100 + (ord(c.upper()) - 64 if c.isalpha() else int(c))
    return v


def section_rank(name):
    low = name.lower()
    for w, n in SECTION_WORDS:
        if re.search(r"\b%s\b" % w, low):
            return n
    m = re.match(r"^\s*(\d+)", name)
    return int(m.group(1)) if m else None


def act_num(name):
    if not name or not name.startswith("Level "):
        return None
    parts = name[6:].split("-")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        return None


def is_successor(cur, tgt):
    if tgt is None or cur is None:
        return False
    return (tgt[0] == cur[0] and tgt[1] == cur[1] + 1) or (tgt[0] == cur[0] + 1 and tgt[1] == 1)


class RoomTrunk:
    """The ordered spawn-to-exit room chain for one level, before the guards run."""

    def __init__(self, lvl, sc):
        self.lvl = lvl
        self.sc = sc
        self.notes = []
        self.owner = {}
        self._scan_rooms()
        self._choose_exit()
        self._filter_rooms()
        self._synth_exit_room()
        self._scopes()
        self._scan_wiring()
        self._ordinal_edges()
        self._link_scopes()
        self._build()

    # -- rooms --------------------------------------------------------------------------------
    def _anc_names(self, t):
        sc, out = self.sc, []
        while t in sc.tf and sc.tf[t]["father"] in sc.tf:
            t = sc.tf[t]["father"]
            out.append(sc.go[sc.tf[t]["go"]]["name"])
        return out

    def _scan_rooms(self):
        sc = self.sc
        wave_go = {sc.go_of(pid) for pid in sc.comps("ActivateNextWave")}
        rooms = {}
        self.excluded_waves = []
        for g, v in sc.go.items():
            t = sc.go_tf.get(g)
            if t is None:
                continue
            p = parse_room_name(v["name"])
            if p is None or BAD_LABEL.search(v["name"]):
                continue
            if g in wave_go:                       # spec 4.3 I2: a wave container is not a room
                self.excluded_waves.append(v["name"])
                continue
            anc = self._anc_names(t)
            if any(BAD_ANC.search(a) for a in anc):
                continue
            ds = [d for d in sc.descendants(g) if d in sc.go_tf]
            pts = [sc.go_pos(d) for d in ds]
            pos = tuple(sc.go_pos(g))
            if pts:
                cen = tuple(sum(q[i] for q in pts) / len(pts) for i in range(3))
                bb = tuple((min(q[i] for q in pts), max(q[i] for q in pts)) for i in range(3))
            else:
                cen, bb = pos, tuple((pos[i], pos[i]) for i in range(3))
            rooms[g] = {"go": g, "name": v["name"], "branch": p[0], "ord": p[1], "suf": p[2],
                        "label": p[3], "pos": pos, "cen": cen, "bbox": bb, "n": len(ds),
                        "active": v["active"], "depth": len(anc),
                        "group": anc[0] if anc else "", "anc": anc,
                        "refs": collections.Counter(), "dropped": None}
        self.rooms = rooms
        self._reown()

    def _reown(self):
        self.owner = {}
        for g in self.rooms:
            if self.rooms[g].get("dropped"):
                continue
            for d in self.sc.descendants(g):
                cur = self.owner.get(d)
                if cur is None or self.rooms[cur]["depth"] < self.rooms[g]["depth"]:
                    self.owner[d] = g
            self.owner[g] = g

    def room_of(self, goid):
        if goid is None:
            return None
        g = self.owner.get(goid)
        return g if (g is not None and not self.rooms[g].get("dropped")) else None

    def room_of_point(self, p, max_m=250.0):
        best, bv = None, None
        for g, r in self.rooms.items():
            if r.get("dropped"):
                continue
            bb = r["bbox"]
            if all(bb[i][0] - 2 <= p[i] <= bb[i][1] + 2 for i in range(3)):
                vol = math.prod((bb[i][1] - bb[i][0] + 1) for i in range(3))
                if bv is None or vol < bv:
                    best, bv = g, vol
        if best is not None:
            return best
        bd = max_m
        for g, r in self.rooms.items():
            if r.get("dropped"):
                continue
            d = min(dist3(p, r["pos"]), dist3(p, r["cen"]))
            if d < bd:
                best, bd = g, d
        return best

    def _filter_rooms(self):
        """Throw out the things that parse as rooms but are not places.

        `refs` is still empty here (the wiring scan runs later), so the `empty` arm reduces to a
        descendant count. That is how the measured pipeline behaved and it is kept deliberately.
        """
        bypos = collections.Counter(tuple(round(v) for v in r["pos"]) for r in self.rooms.values())
        # An ordinal-named object carrying a CheckPoint is a rung whatever its size: 7-4 numbers its
        # checkpoints ("0 - Leg Checkpoint" .. "5 - Return Checkpoint") and has no numbered rooms at
        # all, so this is that level's entire route.
        cp_go = {self.sc.go_of(pid) for pid in self.sc.comps("CheckPoint")}
        for g, r in self.rooms.items():
            if g == self.exit_room:
                continue
            r["is_cp"] = g in cp_go
            if r["is_cp"]:
                continue
            why = None
            if r["n"] < MIN_DESC and not r["refs"]:
                why = "empty"                      # tram-track nodes
            elif BAD_GROUP.search(r["group"]):
                why = "container"                  # "Tram 1 Track", "NonStuff"
            elif bypos[tuple(round(v) for v in r["pos"])] >= 3 and r["n"] < 200:
                why = "stacked"                    # 7-4's Earthmover part list, all at one point
            if why:
                r["dropped"] = why
        self._reown()
        if self.exit_room is not None and self.rooms[self.exit_room].get("dropped"):
            self.rooms[self.exit_room]["dropped"] = None
            self._reown()

    # -- exit ---------------------------------------------------------------------------------
    def _choose_exit(self):
        """The real `FinalPit`, by the mod's own rule (CampaignObserver): skip rankless / second /
        fake pits and `-S` targets, prefer the pit whose target is the next level and is active."""
        sc = self.sc
        cur = act_num("Level " + self.lvl)
        cands = []
        for pid in sc.comps("FinalPit"):
            g = sc.go_of(pid)
            o = sc.read(pid)
            if o["rankless"] or o["secondPit"] or o["fakeEnd"]:
                continue
            t = (o.get("targetLevelName") or "").strip()
            if not t or t.upper().endswith("-S"):
                continue
            cands.append((g, t, sc.active_in_hierarchy(g)))
        chosen, best = None, 1 << 30
        for g, t, act in cands:
            rank = (0 if is_successor(cur, act_num(t)) else 2) + (0 if act else 1)
            if rank < best:
                best, chosen = rank, (g, t, act)
        self.exit = None
        self.exit_room = None
        self.exit_how = "none"
        if chosen is None:
            return
        g, t, act = chosen
        self.exit = {"go": g, "pos": tuple(sc.go_pos(g)), "target": t, "active": act}
        r = self.room_of(g)
        if r is not None:
            self.exit_room, self.exit_how = r, "ancestor"
            return
        r = self.room_of_point(self.exit["pos"], 400.0)
        if r is not None:
            self.exit_room, self.exit_how = r, "spatial"

    def _synth_exit_room(self):
        """The pit's room is sometimes not numbered at all (1-3's "Boss Arena", 8-4's "Pit"). If some
        Door / CheckPoint / wave names that ancestor as a room it is a real room and becomes the last
        rung; a spatial guess at a nearby numbered room is not good enough, because on 1-3 the nearest
        numbered room is on the far side of the level."""
        if self.exit is None or self.exit_how == "ancestor":
            return
        sc = self.sc
        named = set()
        for cls, fld in (("Door", "activatedRooms"), ("CheckPoint", "rooms"),
                         ("ActivateNextWave", "toActivate")):
            for pid in sc.comps(cls):
                try:
                    o = sc.read(pid)
                except Exception:
                    continue
                for r in (o.get(fld) or []):
                    rg = sc.ref_go(r)
                    if rg is not None:
                        named.add(rg)
        t = sc.go_tf[self.exit["go"]]
        while t in sc.tf:
            g = sc.tf[t]["go"]
            if g in named and self.owner.get(g) is None:
                ds = [d for d in sc.descendants(g) if d in sc.go_tf]
                pts = [sc.go_pos(d) for d in ds]
                pos = tuple(sc.go_pos(g))
                cen = tuple(sum(q[i] for q in pts) / len(pts) for i in range(3)) if pts else pos
                bb = tuple((min(q[i] for q in pts), max(q[i] for q in pts)) for i in range(3)) \
                    if pts else tuple((pos[i], pos[i]) for i in range(3))
                self.rooms[g] = {"go": g, "name": sc.go[g]["name"], "branch": "", "ord": 9999,
                                 "suf": "", "label": sc.go[g]["name"], "pos": pos, "cen": cen,
                                 "bbox": bb, "n": len(ds), "active": sc.go[g]["active"], "depth": 0,
                                 "group": "", "anc": [], "refs": collections.Counter(),
                                 "dropped": None, "is_cp": False, "unnumbered": True}
                self._reown()
                self.exit_room, self.exit_how = g, "unnumbered-room"
                self.notes.append("exit room %s is a referenced but unnumbered room" % sc.go[g]["name"])
                return
            t = sc.tf[t]["father"]

    # -- numbering scopes ---------------------------------------------------------------------
    def _scopes(self):
        """Group the rooms into numbering scopes; merge containers whose ordinals do not clash.

        Levels number their rooms 1..N but several split them across container objects. Two
        containers belong to the SAME numbering when their ordinal sets are disjoint (7-3: the root
        holds {1,2,3,12} and "Outdoors Areas" {4..11} -- one numbering, split for editor convenience)
        and to DIFFERENT numberings when they overlap (7-1: "Second Section" and "Third Section" both
        run 1..5 -- two numberings chained end to end).
        """
        live = [g for g, r in self.rooms.items() if not r.get("dropped")]
        buckets = collections.defaultdict(list)
        for g in live:
            buckets[(self.rooms[g]["group"], self.rooms[g]["branch"])].append(g)
        items = sorted(buckets.items(),
                       key=lambda kv: (-len(kv[1]), min(self.rooms[g]["ord"] for g in kv[1])))
        scopes = []
        for (grp, br), gs in items:
            ords = {(self.rooms[g]["ord"], suffix_rank(self.rooms[g]["suf"])) for g in gs}
            placed = False
            for s in scopes:
                if s["branch"] != br or (s["ords"] & ords):
                    continue
                s["ords"] |= ords
                s["rooms"] += gs
                s["groups"].append(grp)
                placed = True
                break
            if not placed:
                scopes.append({"branch": br, "ords": set(ords), "rooms": list(gs), "groups": [grp]})
        for s in scopes:
            s["rooms"].sort(key=lambda g: (self.rooms[g]["ord"], suffix_rank(self.rooms[g]["suf"]),
                                           self.rooms[g]["depth"]))
            s["srank"] = min([section_rank(x) for x in s["groups"] if section_rank(x) is not None],
                             default=None)
        self.scopes = scopes
        for i, s in enumerate(scopes):
            for g in s["rooms"]:
                self.rooms[g]["scope"] = i

    # -- authored wiring ----------------------------------------------------------------------
    def _add(self, a, b, why):
        if a is None or b is None or a == b:
            return
        if self.rooms[a].get("dropped") or self.rooms[b].get("dropped"):
            return
        self.edges[a].setdefault(b, why)
        self.rooms[b]["refs"][why] += 1

    def _scan_wiring(self):
        sc = self.sc
        self.edges = collections.defaultdict(dict)
        self.dirwire = []
        self.doors = []
        for pid in sc.comps("Door"):
            g = sc.go_of(pid)
            o = sc.read(pid)
            pos = tuple(sc.go_pos(g))
            rr = []
            for r in o["activatedRooms"]:
                x = self.room_of(sc.ref_go(r))
                if x is not None and x not in rr:
                    rr.append(x)
            here = self.room_of(g) or self.room_of_point(pos)
            self.doors.append({"pos": pos, "rooms": rr, "here": here, "locked": bool(o["locked"]),
                               "go": g})
            if len(rr) >= 2:
                # Order the rooms a door opens by their own numbering and link consecutively, so a
                # three-room door does not collapse three rungs into one. 1-1's final door lists
                # {1, 3, 12, 13} because the end room opens onto the start field.
                rr2 = sorted(rr, key=lambda g: (self.rooms[g].get("scope", 99), self.rooms[g]["ord"],
                                                suffix_rank(self.rooms[g]["suf"])))
                for i in range(1, len(rr2)):
                    self._add(rr2[i - 1], rr2[i], "door")
                    self._add(rr2[i], rr2[i - 1], "door")
                self.dirwire.append(("door_set", rr2))
            elif len(rr) == 1:
                self._add(here, rr[0], "door1")
                if here is not None:
                    self.dirwire.append(("door1", [here, rr[0]]))
        self.checkpoints = []
        for pid in sc.comps("CheckPoint"):
            g = sc.go_of(pid)
            o = sc.read(pid)
            pos = tuple(sc.go_pos(g))
            rr = []
            for r in o.get("rooms", []):
                x = self.room_of(sc.ref_go(r))
                if x is not None and x not in rr:
                    rr.append(x)
            here = self.room_of(g) or self.room_of_point(pos)
            self.checkpoints.append({"pos": pos, "rooms": rr, "here": here, "name": sc.go[g]["name"],
                                     "oname": parse_room_name(sc.go[g]["name"])})
            if rr:
                self._add(here, rr[0], "cp")
                for i in range(1, len(rr)):
                    self._add(rr[i - 1], rr[i], "cp")
                self.dirwire.append(("cp", ([here] if here is not None else []) + rr))
        self.waves = []
        for pid in sc.comps("ActivateNextWave"):
            g = sc.go_of(pid)
            o = sc.read(pid)
            pos = tuple(sc.go_pos(g))
            here = self.room_of(g) or self.room_of_point(pos)
            ta = []
            for r in o.get("toActivate", []):
                x = self.room_of(sc.ref_go(r))
                if x is not None and x not in ta:
                    ta.append(x)
            self.waves.append({"pos": pos, "here": here, "to": ta, "last": bool(o["lastWave"])})
            for x in ta:
                self._add(here, x, "wave")
                if here is not None:
                    self.dirwire.append(("wave", [here, x]))
        for cls, flds in (("ItemPlaceZone", ("activateOnSuccess",)),
                          ("GearCheckEnabler", ("toActivate",)),
                          ("ActivateArena", ("toActivate", "activateOnSuccess")),
                          ("ObjectActivator", ("toActivate", "objectsToActivate"))):
            for pid in sc.comps(cls):
                g = sc.go_of(pid)
                try:
                    o = sc.read(pid)
                except Exception:
                    continue
                here = self.room_of(g) or self.room_of_point(tuple(sc.go_pos(g)))
                for fld in flds:
                    for r in (o.get(fld) or []):
                        self._add(here, self.room_of(sc.ref_go(r)), "activator")

    def _ordinal_edges(self):
        for s in self.scopes:
            gs = s["rooms"]
            for i in range(1, len(gs)):
                self._add(gs[i - 1], gs[i], "ord")

    def _link_scopes(self):
        """Chain separate numberings end to end: by section word when the containers carry one,
        otherwise by which scope end is physically nearest the next scope's start."""
        if len(self.scopes) < 2:
            return
        ranked = sorted([s for s in self.scopes if s["srank"] is not None], key=lambda s: s["srank"])
        rest = [s for s in self.scopes if s["srank"] is None]
        order = ranked + rest
        for i in range(1, len(order)):
            a, b = order[i - 1], order[i]
            if not a["rooms"] or not b["rooms"]:
                continue
            if ranked and a in ranked and b in ranked:
                self._add(a["rooms"][-1], b["rooms"][0], "section")
                self.notes.append("section link %s -> %s" % (a["groups"], b["groups"]))
                continue
            best, bd = None, 1e18
            for x in a["rooms"][-3:]:
                for y in b["rooms"][:3]:
                    d = dist3(self.rooms[x]["pos"], self.rooms[y]["pos"])
                    if d < bd:
                        best, bd = (x, y), d
            if best:
                self._add(best[0], best[1], "spatial")
                self.notes.append("spatial link %s -> %s (%.0f m)" % (a["groups"], b["groups"], bd))

    # -- the chain ----------------------------------------------------------------------------
    def _attach_exit(self):
        """When the pit sits in no room at all, hang it off the last rung of the longest numbering so
        the ladder still terminates at the pit."""
        if self.exit_room is not None or self.exit is None or not self.scopes:
            return
        best = max(self.scopes, key=lambda s: len(s["rooms"]))
        tail = [g for g in best["rooms"] if not self.rooms[g].get("dropped")]
        if not tail:
            return
        self.exit_room = tail[-1]
        self.exit_how = "terminal-rung"
        self.notes.append("exit not inside any room; hung off last rung %s at %.0f m"
                          % (self.rooms[tail[-1]]["name"],
                             dist3(self.rooms[tail[-1]]["pos"], self.exit["pos"])))

    def _find_start(self, live):
        """The room the level starts in: rooms ahead of the player are switched OFF at load, so the
        lowest-numbered room active in the hierarchy is where play begins. `FirstRoom` is the root
        every level puts the Player prefab under, and its transform is the entry area -- which is what
        tells 5-3's two copies of room 1 apart (unrotated 10 m from it, rotated 325 m)."""
        cand = [g for g in live if self.sc.active_in_hierarchy(g) and not self.rooms[g].get("is_cp")]
        if not cand:
            cand = list(live)
        if not cand:
            return None
        anchor = (0.0, 0.0, 300.0)
        for g, v in self.sc.go.items():
            if v["name"] == "FirstRoom" and self.sc.go_tf.get(g) is not None:
                anchor = tuple(self.sc.go_pos(g))
                break
        return min(cand, key=lambda g: (self.rooms[g]["ord"], suffix_rank(self.rooms[g]["suf"]),
                                        dist3(self.rooms[g]["pos"], anchor)))

    def _scope_order(self):
        if len(self.scopes) < 2:
            return list(self.scopes)
        ranked = sorted([s for s in self.scopes if s["srank"] is not None], key=lambda s: s["srank"])
        if len(ranked) == len(self.scopes):
            return ranked
        start_s = next((s for s in self.scopes if self.start_room in s["rooms"]), None)
        exit_s = next((s for s in self.scopes if self.exit_room in s["rooms"]), None)
        order = []
        if start_s is not None:
            order.append(start_s)
        for s in ranked:
            if s not in order and s is not exit_s:
                order.append(s)
        if exit_s is not None and exit_s not in order:
            order.append(exit_s)
        for s in sorted(self.scopes, key=lambda s: -len(s["rooms"])):
            if s not in order:
                order.insert(max(0, len(order) - 1), s)
        return order

    def _chain_rooms(self, live):
        """The ordered room list. The BASE sequence is the numberings the level runs through end to
        end; any other numbering is a BRANCH spliced in where the base wires into it (8-3's
        "10 - Split Color Door" opens both colour paths, so both belong after rung 10 and not, as a
        plain scope concatenation puts them, before the level has started)."""
        order = self._scope_order()
        if not order:
            return []
        rooms_of = {id(s): [g for g in s["rooms"] if g in live] for s in order}
        base_scopes = [s for s in order if s["srank"] is not None
                       or self.start_room in s["rooms"] or self.exit_room in s["rooms"]]
        if not base_scopes:
            base_scopes = [order[-1]]
        main_set = {id(s) for s in base_scopes}
        base = []
        for s in base_scopes:
            base += rooms_of[id(s)]
        pos_in_base = {g: i for i, g in enumerate(base)}
        attach = collections.defaultdict(list)
        tail = []
        for s in order:
            if id(s) in main_set or not rooms_of[id(s)]:
                continue
            members = set(rooms_of[id(s)])
            best = None
            for a, bs in self.edges.items():
                if a not in pos_in_base:
                    continue
                if members & set(bs):
                    i = pos_in_base[a]
                    if best is None or i < best:
                        best = i
            if best is None:                       # also accept an edge FROM the branch INTO main
                for a in members:
                    for b in self.edges.get(a, {}):
                        if b in pos_in_base:
                            i = max(0, pos_in_base[b] - 1)
                            if best is None or i < best:
                                best = i
            if best is None:
                tail.append(s)
            else:
                attach[best].append(s)
                self.notes.append("branch %s spliced after rung %d (%s)"
                                  % (s["groups"] or s["branch"], best, self.rooms[base[best]]["name"]))
        out = []
        for i, g in enumerate(base):
            out.append(g)
            for s in attach.get(i, []):
                out += rooms_of[id(s)]
        for s in tail:
            out = rooms_of[id(s)] + out            # unattached numbering: assume it comes first
        return out

    def _build(self):
        """The ladder is the ORDER, not a graph distance: `rung` is the room's index in the chain
        (scope order, then ordinal, then suffix) and `hops = last_rung - rung`."""
        self._attach_exit()
        live = {g for g, r in self.rooms.items() if not r.get("dropped")}
        self.start_room = self._find_start(live)
        chain = self._chain_rooms(live)
        # Where does the exit's room sit in the numbering? Near the end: the rooms after it are side
        # content, cut them off. Near the START: the level LOOPS BACK (8-1's real ending is a bathroom
        # off room 2 of 19), and truncating there would throw the level away, so the exit room moves
        # to the end instead.
        self.loops_back = False
        if self.exit_room in chain:
            i = chain.index(self.exit_room)
            after = len(chain) - 1 - i
            if after and after > 0.25 * len(chain):
                self.loops_back = True
                chain = [g for g in chain if g != self.exit_room] + [self.exit_room]
                self.notes.append("loops back: exit room %s is rung %d of %d, moved to the end"
                                  % (self.rooms[self.exit_room]["name"], i, len(chain)))
            else:
                chain = chain[:i + 1]
        self.chain = chain

    def rows(self):
        """One raw rung per chain entry, in route order, with the numbering scope it came from."""
        return [{"name": self.rooms[g]["name"], "pos": list(self.rooms[g]["pos"]),
                 "scope": self.rooms[g].get("scope")} for g in self.chain]

    def checkpoint_positions(self):
        """Every `CheckPoint` transform in the scene, unfiltered -- the ground truth neither ordering
        rule reads, used for `checkpoints_within_60m` and `legs_witnessed`."""
        return [list(c["pos"]) for c in self.checkpoints]

    def gate_guard(self):
        """Layer 1, reproduced offline: does this level's LIVE door-gate ladder pass its own guard?

        The mod publishes one gate per `Door` that activates two or more rooms, keyed on the door's
        rounded position, with `hops` from a BFS over room adjacency starting at the room the pit sits
        in (`CampaignObserver.ScanGates`). `GateProgress._gates` then trusts the ladder only when
        `gates_ordered` -- the door graph found that goal room at all -- and at least
        `hops_min_frac` (0.5) of the phase-1 gates carry a `hops` value. A one-room altar door is a
        phase-2 gate and is excluded from the ratio, which is automatic here: a door activating fewer
        than two rooms is not a gate.

        A level that PASSES this never reads a route file -- `_rooms()` is only reached on the three
        `return []` arms -- so shipping one would be dead weight at best and a coverage change at
        worst. The emitter refuses to write a file for such a level.
        """
        sc = self.sc
        edges = collections.defaultdict(set)
        gates = []
        for pid in sc.comps("Door"):
            g = sc.go_of(pid)
            o = sc.read(pid)
            ids, keys = [], []
            for r in o["activatedRooms"]:
                rg = sc.ref_go(r)
                if rg is None or rg in ids:
                    continue
                ids.append(rg)
                k = pos_key(sc.go_pos(rg))
                if k not in keys:
                    keys.append(k)
            if len(ids) < 2:
                continue
            gates.append({"rooms": keys})
            for a in keys:
                for b in keys:
                    if a != b:
                        edges[a].add(b)
        allr = set()
        for gt in gates:
            allr.update(gt["rooms"])
        goal = None
        if self.exit is not None:
            t = sc.go_tf[self.exit["go"]]
            while True:
                k = pos_key(sc.world(t)[0])
                if k in allr:
                    goal = k
                    break
                f = sc.tf[t]["father"]
                if f not in sc.tf:
                    break
                t = f
        hops = {}
        if goal is not None:
            hops[goal] = 0
            q = collections.deque([goal])
            while q:
                r = q.popleft()
                for n in edges[r]:
                    if n not in hops:
                        hops[n] = hops[r] + 1
                        q.append(n)
        n_hops = sum(1 for gt in gates if any(r in hops for r in gt["rooms"]))
        ratio = (n_hops / len(gates)) if gates else 0.0
        return {"gates": len(gates), "with_hops": n_hops, "ratio": ratio,
                "gates_ordered": goal is not None,
                "pass": bool(gates) and goal is not None and ratio >= 0.5}

    def altar_doors(self):
        """[(door position, [item type])] for every door an `ItemPlaceZone` with a real
        `acceptedItemType` drives (`ItemPlaceZone.doors`, decompiled/ItemPlaceZone.cs). Diagnostic in
        S1-S2; stage S3 turns it into each rung's `gated_by`."""
        sc = self.sc
        by_door = {}
        for pid in sc.comps("ItemPlaceZone"):
            try:
                o = sc.read(pid)
            except Exception:
                continue
            item = ITEM_TYPES.get(o.get("acceptedItemType"), str(o.get("acceptedItemType")))
            if item == "None":
                continue
            for r in (o.get("doors") or []):
                g = sc.ref_go(r)
                if g is not None:
                    by_door.setdefault(g, set()).add(item)
        out = []
        for d in self.doors:
            items = by_door.get(d["go"])
            if items:
                out.append((list(d["pos"]), sorted(items)))
        return out


# ============================================================================ 5. R4: standability
# `_note_reached` is the only thing that advances `best_hops`, and it needs the player inside
# `_is_reached`'s cylinder: 8 m horizontal, 6 m vertical (a room rung ships `open: false`, margin
# 1.0). A room's transform is a prefab pivot, not a doorway the player walks through, so this has to
# be measured. Voxelise the collision geometry within +-14 m of the rung, extract cells with the
# player's capsule clearance free above the real surface, and report the nearest one in the cylinder
# metric. Control over 60 gates from ladders proven in game: 0 unreachable, dxz 0.4 m on every one.

ENV_LAYERS = frozenset((6, 7, 8, 24))     # LMD.Environment -- decompiled/LayerMaskDefaults.cs
CLS_BOX, CLS_MESHC, CLS_SPHERE, CLS_CAPSULE = 65, 64, 135, 136

# Geometry that stops being solid once the player reaches it: doors open, glass and breakables are
# shot through. Only the UP-FACING triangles of these are kept, because the floor slab shipped inside
# a door prefab still has to hold the player up.
OPENABLE = frozenset((
    "Door", "DoorController", "SubDoor", "BigDoor", "FinalDoor", "DoorOpener", "FinalDoorOpener",
    "BigDoorOpener", "DoorLock", "DoorUnlocker", "DoorBlocker", "Breakable", "Glass", "GlassBreaker",
    "LimboSwitchLock"))

_VFMT = {0: ("<f4", 4), 1: ("<f2", 2), 2: ("u1", 1), 3: ("i1", 1), 4: ("<u2", 2), 5: ("<i2", 2),
         6: ("u1", 1), 7: ("i1", 1), 8: ("<u2", 2), 9: ("<i2", 2), 10: ("<u4", 4), 11: ("<i4", 4)}

_BOX_IDX = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                     [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], dtype=np.int32)


def _trs(pos, rot, scale):
    """Unity TRS as a 4x4 in the row-vector convention: v_world = v_local @ M."""
    x, y, z, w = rot
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r = np.array([[1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)],
                  [2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)],
                  [2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)]], dtype=np.float64)
    m = np.eye(4)
    m[:3, :3] = np.diag(np.asarray(scale, dtype=np.float64)) @ r
    m[3, :3] = pos
    return m


def _box_tris(ctr, size):
    h = np.asarray(size, dtype=np.float64) / 2.0
    c = np.asarray(ctr, dtype=np.float64)
    v = np.array([[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1],
                  [-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64) * h + c
    return v[_BOX_IDX]


def _xform(tris, M):
    f = tris.reshape(-1, 3)
    return ((np.c_[f, np.ones(len(f))] @ M)[:, :3]).reshape(-1, 3, 3)


def decode_mesh(sf, obj, ress=None):
    """(verts Nx3 float32, tris Mx3 int32) in mesh-local space, or None."""
    vd = obj.get("m_VertexData") or {}
    n = vd.get("m_VertexCount", 0)
    if not n:
        return None
    chans = vd.get("m_Channels") or []
    if not chans or chans[0].get("dimension", 0) < 3:
        return None
    ch = chans[0]
    stream = ch["stream"]
    stride = 0
    for c in chans:
        if c["stream"] == stream and c["dimension"]:
            _fmt, es = _VFMT[c["format"]]
            stride = max(stride, c["offset"] + es * c["dimension"])
    stride = (stride + 3) & ~3
    sd = obj.get("m_StreamData") or {}
    if sd.get("size", 0) and sd.get("path"):
        if ress is None:
            return None
        blob = ress[sd["offset"]:sd["offset"] + sd["size"]]
    else:
        ds = vd.get("m_DataSize")
        if not (isinstance(ds, tuple) and ds[0] == "<bytes>"):
            return None
        blob = sf.d[ds[2]:ds[2] + ds[1]]
    base = 0                                          # streams are concatenated, each 16-aligned
    for s in range(stream):
        w = 0
        for c in chans:
            if c["stream"] == s and c["dimension"]:
                _fmt, es = _VFMT[c["format"]]
                w = max(w, c["offset"] + es * c["dimension"])
        base += ((((w + 3) & ~3) * n) + 15) & ~15
    fmt, es = _VFMT[ch["format"]]
    if base + stride * (n - 1) + ch["offset"] + es * 3 > len(blob) or stride == 0:
        return None
    raw = np.frombuffer(blob, dtype=np.uint8, count=stride * n, offset=base).reshape(n, stride)
    verts = raw[:, ch["offset"]:ch["offset"] + es * 3].copy().view(np.dtype(fmt)).reshape(n, 3) \
              .astype(np.float32)
    ib = obj.get("m_IndexBuffer")
    if not ib:
        return None
    ibb = bytes(bytearray(ib)) if isinstance(ib, list) else ib
    u32 = obj.get("m_IndexFormat", 0) == 1
    idt, isz = (np.uint32, 4) if u32 else (np.uint16, 2)
    tris = []
    for sm in obj.get("m_SubMeshes") or []:
        if sm.get("topology", 0) != 0:
            continue
        fb, ic, bv = sm["firstByte"], sm["indexCount"], sm.get("baseVertex", 0)
        if ic < 3 or fb + ic * isz > len(ibb):
            continue
        idx = np.frombuffer(ibb, dtype=idt, count=ic - ic % 3, offset=fb).astype(np.int64) + bv
        tris.append(idx.reshape(-1, 3))
    if not tris:
        return None
    t = np.concatenate(tris)
    t = t[(t >= 0).all(1) & (t < n).all(1)]
    return (verts, t.astype(np.int32)) if len(t) else None


class Collider:
    __slots__ = ("kind", "go", "name", "layer", "active", "M", "data")

    def __init__(self, kind, go, name, layer, active, M, data):
        self.kind, self.go, self.name, self.layer = kind, go, name, layer
        self.active, self.M, self.data = active, M, data


class LevelGeom:
    """World-space collision primitives for a scene, including rooms switched off at load: the
    serialized transforms are authored in their final placement, so inactive rooms give correct
    geometry. Player-relevant collision is the Unity layers in LMD.Environment; triggers and
    disabled colliders are excluded."""

    # A room referenced by any of these is part of the level the game actually runs.
    ROOM_REFS = (("Door", ("activatedRooms", "deactivatedRooms")), ("CheckPoint", ("rooms",)),
                 ("ActivateNextWave", ("toActivate",)), ("ActivateArena", ("toActivate", "rooms")),
                 ("FinalDoor", ("layers",)), ("ObjectActivator", ("events",)),
                 ("PlayerActivator", ("events",)))

    def __init__(self, sc, ress):
        self.sc = sc
        self.sf = sc.sf
        self.ress = ress
        self._wm, self._mesh, self._openable_cache, self._root_cache = {}, {}, {}, {}
        self.colliders, self.openable = [], []
        self.live_roots = self._resolve_rooms()
        self._collect()

    # -- variant resolution -------------------------------------------------------------------
    def root_of(self, goid):
        r = self._root_cache.get(goid)
        if r is not None:
            return r
        sc, chain, last = self.sc, [], goid
        t = sc.go_tf.get(goid)
        while t and t in sc.tf:
            last = sc.tf[t]["go"]
            chain.append(last)
            t = sc.tf[t]["father"]
        for g in chain:
            self._root_cache[g] = last
        return last

    def _resolve_rooms(self):
        """Top-level scene objects that belong to the level the game runs. ULTRAKILL ships alternative
        and abandoned versions of rooms in the same scene, in THE SAME WORLD SPACE (0-1 has
        '1 - Starting Room' vs '1Alt - Short Starting Room'); voxelising all of them fuses the variants
        into one solid block and seals the level. A root is kept when it is active at load or when some
        activation mechanism points into it."""
        sc = self.sc
        refd = set()
        for script, fields in self.ROOM_REFS:
            for pid in sc.comps(script):
                try:
                    o = sc.read(pid)
                except Exception:
                    continue
                for f in fields:
                    v = o.get(f)
                    if isinstance(v, list):
                        for e in v:
                            self._harvest(e, refd)
        roots = set()
        for goid in sc.go:
            tf = sc.go_tf.get(goid)
            if tf is None:
                continue
            if (sc.tf[tf]["father"] == 0 or sc.tf[tf]["father"] not in sc.tf) and sc.go[goid]["active"]:
                roots.add(goid)
        for x in refd:
            roots.add(self.root_of(x))
        return roots

    def _harvest(self, e, out, depth=0):
        """GameObject path-ids inside a serialized field (a PPtr, or a UnityEvent's call targets)."""
        if depth > 4:
            return
        if isinstance(e, dict):
            if "m_PathID" in e and "m_FileID" in e:
                t = self.sc.ref_go(e)
                if isinstance(t, int):
                    out.add(t)
                return
            for v in e.values():
                self._harvest(v, out, depth + 1)
        elif isinstance(e, list):
            for v in e[:400]:
                self._harvest(v, out, depth + 1)

    def is_openable(self, goid):
        sc, chain = self.sc, []
        t = sc.go_tf.get(goid)
        while t and t in sc.tf:
            g = sc.tf[t]["go"]
            c = self._openable_cache.get(g)
            if c is not None:
                for x in chain:
                    self._openable_cache[x] = c
                return c
            chain.append(g)
            if OPENABLE & set(sc.go_scripts(g)):
                for x in chain:
                    self._openable_cache[x] = True
                return True
            t = sc.tf[t]["father"]
        for x in chain:
            self._openable_cache[x] = False
        return False

    # -- world matrices (the full TRS chain: correct under nested rotation and non-uniform scale)
    def wm(self, tfid):
        m = self._wm.get(tfid)
        if m is not None:
            return m
        t = self.sc.tf[tfid]
        local = _trs(t["pos"], t["rot"], t["scale"])
        f = t["father"]
        m = local if (f == 0 or f not in self.sc.tf) else local @ self.wm(f)
        self._wm[tfid] = m
        return m

    def mesh(self, pid):
        if pid in self._mesh:
            return self._mesh[pid]
        try:
            r = decode_mesh(self.sf, self.sf.read(pid), self.ress)
        except Exception:
            r = None
        self._mesh[pid] = r
        return r

    def _collect(self):
        sc, sf = self.sc, self.sf
        for cls in (CLS_BOX, CLS_MESHC, CLS_SPHERE, CLS_CAPSULE):
            for pid in sc.by_class.get(cls, []):
                try:
                    o = sf.read(pid)
                except Exception:
                    continue
                if not o.get("m_Enabled", 1) or o.get("m_IsTrigger", 0):
                    continue
                go = sc.comp_go.get(pid)
                if go is None or go not in sc.go:
                    continue
                lay = sc.go[go]["layer"]
                if lay not in ENV_LAYERS:
                    continue
                if self.root_of(go) not in self.live_roots:
                    continue
                tfid = sc.go_tf.get(go)
                if tfid is None:
                    continue
                M = self.wm(tfid)
                act = sc.active_in_hierarchy(go)
                nm = sc.go[go]["name"]
                sink = self.openable if self.is_openable(go) else self.colliders
                if cls == CLS_BOX:
                    sink.append(Collider("box", go, nm, lay, act, M,
                                         (np.array(list(o["m_Center"].values()), dtype=np.float64),
                                          np.array(list(o["m_Size"].values()), dtype=np.float64))))
                elif cls == CLS_SPHERE:
                    sink.append(Collider("sphere", go, nm, lay, act, M,
                                         (np.array(list(o["m_Center"].values()), dtype=np.float64),
                                          o["m_Radius"])))
                elif cls == CLS_CAPSULE:
                    sink.append(Collider("capsule", go, nm, lay, act, M,
                                         (np.array(list(o["m_Center"].values()), dtype=np.float64),
                                          o["m_Radius"], o["m_Height"], o.get("m_Direction", 1))))
                else:
                    mp = o.get("m_Mesh") or {}
                    if mp.get("m_PathID", 0) == 0 or mp.get("m_FileID", 0) != 0:
                        continue
                    if self.mesh(mp["m_PathID"]) is None:
                        continue
                    sink.append(Collider("mesh", go, nm, lay, act, M, mp["m_PathID"]))

    def corners(self, c):
        """A coarse world-space point cloud bounding one collider, for AABB rejection."""
        if c.kind == "box":
            ctr, size = c.data
            s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)],
                         dtype=np.float64)
            pts = ctr + s * (size / 2.0)
        elif c.kind == "sphere":
            ctr, r = c.data
            s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)],
                         dtype=np.float64)
            pts = ctr + s * r
        elif c.kind == "capsule":
            ctr, r, hgt, d = c.data
            ax = np.zeros(3)
            ax[d] = 1.0
            half = max(hgt / 2.0, r)
            s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)],
                         dtype=np.float64)
            pts = np.vstack([ctr + ax * half + s * r, ctr - ax * half + s * r])
        else:
            pts = self.mesh(c.data)[0].astype(np.float64)
        return (np.c_[pts, np.ones(len(pts))] @ c.M)[:, :3]

    def triangles(self, c):
        """World-space triangles for one collider (Nx3x3)."""
        if c.kind == "mesh":
            v, t = self.mesh(c.data)
            return ((np.c_[v.astype(np.float64), np.ones(len(v))] @ c.M)[:, :3])[t]
        if c.kind == "box":
            ctr, size = c.data
            return _xform(_box_tris(ctr, size), c.M)
        if c.kind == "sphere":
            ctr, r = c.data
            return _xform(_box_tris(ctr, np.array([2 * r, 2 * r, 2 * r])), c.M)
        ctr, r, hgt, d = c.data
        size = np.full(3, 2 * r)
        size[d] = max(hgt, 2 * r)
        return _xform(_box_tris(ctr, size), c.M)


def sample_triangles(tris, spacing):
    """Dense barycentric samples on a triangle soup: (points Nx3, normal_y N)."""
    if not len(tris):
        return np.zeros((0, 3)), np.zeros(0)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    e1, e2 = b - a, c - a
    nrm = np.cross(e1, e2)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-12
    a, e1, e2, nrm, ln = a[ok], e1[ok], e2[ok], nrm[ok], ln[ok]
    if not len(a):
        return np.zeros((0, 3)), np.zeros(0)
    ny = nrm[:, 1] / ln
    longest = np.maximum(np.maximum(np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1)),
                         np.linalg.norm(e2 - e1, axis=1))
    n = np.clip(np.ceil(longest / spacing).astype(np.int64), 1, 2000)
    pts, nys = [], []
    order = np.argsort(n)
    n_s = n[order]
    uniq = np.unique(n_s)
    bounds = np.searchsorted(n_s, uniq, side="left")
    for k, nv in enumerate(uniq):
        lo = bounds[k]
        hi = bounds[k + 1] if k + 1 < len(bounds) else len(n_s)
        idx = order[lo:hi]
        g = np.arange(nv + 1) / nv
        uu, vv = np.meshgrid(g, g, indexing="ij")
        m = (uu + vv) <= 1.0 + 1e-9
        u, v = uu[m], vv[m]
        p = (a[idx][:, None, :] + e1[idx][:, None, :] * u[None, :, None]
             + e2[idx][:, None, :] * v[None, :, None])
        pts.append(p.reshape(-1, 3))
        nys.append(np.repeat(ny[idx], len(u)))
    return np.concatenate(pts), np.concatenate(nys)


class HeightField:
    """Sparse voxel heightfield: triangles are surface-sampled into solid voxels, up-facing samples
    also mark floor voxels. Voxel key = (ix*nz + iz)*ny + iy."""

    def __init__(self, lo, hi, cs=VOX_CS, ch=VOX_CH):
        self.cs, self.ch = cs, ch
        self.origin = np.asarray(lo, dtype=np.float64) - np.array([cs, ch, cs])
        span = (np.asarray(hi) - self.origin) + np.array([cs, ch, cs]) * 2
        self.nx = int(np.ceil(span[0] / cs)) + 1
        self.ny = int(np.ceil(span[1] / ch)) + 1
        self.nz = int(np.ceil(span[2] / cs)) + 1
        self._s, self._y, self._f, self._fy = [], [], [], []

    def key(self, ijk):
        ijk = np.atleast_2d(ijk)
        return (ijk[:, 0] * self.nz + ijk[:, 2]) * self.ny + ijk[:, 1]

    def unkey(self, k):
        k = np.atleast_1d(k)
        iy = k % self.ny
        r = k // self.ny
        return np.stack([r // self.nz, iy, r % self.nz], axis=1)

    def centre(self, ijk):
        return self.origin + (np.atleast_2d(ijk) + 0.5) * np.array([self.cs, self.ch, self.cs])

    def add(self, pts, nys, floor_cos):
        if not len(pts):
            return
        d = (pts - self.origin) / np.array([self.cs, self.ch, self.cs])
        ijk = np.floor(d).astype(np.int64)
        good = ((ijk >= 0).all(1) & (ijk[:, 0] < self.nx) & (ijk[:, 1] < self.ny)
                & (ijk[:, 2] < self.nz))
        ijk, nys, ys = ijk[good], nys[good], pts[good, 1]
        k = self.key(ijk)
        self._s.append(k)
        self._y.append(ys)
        up = nys >= floor_cos
        self._f.append(k[up])
        self._fy.append(ys[up])

    def finish(self):
        """Solid and floor voxel keys, plus the CONTINUOUS surface height inside each voxel:
        discretisation otherwise costs up to a cell of head-room at each end, which is enough to
        reject a real 3.7 m corridor for a 3.5 m player."""
        if not self._s:
            self.solid = self.floor = np.zeros(0, np.int64)
            self.solid_ymin = self.floor_ytop = np.zeros(0)
            return
        k = np.concatenate(self._s)
        y = np.concatenate(self._y)
        self.solid, inv = np.unique(k, return_inverse=True)
        self.solid_ymin = np.full(len(self.solid), np.inf)
        np.minimum.at(self.solid_ymin, inv, y)
        fk = np.concatenate(self._f)
        fy = np.concatenate(self._fy)
        if len(fk):
            self.floor, finv = np.unique(fk, return_inverse=True)
            ytop = np.full(len(self.floor), -np.inf)
            np.maximum.at(ytop, finv, fy)
            keep = np.isin(self.floor, self.solid, assume_unique=True)
            self.floor, self.floor_ytop = self.floor[keep], ytop[keep]
        else:
            self.floor, self.floor_ytop = np.zeros(0, np.int64), np.zeros(0)
        self._s = self._y = self._f = self._fy = []

    def ceiling_above(self, keys):
        """Continuous y of the lowest solid surface strictly above each key, in the same column."""
        col = keys // self.ny
        pos = np.searchsorted(self.solid, keys, side="right")
        out = np.full(len(keys), np.inf)
        valid = pos < len(self.solid)
        idx = np.clip(pos, 0, max(len(self.solid) - 1, 0))
        same = valid & ((self.solid[idx] // self.ny) == col)
        out[same] = self.solid_ymin[idx][same]
        return out

    def standable(self, clearance=PLAYER_H):
        if not len(self.floor):
            return np.zeros(0, np.int64)
        ceil = self.ceiling_above(self.floor)
        return self.floor[(ceil - self.floor_ytop) >= clearance]


def standability(geom, rows):
    """R4 per rung: the nearest standable cell in `_is_reached`'s own cylinder metric."""
    if not rows:
        return []
    pts = np.array([r["pos"] for r in rows], dtype=np.float64)
    lo, hi = pts.min(0) - PROBE_PAD, pts.max(0) + PROBE_PAD
    fc = math.cos(math.radians(SLOPE_DEG))
    keep = []
    for c in list(geom.colliders) + list(geom.openable):
        try:
            cor = geom.corners(c)
        except Exception:
            continue
        clo, chi = cor.min(0), cor.max(0)
        if (chi < lo).any() or (clo > hi).any():
            continue
        if any((chi >= p - PROBE_PAD).all() and (clo <= p + PROBE_PAD).all() for p in pts):
            keep.append(c)
    open_ids = {id(c) for c in geom.openable}

    out = []
    for r, p in zip(rows, pts):
        blo, bhi = p - PROBE_PAD, p + PROBE_PAD
        hf = HeightField(blo, bhi)
        ntri = 0
        for c in keep:
            try:
                cor = geom.corners(c)
            except Exception:
                continue
            if (cor.max(0) < blo).any() or (cor.min(0) > bhi).any():
                continue
            try:
                tris = geom.triangles(c)
            except Exception:
                continue
            if not len(tris):
                continue
            m = (tris.max(1) >= blo).all(1) & (tris.min(1) <= bhi).all(1)
            tris = tris[m]
            if not len(tris):
                continue
            ntri += len(tris)
            sp, ny = sample_triangles(tris, SAMPLE_SPACING)
            if id(c) in open_ids:                    # a door holds the player up but does not wall
                up = ny >= fc
                sp, ny = sp[up], ny[up]
            hf.add(sp, ny, fc)
        hf.finish()
        st = hf.standable()
        if not len(st):
            out.append({"name": r["name"], "tris": ntri, "stand": 0,
                        "dxz": None, "dy": None, "reach": False})
            continue
        ctr = hf.centre(hf.unkey(st))
        ytop = hf.floor_ytop[np.isin(hf.floor, st)]
        dxz = np.hypot(ctr[:, 0] - p[0], ctr[:, 2] - p[2])
        dy = np.abs((ytop + EYE_OFFSET) - p[1])      # the player transform is the capsule centre
        inside = (dxz <= REACH_H) & (dy <= REACH_V)
        j = int(np.argmin(dxz + 0.001 * dy))
        out.append({"name": r["name"], "tris": ntri, "stand": int(len(st)),
                    "dxz": round(float(dxz[j]), 1), "dy": round(float(dy[j]), 1),
                    "reach": bool(inside.any()), "n_inside": int(inside.sum())})
    return out


# ============================================================================ 6. the guards
# Order of operations: I6 -> T -> R4 -> I2 -> R1 -> I3 -> R2 (spec 4.3). R2 is judged LAST, on what
# ships: the revision-1 files stored a tour ratio for a chain that the budget cap then cut, and 8-3's
# stored 1.264 described 28 rungs while the 24 that shipped scored 1.421, over R2's own threshold.

RX_G_ORD = re.compile(r"^(\d+)([A-Za-z][A-Za-z0-9]?)?\s*-\s+(.*)$")
RX_G_BR = re.compile(r"^([A-Z])(\d{1,2})\s*-\s+(.*)$")
RX_SECRET = re.compile(r"\b(secret|bonus)\b", re.I)


def lead(name):
    """A rung's own name: the part before any `+` an I2 merge concatenated."""
    return name.split(" + ")[0].strip()


def name_kind(name):
    """(kind, ordinal, suffix) for a rung's leading name: "ord", "branch" or "other"."""
    m = RX_G_ORD.match(name)
    if m:
        return ("ord", int(m.group(1)), (m.group(2) or ""))
    m = RX_G_BR.match(name)
    if m:
        return ("branch", int(m.group(2)), m.group(1))
    return ("other", None, "")


def is_optional(name):
    """I6: the name rules that say a rung is an optional area rather than the route.

    The `<N>S` suffix occurs exactly three times campaign-wide -- `6S - P Door` (3-1),
    `1S - P Door` (6-2), `10S - Secret Arena` (8-3) -- and all three are optional. The Prime Sanctum
    door needs every level in its layer at P rank and is never on the route to the exit.
    """
    nm = lead(name)
    _k, _o, suf = name_kind(nm)
    low = nm.lower()
    return bool(RX_SECRET.search(nm) or suf.upper().startswith("S")
                or "p door" in low or "prime" in low)


def guard_i6(rows):
    keep = [r for r in rows if not is_optional(r["name"])]
    return keep, [r["name"] for r in rows if is_optional(r["name"])]


def guard_trunk(rows):
    """T: collapse parallel sets onto the rung they hang off.

    A parallel set is (a) a same-ordinal group of >= 2 SUFFIXED rooms inside one numbering scope, or
    (b) the whole branch-prefix region when the level carries more than one branch prefix.
    `GateProgress` keeps one `best_hops` and `_note_reached` collapses it to the minimum over every
    rung, so whichever member the agent enters first pays the whole set at once and the rest go dark.
    Coarser, never wrong. A lone suffix is a chain, not a star, and is kept.

    Rule (a) additionally requires the members to be DISTINCT PLACES -- at least `SEP` apart, the same
    16 m `_is_reached` cannot tell apart. A group packed inside one reach cylinder is one place, not a
    choice: arriving marks every member at once whatever order the agent took, so T's justification
    does not apply to it and I2 is what collapses it, into a single rung carrying both names.
    Measured over all 33 levels there is exactly one such group -- 5-2's `1A - Opening` and
    `1B - Second Rock`, both at (0, -10, 300), spread 0.0 m -- and it is also the only same-ordinal
    group with no unsuffixed rung at its ordinal to hang off, so without this clause 5-2 loses the
    rung at its own spawn and its first rung jumps 129 m to `2 - Fort`. Every other group in the
    campaign is 41 m or wider and is collapsed.
    """
    keep, dropped, kept_notes = list(rows), [], []
    prefixes = {name_kind(lead(r["name"]))[2] for r in rows
                if name_kind(lead(r["name"]))[0] == "branch"}
    if len(prefixes) > 1:                                      # (b) multi-branch region
        out = []
        for r in keep:
            k, _o, pref = name_kind(lead(r["name"]))
            if k == "branch":
                dropped.append(("branch:%s" % pref, r["name"]))
            else:
                out.append(r)
        keep = out
    groups = {}
    for r in keep:                                             # (a) same-ordinal suffix groups
        k, ordn, suf = name_kind(lead(r["name"]))
        if k == "ord" and suf:
            groups.setdefault((r.get("scope"), ordn), []).append(r)
    doomed = []
    for (scope, ordn), members in groups.items():
        if len(members) < 2:
            continue
        spread = max(dist3(a["pos"], b["pos"]) for a, b in itertools.combinations(members, 2))
        if spread < SEP:                                       # one place, not a choice: leave it to I2
            kept_notes.append("scope %s ordinal %d is one place, spread %.1f m: %s -- left to I2"
                              % (scope, ordn, spread,
                                 " + ".join(r["name"] for r in members)))
            continue
        doomed += members
        for r in members:
            dropped.append(("parallel:scope%s.ord%d" % (scope, ordn), r["name"]))
    if doomed:
        ids = {id(r) for r in doomed}
        keep = [r for r in keep if id(r) not in ids]
    return keep, dropped, kept_notes


def guard_i2(rows):
    """I2: merge any pair inside 16 m -- `_is_reached` would mark both from one arrival. The closest
    pair is merged into the LATER (lower-hops) rung, names concatenated, until nothing is inside."""
    rr = [dict(r) for r in rows]
    merged = []
    while True:
        worst = None
        for i, j in itertools.combinations(range(len(rr)), 2):
            d = dist3(rr[i]["pos"], rr[j]["pos"])
            if d < SEP and (worst is None or d < worst[0]):
                worst = (d, i, j)
        if worst is None:
            return rr, merged
        d, i, j = worst                                        # i is earlier in route order
        merged.append((round(d, 1), rr[i]["name"], rr[j]["name"]))
        rr[j]["name"] = rr[i]["name"] + " + " + rr[j]["name"]
        rr.pop(i)


def guard_i3(rows):
    """I3: at most MAX_RUNGS, dropping the cheapest detour first, never the first or the last."""
    rr = list(rows)
    dropped = []
    while len(rr) > MAX_RUNGS:
        best, bi = None, None
        for i in range(1, len(rr) - 1):
            det = (dist3(rr[i - 1]["pos"], rr[i]["pos"]) + dist3(rr[i]["pos"], rr[i + 1]["pos"])
                   - dist3(rr[i - 1]["pos"], rr[i + 1]["pos"]))
            if best is None or det < best:
                best, bi = det, i
        dropped.append(rr[bi]["name"])
        rr.pop(bi)
    return rr, dropped


def tour_ratio(rows):
    """The ladder's own polyline over a greedy nearest-neighbour tour of the same points from the
    same start. A numbering that is not route order looks like a random order and scores high."""
    if len(rows) < 3:
        return 1.0
    pts = [r["pos"] for r in rows]
    poly = sum(dist3(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    left, cur, tour = list(range(1, len(pts))), 0, 0.0
    while left:
        j = min(left, key=lambda k: dist3(pts[cur], pts[k]))
        tour += dist3(pts[cur], pts[j])
        left.remove(j)
        cur = j
    return poly / tour if tour else 1.0


def polyline(rows):
    return sum(dist3(rows[i]["pos"], rows[i + 1]["pos"]) for i in range(len(rows) - 1))


def load_overrides(path=None):
    """`rung_overrides.json` as `{level short form: [entry, ...]}`, or {} when there is none.

    Missing is the normal case and is silent. A file that exists but cannot be read is NOT silent:
    the whole point of the mechanism is that a regeneration keeps a measured fix, so losing it to a
    typo has to be visible. See the OVERRIDES_NAME comment for the format and the three rules.
    """
    path = Path(path) if path else (ROOT / "ultrakill_ai" / "routes" / OVERRIDES_NAME)
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf8"))
    return {k: v for k, v in doc.items() if not k.startswith("_")}


def apply_overrides(lvl, rows, overrides, geom=None):
    """Rule 1: move the named rungs, last, before anything is measured. Returns (rows, notes).

    Matching is by the rung's room NAME, not by index, hops or position: names survive a
    regeneration, and the other three do not. Every refusal is a note rather than an exception,
    because one bad entry must not stop the other 13 levels from being rebuilt -- `--validate` is
    what turns the notes into a non-zero exit.

    `geom` re-runs R4 (standability) and I2 (16 m separation) on the moved rungs, since going last
    means the new point did not pass either on the way through the pipeline. Either failing reverts
    that one override and leaves the generated position in place. `geom=None` (a --no-probe run)
    keeps the separation check, which needs no geometry, and skips only R4.
    """
    notes = []
    by_name = {r["name"]: r for r in rows}
    moved = []
    for entry in overrides.get(lvl, ()):
        name, want, was = entry.get("name"), entry.get("pos"), entry.get("was")
        row = by_name.get(name)
        if row is None:
            notes.append("override REFUSED: %s has no rung named %r (it may have been dropped by a "
                         "guard, or the room was renamed)" % (lvl, name))
            continue
        if not (isinstance(want, list) and len(want) == 3 and isinstance(was, list) and len(was) == 3):
            notes.append("override REFUSED: %s %r needs a 3-element `pos` and `was`" % (lvl, name))
            continue
        drift = dist3(row["pos"], was)
        if drift > OVERRIDE_WAS_TOL_M:
            # Rule 3. The hand-picked point was chosen against geometry that has since moved, so it
            # is no longer known to be inside the room it names.
            notes.append("override REFUSED: %s %r now builds at %s, %.1f m from the recorded `was` "
                         "%s -- re-measure the override against the new geometry"
                         % (lvl, name, [round(v, 1) for v in row["pos"]], drift, was))
            continue
        moved.append((row, list(row["pos"]), [float(v) for v in want], entry.get("why", "")))
        row["pos"] = [float(v) for v in want]

    # I2 on the moved rungs against every other rung, and R4 on the moved rungs alone.
    reach = {}
    if geom is not None and moved:
        reach = {r["name"]: ok for r, ok in
                 zip([m[0] for m in moved],
                     [x["reach"] for x in standability(geom, [m[0] for m in moved])])}
    for row, before, want, why in moved:
        clash = next((o for o in rows if o is not row and dist3(o["pos"], row["pos"]) < SEP), None)
        bad = None
        if clash is not None:
            bad = "it lands %.1f m from %r, inside I2's %.0f m separation" \
                  % (dist3(clash["pos"], row["pos"]), clash["name"], SEP)
        elif reach.get(row["name"]) is False:
            bad = "R4 finds no standable cell inside its reach cylinder there"
        if bad:
            row["pos"] = before
            notes.append("override REFUSED: %s %r -> %s: %s" % (lvl, row["name"], want, bad))
        else:
            notes.append("override applied: %s %r %s -> %s (%s)"
                         % (lvl, row["name"], [round(v, 1) for v in before], want,
                            why or "no reason recorded"))
    return rows, notes


# ============================================================================ 7. one level

ALTAR_NEAR = 25.0       # a lock is "on the trunk" when its door is this close to a leg


def load_scene(lvl):
    """(Scene, .resS blob) for one campaign level, or (None, None) when the bundle is not shipped."""
    path = bundles_dir() / ("campaign_scenes_level%s.bundle" % lvl.lower())
    if not path.exists():
        return None, None
    nodes = unpack_bundle(path)
    main = [k for k in nodes if not k.endswith((".resS", ".resource", ".sharedAssets"))]
    if not main:
        return None, None
    return Scene(nodes[main[0]], script_map()), nodes.get(main[0] + ".resS")


def build_level(lvl, probe=True, analyse_gates=False, overrides=None):
    """Run the whole pipeline on one level and return a report, with `doc` set when it ships.

    `analyse_gates` measures the trunk on a level layer 1 already routes. No file is ever written for
    such a level -- it would be dead weight, since `_rooms()` is only reached on the three `return []`
    arms -- but the measurement is what a "replace the gate ladder here" question needs.
    """
    t0 = time.time()
    sc, ress = load_scene(lvl)
    if sc is None:
        return {"lvl": lvl, "ships": False, "why": ["bundle not shipped in this build"]}
    trunk = RoomTrunk(lvl, sc)
    raw = trunk.rows()
    gate = trunk.gate_guard()
    rep = {"lvl": lvl, "raw": len(raw), "start_room": None, "exit_how": trunk.exit_how,
           "excluded_waves": trunk.excluded_waves, "notes": trunk.notes, "gate": gate, "why": []}
    if trunk.start_room is not None:
        rep["start_room"] = trunk.rooms[trunk.start_room]["name"]
    if trunk.exit is None:
        rep["ships"] = False
        rep["why"].append("no usable FinalPit")
        return rep
    collapsed_ship = lvl in COLLAPSED_SHIP
    if gate["pass"]:
        # Layer 1 already routes this level and never reads a file (spec 1). Measured, not assumed.
        # The exception is a COLLAPSED ladder on the allow-list above, whose file ships as an
        # alternative the run may select; the level is still reported as signal "gates", because
        # that is what drives it unless `prefer_route_when_collapsed` is turned on.
        rep["signal"] = "gates"
        rep["why"].append("layer 1: %d/%d gates carry hops (%.3f)"
                          % (gate["with_hops"], gate["gates"], gate["ratio"]))
        if not (analyse_gates or collapsed_ship):
            rep["ships"] = False
            return rep

    rows, drop_i6 = guard_i6(raw)                                        # I6
    rows, drop_t, trunk_notes = guard_trunk(rows)                        # T
    rep["notes"] = list(rep["notes"]) + trunk_notes
    if probe and rows:                                                   # R4
        reach = standability(LevelGeom(sc, ress), rows)
    else:
        reach = [{"name": r["name"], "reach": True, "stand": None, "dxz": None, "dy": None}
                 for r in rows]
    bad = {r["name"] for r in reach if not r["reach"]}
    drop_r4 = [r["name"] for r in rows if r["name"] in bad]
    rows = [r for r in rows if r["name"] not in bad]
    rows, merges = guard_i2(rows)                                        # I2
    r1 = len(rows) >= MIN_RUNGS                                          # R1
    rows, drop_i3 = guard_i3(rows)                                       # I3
    # Rule 1: the manual overrides go LAST, after every guard and before every measurement below, so
    # the tour ratio, the two "a/b" counts and `last_rung_to_exit_m` all describe what ships -- and so
    # that an override can never change WHICH rungs ship or their order, only where one of them sits.
    # Going last does mean the moved point skipped R4 and I2 on the way past, so both are re-checked
    # on the moved rungs alone and a failure REVERTS that one override: a rung the player cannot stand
    # at, or one that has been moved inside a neighbour's reach cylinder, is worse than the bug.
    rows, override_notes = apply_overrides(lvl, rows, overrides or {},
                                           geom=LevelGeom(sc, ress) if probe else None)
    rep["notes"] = list(rep["notes"]) + override_notes
    rep["override_refused"] = [n for n in override_notes if "REFUSED" in n]
    tour = tour_ratio(rows)                                              # R2
    r2 = tour <= MAX_TOUR

    if not r1:
        rep["why"].append("R1 fewer than %d rungs (%d)" % (MIN_RUNGS, len(rows)))
    if not r2:
        rep["why"].append("R2 tour ratio %.3f > %.2f" % (tour, MAX_TOUR))

    exit_pos = [round(v, 1) for v in trunk.exit["pos"]]
    cps = trunk.checkpoint_positions()
    pos = [[round(v, 1) for v in r["pos"]] for r in rows]
    pts = pos + [exit_pos]
    gaps = [dist3(pos[i], pos[i + 1]) for i in range(len(pos) - 1)]
    legs_wit = sum(1 for i in range(len(pts) - 1)
                   if any(seg_dist(c, pts[i], pts[i + 1]) <= LEG_NEAR for c in cps))
    cp_near = sum(1 for c in cps if pos and min(dist3(c, p) for p in pos) <= CP_NEAR)
    locks = []
    for dpos, items in trunk.altar_doors():
        if len(pts) < 2:
            continue
        d, leg = min((seg_dist(dpos, pts[i], pts[i + 1]), i) for i in range(len(pts) - 1))
        if d <= ALTAR_NEAR:
            locks.append({"leg": leg, "m": round(d, 1), "items": items,
                          "pos": [round(v, 1) for v in dpos]})
    locks.sort(key=lambda x: (x["leg"], x["m"]))

    rep.update({
        "ships": bool(r1 and r2 and (not gate["pass"] or collapsed_ship)),
        "rungs": len(rows),
        "drop_i6": drop_i6, "drop_trunk": drop_t, "drop_r4": drop_r4, "drop_i3": drop_i3,
        "merges": merges,
        "tour": round(tour, 3),
        "med_gap": round(statistics.median(gaps), 1) if gaps else 0.0,
        "max_gap": round(max(gaps), 1) if gaps else 0.0,
        "poly": round(polyline(rows), 1),
        "last_leg": round(dist3(pos[-1], exit_pos), 1) if pos else None,
        "legs_witnessed": "%d/%d" % (legs_wit, max(len(pts) - 1, 0)),
        "checkpoints_within_60m": "%d/%d" % (cp_near, len(cps)),
        "reach": reach,
        "unreachable": sorted(bad),
        "locks": locks,
        "secs": round(time.time() - t0, 1),
    })
    if not rep["ships"]:
        return rep

    n = len(rows)
    rep["doc"] = {
        "level": "Level %s" % lvl,
        "version": VERSION,
        "source": SOURCE,
        "exit": {"pos": exit_pos, "target": trunk.exit["target"]},
        "start_room": rep["start_room"],
        "trunk_collapsed": [nm for _why, nm in drop_t],
        "tour_ratio": round(tour, 3),
        "checkpoints_within_60m": rep["checkpoints_within_60m"],
        "legs_witnessed": rep["legs_witnessed"],
        "last_rung_to_exit_m": rep["last_leg"],
        "rungs": [{"key": "%d,%d,%d" % tuple(int(round(v)) for v in pos[k]),
                   "pos": pos[k],
                   "hops": n - 1 - k,
                   "name": r["name"],
                   "gated_by": [],
                   "open": False, "locked": False, "active": True}
                  for k, r in enumerate(rows)],
    }
    return rep


# ============================================================================ 8. report + CLI

TIER = {"0-1": "A", "0-3": "A", "0-4": "A", "2-1": "A", "2-2": "A", "3-1": "A", "4-1": "A",
        "0-2": "B", "1-1": "B", "1-2": "B", "2-3": "B", "5-1": "B", "5-3": "B", "8-2": "B",
        "0-5": "C", "2-4": "C", "3-2": "C", "4-2": "C", "5-2": "C", "6-2": "C", "7-4": "C",
        "1-3": "D", "1-4": "D", "4-3": "D", "4-4": "D", "6-1": "D", "7-3": "D", "8-1": "D",
        "8-4": "D", "5-4": "E", "7-1": "E", "7-2": "E", "8-3": "E"}


def write_docs(reports, out_dir, dry_run=False):
    """Write one file per shipping level and remove the file of a level that stopped shipping."""
    out_dir = Path(out_dir)
    written, removed = [], []
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    for rep in reports:
        path = out_dir / ("route_%s.json" % safe_name("Level %s" % rep["lvl"]))
        if rep.get("doc"):
            if not dry_run:
                tmp = path.with_suffix(".json.tmp")
                with open(tmp, "w", encoding="utf8", newline="\n") as f:
                    json.dump(rep["doc"], f, separators=(",", ":"))
                    f.write("\n")
                os.replace(tmp, path)
            written.append(path)
        elif path.exists():
            if not dry_run:
                path.unlink()
            removed.append(path)
    return written, removed


def print_table(reports):
    print()
    print("%-5s %-4s %-11s %-5s %-7s %-7s %-6s %-9s %-9s %-8s %s"
          % ("lvl", "tier", "signal", "rung", "med gap", "max gap", "tour", "legs wit.",
             "last->pit", "cp<=60", "what breaks it"))
    for r in reports:
        if "tour" not in r or r.get("last_leg") is None:
            print("%-5s %-4s %-11s %-5s %-7s %-7s %-6s %-9s %-9s %-8s %s"
                  % (r["lvl"], TIER.get(r["lvl"], "?"), r.get("signal", "exit vector"),
                     r.get("rungs", "-"), "-", "-", "-", "-", "-", "-",
                     "; ".join(r["why"]) or "-"))
            continue
        locks = ", ".join("leg%d %s" % (x["leg"], "/".join(x["items"])) for x in r["locks"])
        note = ("stops at a lock: " + locks) if locks else "reaches the exit"
        if not r["ships"]:
            note = "%s [not shipped: %s]" % (note, "; ".join(r["why"]))
        print("%-5s %-4s %-11s %-5d %-7.0f %-7.0f %-6.3f %-9s %-9.0f %-8s %s"
              % (r["lvl"], TIER.get(r["lvl"], "?"),
                 "rooms" if r["ships"] else r.get("signal", "exit vector"),
                 r["rungs"], r["med_gap"], r["max_gap"], r["tour"], r["legs_witnessed"],
                 r["last_leg"], r["checkpoints_within_60m"], note))


def shape_of(rep):
    """The `EXPECTED_SHAPE` row a level's report describes."""
    return (rep["rungs"], rep["med_gap"], rep["max_gap"], rep["tour"],
            rep["legs_witnessed"], rep["checkpoints_within_60m"], rep["last_leg"])


def shape_diff(want, got):
    """The fields of one `EXPECTED_SHAPE` row that moved, as readable strings."""
    out = []
    for i, (w, g) in enumerate(zip(want, got)):
        if i in (0, 4, 5):                      # rungs, and the two "a/b" counts
            same = w == g
        elif i == 3:                            # tour
            same = isinstance(g, (int, float)) and abs(w - g) <= SHAPE_TOL_TOUR
        else:                                   # the three distances, stored to 0.1 m
            same = isinstance(g, (int, float)) and abs(w - g) <= SHAPE_TOL_M
        if same:
            continue
        fmt = SHAPE_FMT[i]
        out.append("%s %s -> %s" % (SHAPE_FIELDS[i], fmt % w,
                                    (fmt % g) if isinstance(g, (int, float, str)) else g))
    return out


def print_validate(reports, expect_shipped=True):
    """The acceptance run of spec 7.5: the connectivity report, plus the checks that must hold.

    Returns the number of failures, so a regeneration that quietly changes coverage exits non-zero.
    """
    fails = []
    ship = [r for r in reports if r.get("ships")]
    # A refused override means a measured fix was silently dropped, which is exactly the regeneration
    # failure the file exists to prevent. It is a failure even on a level that still ships.
    for r in reports:
        for note in r.get("override_refused", ()):
            fails.append(note)
    print("\n--- layer 1: the live gate ladder's own guard, reproduced offline ---")
    for r in reports:
        g = r.get("gate")
        if not g:
            continue
        print("%-5s %-4s gates %2d  hops %2d  ratio %.3f  ordered %-5s  %s"
              % (r["lvl"], TIER.get(r["lvl"], "?"), g["gates"], g["with_hops"], g["ratio"],
                 g["gates_ordered"], "PASS -> layer 1" if g["pass"] else "fail -> layer 2/3"))
    n_gates = sum(1 for r in reports if (r.get("gate") or {}).get("pass"))
    print("layer 1 routes %d levels" % n_gates)
    if n_gates != 18:
        fails.append("the gates guard passes on %d levels, not the measured 18" % n_gates)

    print("\n--- R4: per-rung standability (shipped rungs; the R4 column is what it dropped) ---")
    n_ok = n_all = n_rung = 0
    for r in ship:
        # A rung I2 merged carries both names, and both places were probed, so match on each piece.
        pieces = set()
        for x in r["doc"]["rungs"]:
            pieces.update(p.strip() for p in x["name"].split(" + "))
        rows = [x for x in r["reach"] if x["name"] in pieces]
        inside = [x for x in rows if x["reach"]]
        n_ok += len(inside)
        n_all += len(rows)
        n_rung += len(r["doc"]["rungs"])
        worst = max((x["dxz"] for x in rows if x["dxz"] is not None), default=None)
        print("%-5s %2d/%-2d standable   worst dxz %-8s   dropped by R4: %s"
              % (r["lvl"], len(inside), len(rows), ("%.1f m" % worst) if worst is not None else "-",
                 ", ".join(r["drop_r4"]) or "none"))
    print("shipped: %d rungs, %d probed places (an I2 merge puts two in one rung), %d standable"
          % (n_rung, n_all, n_ok))
    if n_ok != n_all:
        fails.append("R4 is a guard: every shipped rung must be standable, %d/%d" % (n_ok, n_all))

    print("\n--- connectivity: leg witness (a checkpoint within %.0f m of the leg) ---" % LEG_NEAR)
    tw = tl = 0
    for r in ship:
        w, l = (int(x) for x in r["legs_witnessed"].split("/"))
        tw += w
        tl += l
        print("%-5s %-8s legs   tour %-6.3f   polyline %-7.0f m   last leg %-7.0f m"
              % (r["lvl"], r["legs_witnessed"], r["tour"], r["poly"], r["last_leg"]))
    if tl:
        print("legs witnessed: %d/%d (%.0f%%)" % (tw, tl, 100.0 * tw / tl))

    print("\n--- I2 merges and I3 drops ---")
    for r in ship:
        if r["merges"] or r["drop_i3"]:
            print("%-5s merges %s   cap dropped %s"
                  % (r["lvl"], r["merges"] or "-", r["drop_i3"] or "-"))
    if not any(r["merges"] or r["drop_i3"] for r in ship):
        print("none")

    print("\n--- locks on the trunk (stage S3 closes these) ---")
    for r in ship:
        print("%-5s %s" % (r["lvl"], "; ".join(
            "leg%d %s @%.0fm" % (x["leg"], "/".join(x["items"]), x["m"]) for x in r["locks"]) or "none"))

    print("\n--- invariants over the emitted documents ---")
    for r in ship:
        doc = r["doc"]
        rungs = doc["rungs"]
        hops = [x["hops"] for x in rungs]
        if hops != list(range(len(rungs) - 1, -1, -1)):
            fails.append("%s I1: hops are not n-1..0: %s" % (r["lvl"], hops))
        for x in rungs:
            if x["key"] != "%d,%d,%d" % tuple(int(round(v)) for v in x["pos"]):
                fails.append("%s key %s does not match pos %s" % (r["lvl"], x["key"], x["pos"]))
            if x["open"] or x["locked"] or not x["active"]:
                fails.append("%s %s: open/locked must be false and active true" % (r["lvl"], x["name"]))
            if is_optional(x["name"]):
                fails.append("%s I6: optional area shipped: %s" % (r["lvl"], x["name"]))
        for i, j in itertools.combinations(range(len(rungs)), 2):
            d = dist3(rungs[i]["pos"], rungs[j]["pos"])
            if d < SEP:
                fails.append("%s I2: %s and %s are %.1f m apart"
                             % (r["lvl"], rungs[i]["name"], rungs[j]["name"], d))
        if len(rungs) > MAX_RUNGS:
            fails.append("%s I3: %d rungs" % (r["lvl"], len(rungs)))
        if len(rungs) < MIN_RUNGS:
            fails.append("%s R1: %d rungs" % (r["lvl"], len(rungs)))
        repro = tour_ratio([{"pos": x["pos"]} for x in rungs])
        if abs(repro - doc["tour_ratio"]) > 5e-4:
            fails.append("%s R2: stored %.3f but the shipped pos list scores %.3f"
                         % (r["lvl"], doc["tour_ratio"], repro))
        if doc["tour_ratio"] > MAX_TOUR:
            fails.append("%s R2: %.3f > %.2f" % (r["lvl"], doc["tour_ratio"], MAX_TOUR))
        for x in rungs:
            if lead(x["name"]) == "Pit":
                if x["hops"] != 0:
                    fails.append("%s: a rung named Pit must be hops 0, not %d" % (r["lvl"], x["hops"]))
                elif dist3(x["pos"], doc["exit"]["pos"]) > PIT_MAX_M:
                    fails.append("%s: the Pit rung is %.0f m from the pit, over %.0f"
                                 % (r["lvl"], dist3(x["pos"], doc["exit"]["pos"]), PIT_MAX_M))
        by_ord = {}
        prefs = set()
        for x in rungs:
            k, ordn, suf = name_kind(lead(x["name"]))
            if k == "ord":
                by_ord.setdefault(ordn, set()).add(suf)
            elif k == "branch":
                prefs.add(suf)
        for ordn, sufs in by_ord.items():
            if len([s for s in sufs if s]) >= 2:
                fails.append("%s T: ordinal %d ships suffixes %s"
                             % (r["lvl"], ordn, sorted(s for s in sufs if s)))
        if len(prefs) > 1:
            fails.append("%s T: %d branch prefixes ship: %s" % (r["lvl"], len(prefs), sorted(prefs)))
    print("checked %d documents, %d rungs" % (len(ship), sum(len(r["doc"]["rungs"]) for r in ship)))

    # The shape of each ladder, not just the set of them. A --no-probe run fails here by design:
    # R4 drops four rungs and is part of the measured shape.
    print("\n--- shape: each shipped ladder against its recorded row (risk 3) ---")
    for r in ship:
        want = EXPECTED_SHAPE.get(r["lvl"])
        if want is None:
            print("%-5s no row in EXPECTED_SHAPE" % r["lvl"])
            fails.append("%s ships a route with no row in EXPECTED_SHAPE: add one with a measured "
                         "reason, or this level's ladder can change silently" % r["lvl"])
            continue
        moved = shape_diff(want, shape_of(r))
        print("%-5s %s" % (r["lvl"], "unchanged" if not moved else "CHANGED: " + "; ".join(moved)))
        if moved:
            fails.append("%s: the ladder changed shape (%s). Re-aiming a level is never a detail: "
                         "check the new trunk by hand, then edit EXPECTED_SHAPE and "
                         "tests/test_route_files.py's EXPECTED_LADDERS with the reason"
                         % (r["lvl"], "; ".join(moved)))
    if tuple(EXPECTED_SHAPE) != EXPECTED_SHIPPED:
        fails.append("EXPECTED_SHAPE and EXPECTED_SHIPPED disagree: %s vs %s"
                     % (" ".join(EXPECTED_SHAPE), " ".join(EXPECTED_SHIPPED)))

    got = tuple(r["lvl"] for r in ship)
    if expect_shipped and got != EXPECTED_SHIPPED:
        fails.append("the shipped set changed: expected %s, got %s"
                     % (" ".join(EXPECTED_SHIPPED), " ".join(got)))

    print()
    if fails:
        print("FAIL (%d)" % len(fails))
        for f in fails:
            print("  - %s" % f)
    else:
        print("PASS: %d route files, %d rungs, %d/%d places standable, %d/%d legs witnessed"
              % (len(ship), n_rung, n_ok, n_all, tw, tl))
    return len(fails)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", nargs="*", default=None,
                    help='levels to rebuild, short form ("0-5 7-1"); default every shipped level')
    ap.add_argument("--out", default=None,
                    help="output directory (default python/ultrakill_ai/routes)")
    ap.add_argument("--validate", action="store_true",
                    help="also print the acceptance report of spec 7.5 -- the invariants, the "
                         "shipped set and each ladder's shape against EXPECTED_SHAPE -- and exit "
                         "non-zero on a failure")
    ap.add_argument("--dry-run", action="store_true", help="measure and report, write nothing")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the voxel standability pass (R4). For debugging the chain only: R4 is "
                         "a guard, and skipping it can ship a rung the agent can never reach. It "
                         "drops four rungs across the campaign, so --validate's shape check fails "
                         "with it off")
    ap.add_argument("--analyse-gates-levels", action="store_true",
                    help="also measure the room trunk on the levels layer 1 already routes. Never "
                         "writes a file for them; this is the measurement a 'would the trunk be "
                         "better here?' question needs (spec risk 5, 1-1)")
    ap.add_argument("--overrides", default=None,
                    help="the manual rung-position overrides file (default "
                         "python/ultrakill_ai/routes/" + OVERRIDES_NAME + "). They are applied after "
                         "every guard and before every measurement; see the OVERRIDES_NAME comment")
    ap.add_argument("--json", default=None, help="write the full per-level report to this file")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else (ROOT / "ultrakill_ai" / "routes")
    levels = args.levels if args.levels else levels_in_order()
    partial = bool(args.levels)
    if args.no_probe and not args.out and not args.dry_run:
        # R4 is a guard, not a report: without it 5-2 ships a rung with no standable cell within
        # +-40 m and can never target its own pit. Refuse to overwrite the committed data with it off.
        ap.error("--no-probe cannot write the committed files; add --dry-run or --out <dir>")

    overrides = load_overrides(args.overrides)
    reports = []
    for lvl in levels:
        rep = build_level(lvl, probe=not args.no_probe, analyse_gates=args.analyse_gates_levels,
                          overrides=overrides)
        state = ("rooms %d" % rep["rungs"]) if rep.get("ships") else rep.get("signal", "exit vector")
        print("%-5s %-13s %5.1fs  %s"
              % (lvl, state, rep.get("secs", 0.0), "; ".join(rep["why"]) or ""))
        sys.stdout.flush()
        reports.append(rep)

    print_table(reports)
    ship = [r for r in reports if r.get("ships")]
    n_gates = sum(1 for r in reports if r.get("signal") == "gates")
    print("\ngates %d | rooms %d | exit vector only %d   (today: gates 18, nothing 15)"
          % (n_gates, len(ship), len(reports) - len(ship) - n_gates))
    print("%d route files, %d rungs" % (len(ship), sum(r["rungs"] for r in ship)))

    written, removed = write_docs(reports, out_dir, dry_run=args.dry_run)
    if args.dry_run:
        print("dry run: nothing written")
    else:
        total = sum(p.stat().st_size for p in written)
        print("wrote %d files to %s, %.1f KB total" % (len(written), out_dir, total / 1024.0))
        for p in removed:
            print("removed %s (level no longer ships a route)" % p.name)

    if args.json:
        with open(args.json, "w", encoding="utf8") as f:
            json.dump([{k: v for k, v in r.items() if k != "doc"} for r in reports], f, indent=1)

    rc = 0
    if args.validate:
        rc = 1 if print_validate(reports, expect_shipped=not partial) else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())

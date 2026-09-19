
# FULL-DOMAIN 3D RECTANGULAR EXPERIMENT SIMULATION
# PHYSICAL TANK V1 -- Legacy FEniCS / DOLFIN 2019.1

# This is the COMPLETE rectangular tank, not a quarter model:
#   x = -0.400 ... +0.400 m
#   y = -0.200 ... +0.200 m
#   z =  0.000 ... +0.450 m
# There are no symmetry boundaries at x=0 or y=0.  Those are interior
# planes, so the plume can wander, flap and impinge asymmetrically.

# Fresh run + robust restart design:
#   * uses a new output root, so no old 600 x 300 x 200 mm run is read
#   * always regenerates the deterministic shared-facet mesh
#   * NEVER reloads a mesh from HDF5
#   * checkpoints physical FE values keyed by DOF coordinates
#     rather than relying on DOLFIN's parallel vector numbering
#   * validates the reconstructed state before continuing
#   * re-submit the SAME job/script to continue automatically

# Physics/numerics retained from the validated H(div) 3D run:
#   * physical 0.80 x 0.40 x 0.45 m rectangular tank
#   * 2 mm circular central nozzle (20 cc/min by default)
#   * monolithic Taylor-Hood P2/P1 Oseen solve
#   * Boussinesq buoyancy
#   * lowest-order H(div) Raviart-Thomas reconstruction for scalar advection
#   * conservative implicit DG0 upwind concentration transport
#   * molecular diffusivity D = 1e-7 m^2/s
#   * deterministic, flux-preserving 3D inlet imperfection
#   * dt = 0.001 s by default

# Runtime overrides (examples):
#   FLOW_RATE_CC_MIN=40 T_GLOBAL=20 srun python3 full_rectangular_3d_physical.py
#   INLET_PERTURBATION=0  srun python3 full_rectangular_3d_physical.py


from dolfin import *
from petsc4py import PETSc
from mpi4py import MPI as PY_MPI

import glob
import fcntl
import json
import math
import os
import sys
import time

import numpy as np



# MPI / FENICS SETTINGS


comm = MPI.comm_world
rank = MPI.rank(comm)
nproc = MPI.size(comm)
pycomm = PY_MPI.COMM_WORLD

set_log_level(LogLevel.WARNING)

# DG interior-facet integrals require ghosted cells in parallel.
# IMPORTANT: this is why we regenerate the mesh on every job and
# never attempt to read a checkpoint mesh from HDF5.
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["quadrature_degree"] = 4



# PHYSICAL PARAMETERS


L_tank = 0.800
W_tank = 0.400
H_tank = 0.450

x_min = -0.5 * L_tank
x_max =  0.5 * L_tank
y_min = -0.5 * W_tank
y_max =  0.5 * W_tank
z_min = 0.0
z_max = H_tank

d_nozzle = 0.002
a_nozzle = 0.5 * d_nozzle

# The same file can run each experimental flow rate without being copied.
Q_cc_per_min = float(os.environ.get("FLOW_RATE_CC_MIN", "20.0"))
Q_in = Q_cc_per_min * 1.0e-6 / 60.0

# A very small, deterministic non-axisymmetric imperfection prevents a
# perfectly centred numerical source from artificially preserving symmetry.
# It is tangential to the inlet plane, so it contributes exactly zero normal
# volume flux. The physical baseline is zero; set it to 0.005 only for an
# asymmetry sensitivity run. The forcing is deterministic in absolute physical
# time, hence restart-safe.
inlet_perturbation = float(os.environ.get("INLET_PERTURBATION", "0.0"))
perturbation_period_x = 0.37
perturbation_period_y = 0.61

nu = 1.0e-6
D = 1.0e-7
g = 9.81

rho_fresh = 999.3
rho_salt = 1005.5
beta = (rho_salt - rho_fresh) / rho_salt



# TIME / OUTPUT SETTINGS


dt_default = min(1.0e-3, 1.0e-3 * 20.0 / Q_cc_per_min)
dt = float(os.environ.get("DT", "{:.16g}".format(dt_default)))
T_global = float(os.environ.get("T_GLOBAL", "20.0"))
ramp_time = 0.5

# Full-domain 3D is expensive. Each scheduler submission advances a bounded
# amount and writes restart files throughout. Re-submit the same job to carry
# on. Override only after timing the first segment on the target partition.
segment_duration = float(os.environ.get("SEGMENT_DURATION", "0.5"))

# Diagnostics every 0.1 s; ParaView output every 0.25 s.
diagnostic_every = 100
write_every = 250

# Full restart checkpoint every 0.1 s.
checkpoint_every = 100
keep_checkpoints = 2

# Also checkpoint by wall clock. This prevents a short scheduler wall limit
# from killing an expensive full-domain job before 0.1 physical seconds have
# elapsed. The check occurs between completed timesteps.
checkpoint_wall_seconds = float(
    os.environ.get("CHECKPOINT_WALL_SECONDS", "1200.0")
)

# A completely new root for each flow rate. Old reduced-domain runs are
# deliberately ignored and different experimental cases cannot collide.
flow_tag = ("{:g}".format(Q_cc_per_min)).replace(".", "p")
perturbation_tag = ("{:g}".format(inlet_perturbation)).replace(".", "p")
dt_tag = ("{:g}".format(dt)).replace(".", "p")
default_output_root = (
    "results_3D_full_physical_Q{}ccmin_eps{}_dt{}".format(
        flow_tag,
        perturbation_tag,
        dt_tag,
    )
)
script_directory = os.path.dirname(os.path.abspath(__file__))
requested_output_root = os.environ.get("OUTPUT_ROOT", default_output_root)
if os.path.isabs(requested_output_root):
    OUTPUT_ROOT = requested_output_root
else:
    OUTPUT_ROOT = os.path.join(script_directory, requested_output_root)
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoint")
SEGMENTS_DIR = os.path.join(OUTPUT_ROOT, "segments")
DIAGNOSTICS_PATH = os.path.join(OUTPUT_ROOT, "diagnostics.csv")
CONFIGURATION_PATH = os.path.join(OUTPUT_ROOT, "configuration.json")

# Safety thresholds.
velocity_abort_threshold = 1.0
scalar_abort_margin = 2.0e-2

# Restart reconstruction tolerances.
coord_decimals = 13
fingerprint_rtol = 1.0e-8
fingerprint_atol = 1.0e-12

# Explicit state-evolution identity. Change this string whenever the weak forms,
# boundary recipe, element families/orders, or refinement recipe change.
solver_schema = "full_rectangular_physical_v1_schema_20260902a"
mesh_recipe = (
    "box_40x20x30;ceiling_0p090;plume_0p020_0p010;"
    "nozzle_0p006_0p0035_0p002_0p002"
)

if Q_cc_per_min <= 0.0:
    raise ValueError("FLOW_RATE_CC_MIN must be positive.")
if (
    dt <= 0.0
    or T_global <= 0.0
    or segment_duration <= 0.0
    or checkpoint_wall_seconds <= 0.0
):
    raise ValueError(
        "DT, T_GLOBAL, SEGMENT_DURATION and CHECKPOINT_WALL_SECONDS "
        "must be positive."
    )
if not (0.0 <= inlet_perturbation <= 0.05):
    raise ValueError("INLET_PERTURBATION must lie between 0 and 0.05.")
if not (
    x_min < 0.0 < x_max
    and y_min < 0.0 < y_max
    and abs(x_max + x_min) < 1.0e-15
    and abs(y_max + y_min) < 1.0e-15
):
    raise RuntimeError("The requested full tank must span both x and y axes.")







def root_print(*args, **kwargs):
    if rank == 0:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


root_print("[startup] MPI ranks =", nproc)
root_print("[startup] output root =", OUTPUT_ROOT)







def global_entity_count(mesh, dim):
    try:
        return int(mesh.num_entities_global(dim))
    except Exception:
        return int(MPI.sum(comm, mesh.num_entities(dim)))



def print_mesh_info(mesh, label):
    nv = global_entity_count(mesh, 0)
    nc = global_entity_count(mesh, mesh.topology().dim())
    hmin_global = MPI.min(comm, mesh.hmin())
    hmax_global = MPI.max(comm, mesh.hmax())

    if rank == 0:
        print("")
        print("----------------------------------------")
        print(label)
        print("----------------------------------------")
        print("Global vertices :", nv)
        print("Global cells    :", nc)
        print("Global hmin     :", hmin_global)
        print("Global hmax     :", hmax_global)
        sys.stdout.flush()



def mark_refinement_region(mesh, radial_cutoff, vertical_cutoff, expansion=0.70):
    markers = MeshFunction("bool", mesh, mesh.topology().dim())
    markers.set_all(False)

    marked_local = 0

    for cell in cells(mesh):
        mp = cell.midpoint()
        xx = mp.x()
        yy = mp.y()
        zz = mp.z()
        rr = math.sqrt(xx * xx + yy * yy)
        hcell = cell.h()

        radial_test = rr <= radial_cutoff + expansion * hcell

        if vertical_cutoff >= H_tank:
            vertical_test = True
        else:
            vertical_test = zz <= vertical_cutoff + expansion * hcell

        if radial_test and vertical_test:
            markers[cell] = True
            marked_local += 1

    marked_global = MPI.sum(comm, marked_local)
    return markers, marked_global



def mark_ceiling_layer(mesh, layer_depth, expansion=0.70):
    """Mark a full-width layer below the rigid ceiling for refinement."""
    markers = MeshFunction("bool", mesh, mesh.topology().dim())
    markers.set_all(False)

    z_cutoff = H_tank - layer_depth
    marked_local = 0

    for cell in cells(mesh):
        if cell.midpoint().z() >= z_cutoff - expansion * cell.h():
            markers[cell] = True
            marked_local += 1

    return markers, MPI.sum(comm, marked_local)



def smooth_ramp(t_value, T_ramp):
    if t_value <= 0.0:
        return 0.0
    if t_value >= T_ramp:
        return 1.0

    xi = t_value / T_ramp
    return 3.0 * xi**2 - 2.0 * xi**3



# CREATE OUTPUT DIRECTORIES


directory_error = None
if rank == 0:
    try:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(SEGMENTS_DIR, exist_ok=True)
    except Exception as exc:
        directory_error = "Could not create output directories: {}".format(exc)

directory_error = pycomm.bcast(directory_error, root=0)
if directory_error is not None:
    raise RuntimeError(directory_error)

MPI.barrier(comm)

# One writer per physical case. Advisory flock is released automatically if
# Slurm kills the process, unlike a stale lock directory.
lock_handle = None
lock_error = None

if rank == 0:
    try:
        lock_path = os.path.join(OUTPUT_ROOT, ".writer.lock")
        lock_handle = open(lock_path, "a+")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(
            "job={} pid={} started={}\n".format(
                os.environ.get("SLURM_JOB_ID", "interactive"),
                os.getpid(),
                time.time(),
            )
        )
        lock_handle.flush()
    except Exception as exc:
        lock_error = "Could not acquire the case writer lock: {}".format(exc)

lock_error = pycomm.bcast(lock_error, root=0)
if lock_error is not None:
    raise RuntimeError(lock_error)



# DETECT RESTART METADATA BEFORE MESH GENERATION



def _read_npz_metadata(path):
    with np.load(path, allow_pickle=False) as data:
        if "meta_json" not in data:
            raise RuntimeError("checkpoint contains no metadata")
        required = {
            "meta_json",
            "ux_coords", "ux_values",
            "uy_coords", "uy_values",
            "uz_coords", "uz_values",
            "p_coords", "p_values",
            "c_coords", "c_values",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(
                "checkpoint is missing arrays: " + ", ".join(missing)
            )
        return json.loads(str(data["meta_json"].item()))



def find_latest_checkpoint_root():
    """Return newest complete physical-tank checkpoint + metadata on rank 0."""

    latest_json = os.path.join(CHECKPOINT_DIR, "latest.json")

    candidates = []

    # Preferred: atomic latest pointer.
    if os.path.exists(latest_json):
        try:
            with open(latest_json, "r") as f:
                latest = json.load(f)
            path = os.path.join(CHECKPOINT_DIR, latest["file"])
            if os.path.isfile(path):
                candidates.append(path)
        except Exception:
            pass

    # Fallback: scan newest-to-oldest.
    all_state_paths = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, "state_step_*.npz")),
        reverse=True,
    )

    for path in all_state_paths:
        if path not in candidates:
            candidates.append(path)

    for path in candidates:
        try:
            meta = _read_npz_metadata(path)
            if int(meta.get("restart_format_version", -1)) != 4:
                continue
            return path, meta
        except Exception:
            continue

    if all_state_paths:
        raise RuntimeError(
            "Checkpoint files exist, but none has a complete compatible "
            "restart-format-4 payload. Refusing to start clean over them."
        )

    return None, None


if rank == 0:
    try:
        restart_path, restart_meta = find_latest_checkpoint_root()
        restart_discovery_error = None
    except Exception as exc:
        restart_path, restart_meta = None, None
        restart_discovery_error = str(exc)
else:
    restart_path, restart_meta = None, None
    restart_discovery_error = None

restart_discovery_error = pycomm.bcast(restart_discovery_error, root=0)
restart_path = pycomm.bcast(restart_path, root=0)
restart_meta = pycomm.bcast(restart_meta, root=0)

if restart_discovery_error is not None:
    raise RuntimeError(restart_discovery_error)

if restart_path is None:
    root_print("")
    root_print("================================================")
    root_print("RESTART STATUS - PHYSICAL TANK V1")
    root_print("================================================")
    root_print("No compatible checkpoint found.")
    root_print("Starting a CLEAN simulation from t = 0.")
else:
    root_print("")
    root_print("================================================")
    root_print("RESTART STATUS - PHYSICAL TANK V1")
    root_print("================================================")
    root_print("Checkpoint found:", restart_path)
    root_print("step =", restart_meta["step"])
    root_print("time =", restart_meta["time"], "s")
    root_print("Mesh will be regenerated; no HDF5 mesh will be read.")



# GENERATE RECTANGULAR BOX MESH


root_print("")
root_print("================================================")
root_print("GENERATING 3D RECTANGULAR EXPERIMENTAL MESH")
root_print("================================================")
root_print("Tank length  =", L_tank, "m")
root_print("Tank width   =", W_tank, "m")
root_print("Tank height  =", H_tank, "m")
root_print("Nozzle dia.  =", d_nozzle, "m")

# Approximately 20 mm x 20 mm x 15 mm in the remote bulk. Local refinement
# resolves the 2 mm inlet and the plume column; full-width ceiling refinement
# is essential because the stratified layer is not confined to the centreline.
nx_base = 40
ny_base = 20
nz_base = 30

mesh = BoxMesh.create(
    comm,
    [
        Point(x_min, y_min, z_min),
        Point(x_max, y_max, z_max),
    ],
    [nx_base, ny_base, nz_base],
    CellType.Type.tetrahedron,
)

print_mesh_info(mesh, "INITIAL RECTANGULAR MESH")

ceiling_refinement_passes = [
    ("full ceiling layer refinement 1", 0.090),
]

for name, layer_depth in ceiling_refinement_passes:
    markers, number_marked = mark_ceiling_layer(mesh, layer_depth)

    root_print("")
    root_print(name)
    root_print("cells marked =", number_marked)

    mesh = refine(mesh, markers)
    print_mesh_info(mesh, name)

refinement_passes = [
    ("central plume refinement 1", 0.020, H_tank),
    ("central plume refinement 2", 0.010, H_tank),
    ("near-nozzle refinement 1", 0.0060, 0.025),
    ("near-nozzle refinement 2", 0.0035, 0.012),
    ("near-nozzle refinement 3", 0.0020, 0.006),
    ("near-nozzle refinement 4", 0.0020, 0.004),
]

for name, radial_cutoff, vertical_cutoff in refinement_passes:
    markers, number_marked = mark_refinement_region(
        mesh,
        radial_cutoff,
        vertical_cutoff,
    )

    root_print("")
    root_print(name)
    root_print("cells marked =", number_marked)

    mesh = refine(mesh, markers)
    print_mesh_info(mesh, name)

print_mesh_info(mesh, "FINAL 3D MESH")

mesh_vertices_global = global_entity_count(mesh, 0)
mesh_cells_global = global_entity_count(mesh, mesh.topology().dim())



# BOUNDARY MARKERS

# Match the validated production run exactly: classify EXTERIOR FACETS by
# their midpoint after the final mesh refinement.  This avoids changing the
# discrete nozzle/return split when the job is restarted on a regenerated
# mesh.

BOUNDARY_TOP = 1
BOUNDARY_WALL = 2
BOUNDARY_NOZZLE = 3
BOUNDARY_RETURN = 4

# The real experiment must displace Q as fresh water enters. In the absence of
# a measured outlet-patch position/shape in the supplied setup, the compensating
# flux is spread very weakly over the floor outside the nozzle. This makes the
# fixed incompressible domain solvable while disturbing the plume as little as
# possible. If the physical outlet geometry is measured later, replace marker 4
# with that patch and keep the same exact flux-balance checks below.

mesh.init()
hmin_global = MPI.min(comm, mesh.hmin())
boundary_tol = max(1.0e-10, 0.10 * hmin_global)

boundaries = MeshFunction(
    "size_t",
    mesh,
    mesh.topology().dim() - 1,
    0,
)

for facet in facets(mesh):
    if not facet.exterior():
        continue

    mp = facet.midpoint()
    x_f = mp.x()
    y_f = mp.y()
    z_f = mp.z()
    r_f = math.sqrt(x_f*x_f + y_f*y_f)

    if near(z_f, z_min, boundary_tol):
        if r_f <= a_nozzle:
            boundaries[facet] = BOUNDARY_NOZZLE
        else:
            boundaries[facet] = BOUNDARY_RETURN
    elif near(z_f, z_max, boundary_tol):
        boundaries[facet] = BOUNDARY_TOP
    else:
        boundaries[facet] = BOUNDARY_WALL

normal = FacetNormal(mesh)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dx_measure = Measure("dx", domain=mesh)
dS_measure = Measure("dS", domain=mesh)
x_coord, y_coord, z_coord = SpatialCoordinate(mesh)

volume_mesh = assemble(Constant(1.0) * dx_measure)
area_nozzle = assemble(Constant(1.0) * ds(BOUNDARY_NOZZLE))
area_return = assemble(Constant(1.0) * ds(BOUNDARY_RETURN))
area_top = assemble(Constant(1.0) * ds(BOUNDARY_TOP))
area_wall = assemble(Constant(1.0) * ds(BOUNDARY_WALL))
number_nozzle_facets = MPI.sum(
    comm,
    sum(1 for facet in facets(mesh)
        if facet.exterior() and boundaries[facet] == BOUNDARY_NOZZLE)
)
max_nozzle_facet_local = 0.0
for facet in facets(mesh):
    if facet.exterior() and boundaries[facet] == BOUNDARY_NOZZLE:
        pts = [v.point() for v in vertices(facet)]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = pts[i].distance(pts[j])
                max_nozzle_facet_local = max(max_nozzle_facet_local, d)
max_nozzle_facet = MPI.max(comm, max_nozzle_facet_local)

root_print("")
root_print("================================================")
root_print("DISCRETE GEOMETRY")
root_print("================================================")
root_print("tank volume =", volume_mesh)
root_print("nozzle area =", area_nozzle)
root_print("return area =", area_return)
root_print("top area    =", area_top)
root_print("wall area   =", area_wall)
root_print("summed local nozzle-facet count =", number_nozzle_facets)
root_print("largest nozzle facet =", max_nozzle_facet)

# The generated counts are saved in every checkpoint and checked on restart.
# They are intentionally not hard-coded here: this is a new physical-domain
# mesh, and its first clean MPI build establishes the deterministic signature.
relative_nozzle_area_error = abs(
    area_nozzle - math.pi * a_nozzle**2
) / (math.pi * a_nozzle**2)

root_print("relative nozzle-area error =", relative_nozzle_area_error)

if relative_nozzle_area_error > 0.05:
    raise RuntimeError(
        "The discrete nozzle area differs from pi*a^2 by more than 5%."
    )

if max_nozzle_facet > 0.75 * d_nozzle:
    raise RuntimeError(
        "The largest nozzle-boundary facet is too large for a 2 mm source."
    )


# TAYLOR-HOOD / SCALAR FUNCTION SPACES

# Build these exactly in the same style as the working production code.
# Do NOT pre-build duplicate P2/P1 helper spaces: that was the V2 startup
# memory regression.

velocity_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
pressure_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
mixed_element = MixedElement([velocity_element, pressure_element])

W = FunctionSpace(mesh, mixed_element)
C = FunctionSpace(mesh, "DG", 0)

# One standalone P2 vector space is retained only because the H(div) RHS
# uses a coefficient updated from the Taylor-Hood velocity each timestep.
V = FunctionSpace(mesh, velocity_element)

unit_inlet_expression = Expression(
    (
        "0.0",
        "0.0",
        "((x[0]*x[0] + x[1]*x[1]) <= a*a) ? "
        "(1.0 - (x[0]*x[0] + x[1]*x[1])/(a*a)) : 0.0",
    ),
    degree=2,
    a=a_nozzle,
)

unit_return_expression = Expression(
    (
        "0.0",
        "0.0",
        "((x[0]*x[0] + x[1]*x[1]) <= a*a) ? 0.0 : -1.0",
    ),
    degree=1,
    a=a_nozzle,
)


def calibrated_boundary_pair(space, label):
    """
    Calibrate the exact discrete trace used at runtime.

    Both conditional expressions are imposed on BOTH bottom marker sets. This
    includes every crossing-facet/shared-DOF contribution produced by the
    non-fitted circular edge, rather than assuming marker 3 contains only inlet
    trace and marker 4 only return trace.
    """
    unit_in = Function(space)
    unit_ret = Function(space)
    unit_in.vector().zero()
    unit_ret.vector().zero()

    for marker in (BOUNDARY_NOZZLE, BOUNDARY_RETURN):
        DirichletBC(
            space, unit_inlet_expression, boundaries, marker
        ).apply(unit_in.vector())
        DirichletBC(
            space, unit_return_expression, boundaries, marker
        ).apply(unit_ret.vector())

    unit_in.vector().apply("insert")
    unit_ret.vector().apply("insert")

    matrix = np.array(
        [
            [
                -assemble(dot(unit_in, normal) * ds(BOUNDARY_NOZZLE)),
                -assemble(dot(unit_ret, normal) * ds(BOUNDARY_NOZZLE)),
            ],
            [
                assemble(dot(unit_in, normal) * ds(BOUNDARY_RETURN)),
                assemble(dot(unit_ret, normal) * ds(BOUNDARY_RETURN)),
            ],
        ],
        dtype=float,
    )

    if abs(np.linalg.det(matrix)) < 1.0e-20:
        raise RuntimeError(label + " flux-calibration matrix is singular.")

    inlet_amp, return_amp = np.linalg.solve(
        matrix,
        np.array([Q_in, Q_in], dtype=float),
    )

    full = Function(space)
    full.vector().zero()
    full.vector().axpy(float(inlet_amp), unit_in.vector())
    full.vector().axpy(float(return_amp), unit_ret.vector())
    full.vector().apply("insert")

    q_nozzle = -assemble(dot(full, normal) * ds(BOUNDARY_NOZZLE))
    q_return = assemble(dot(full, normal) * ds(BOUNDARY_RETURN))

    flux_tolerance = max(1.0e-14, 1.0e-8 * Q_in)
    if (
        abs(q_nozzle - Q_in) > flux_tolerance
        or abs(q_return - Q_in) > flux_tolerance
        or abs(q_return - q_nozzle) > flux_tolerance
    ):
        raise RuntimeError(
            "{} trace calibration failed: Qin={}, Qout={}, target={}.".format(
                label,
                q_nozzle,
                q_return,
                Q_in,
            )
        )

    return (
        float(inlet_amp),
        float(return_amp),
        float(q_nozzle),
        float(q_return),
    )


th_inlet_amp, th_return_amp, th_q_nozzle, th_q_return = (
    calibrated_boundary_pair(V, "Taylor-Hood P2")
)

root_print("")
root_print("================================================")
root_print("FUNCTION SPACES")
root_print("================================================")
root_print("Mixed velocity-pressure DOFs =", W.dim())
root_print("Concentration DOFs           =", C.dim())
root_print("Total primary flow+scalar DOFs =", W.dim() + C.dim())
root_print("Q target                      =", Q_in)
root_print("analytical mean inlet speed   =", Q_in / (math.pi * a_nozzle**2))
root_print("analytical peak inlet speed   =", 2.0 * Q_in / (math.pi * a_nozzle**2))
root_print("P2 inlet amplitude            =", th_inlet_amp)
root_print("P2 return amplitude           =", th_return_amp)
root_print("P2 nozzle flux                =", th_q_nozzle)
root_print("P2 return flux                =", th_q_return)
root_print("[startup] Taylor-Hood spaces built; starting H(div) setup.")



# H(DIV) SPACE + DISCRETE BOUNDARY CALIBRATION


V_hdiv = FunctionSpace(mesh, "RT", 1)
rt_element = FiniteElement("RT", mesh.ufl_cell(), 1)
multiplier_element = FiniteElement("DG", mesh.ufl_cell(), 0)
HDivMixed = MixedElement([rt_element, multiplier_element])
Z_hdiv = FunctionSpace(mesh, HDivMixed)

hdiv_inlet_amp, hdiv_return_amp, hdiv_q_nozzle, hdiv_q_return = (
    calibrated_boundary_pair(V_hdiv, "H(div)")
)

root_print("")
root_print("================================================")
root_print("H(DIV) SCALAR-FLUX CALIBRATION")
root_print("================================================")
root_print("H(div) reconstruction DOFs   =", V_hdiv.dim())
root_print("H(div) mixed solve DOFs       =", Z_hdiv.dim())
root_print("H(div) inlet amplitude =", hdiv_inlet_amp)
root_print("H(div) return amplitude =", hdiv_return_amp)
root_print("H(div) nozzle full =", hdiv_q_nozzle)
root_print("H(div) return full =", hdiv_q_return)
root_print("H(div) net full =", hdiv_q_return - hdiv_q_nozzle)

mesh_hmin_global = MPI.min(comm, mesh.hmin())
mesh_hmax_global = MPI.max(comm, mesh.hmax())

def write_configuration_and_diagnostics_header():
    error_message = None

    if rank == 0:
        try:
            configuration = {
                "model": "full rectangular physical tank",
                "legacy_fenics": "DOLFIN 2019.1",
                "solver_schema": solver_schema,
                "mesh_recipe": mesh_recipe,
                "quarter_domain": False,
                "dimensions_m": [L_tank, W_tank, H_tank],
                "coordinate_bounds_m": {
                    "x": [x_min, x_max],
                    "y": [y_min, y_max],
                    "z": [z_min, z_max],
                },
                "nozzle_diameter_m": d_nozzle,
                "flow_rate_cc_per_min": Q_cc_per_min,
                "flow_rate_m3_per_s": Q_in,
                "rho_fresh_kg_per_m3": rho_fresh,
                "rho_salt_kg_per_m3": rho_salt,
                "nu_m2_per_s": nu,
                "scalar_diffusivity_m2_per_s": D,
                "dt_s": dt,
                "target_time_s": T_global,
                "segment_duration_s": segment_duration,
                "checkpoint_wall_seconds": checkpoint_wall_seconds,
                "inlet_perturbation": inlet_perturbation,
                "mesh_vertices": mesh_vertices_global,
                "mesh_cells": mesh_cells_global,
                "mesh_hmin_m": mesh_hmin_global,
                "mesh_hmax_m": mesh_hmax_global,
                "discrete_nozzle_area_m2": area_nozzle,
                "numerical_return": (
                    "flux-balanced weak downward velocity over the tank floor "
                    "outside the nozzle"
                ),
            }

            config_tmp = CONFIGURATION_PATH + ".tmp"
            with open(config_tmp, "w") as f:
                json.dump(configuration, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(config_tmp, CONFIGURATION_PATH)

            if not os.path.exists(DIAGNOSTICS_PATH):
                with open(DIAGNOSTICS_PATH, "w") as f:
                    f.write(
                        "step,time_s,ramp,tilt_x,tilt_y,"
                        "max_velocity_component,div_hdiv_l2,div_th_l2,"
                        "q_nozzle_hdiv,q_return_hdiv,q_nozzle_th,q_return_th,"
                        "c_min,c_max,fresh_volume_m3,fresh_expected_m3,"
                        "fresh_ratio,x_fresh_centroid_m,y_fresh_centroid_m,"
                        "z_fresh_centroid_m,specific_lateral_ke_integral\n"
                    )
        except Exception as exc:
            error_message = (
                "Could not write configuration/diagnostics header: {}".format(exc)
            )

    error_message = pycomm.bcast(error_message, root=0)
    if error_message is not None:
        raise RuntimeError(error_message)



# TIME-DEPENDENT VELOCITY BOUNDARY CONDITIONS

# One conditional expression is imposed on both bottom marker sets. Using one
# expression prevents an interface DOF from receiving inconsistent values.

# The perturbation is a tiny tangential component of the inlet velocity. It
# changes the instantaneous jet direction by at most inlet_perturbation radians
# but contributes exactly zero normal volume flux. Different incommensurate
# periods in x and y avoid imposing a preferred symmetry plane. It is not
# random, so a restarted MPI job reproduces the same forcing at the same time.

bottom_velocity = Expression(
    (
        "ramp * (((x[0]*x[0] + x[1]*x[1]) <= a*a) ? "
        "(tilt_x*Uin*(1.0-(x[0]*x[0] + x[1]*x[1])/(a*a))) : 0.0)",
        "ramp * (((x[0]*x[0] + x[1]*x[1]) <= a*a) ? "
        "(tilt_y*Uin*(1.0-(x[0]*x[0] + x[1]*x[1])/(a*a))) : 0.0)",
        "ramp * (((x[0]*x[0] + x[1]*x[1]) <= a*a) ? "
        "(Uin*(1.0-(x[0]*x[0] + x[1]*x[1])/(a*a))) : (-Uret))",
    ),
    degree=2,
    ramp=0.0,
    tilt_x=0.0,
    tilt_y=0.0,
    a=a_nozzle,
    Uin=th_inlet_amp,
    Uret=th_return_amp,
)

zero_velocity = Constant((0.0, 0.0, 0.0))

bcs_mixed = [
    DirichletBC(W.sub(0), zero_velocity, boundaries, BOUNDARY_TOP),
    DirichletBC(W.sub(0), zero_velocity, boundaries, BOUNDARY_WALL),
    DirichletBC(W.sub(0), bottom_velocity, boundaries, BOUNDARY_NOZZLE),
    DirichletBC(W.sub(0), bottom_velocity, boundaries, BOUNDARY_RETURN),
]


# PRIMARY STATE


w_n = Function(W)
w_new = Function(W)
c_n = Function(C)
c_new = Function(C)
u_advect = Function(V)



# RESTART FORMAT 4: COORDINATE-KEYED FE STATE


# The checkpoint stores owned FE DOF values together with their physical
# coordinates. On restart the mesh is regenerated and each current DOF is
# matched back to its coordinate. This avoids silently assuming that PETSc/
# DOLFIN will assign the same parallel/global vector numbers after a fresh
# mesh build.



def owned_dof_coordinates(space):
    """
    Return owned DOF coordinates in PETSc owned-vector order.

    vector().get_local() is ordered by the owned global index interval
    [lo, hi).  Local DOLFIN DOF numbering need not put those DOFs first, so
    map coordinates into slots using global_index-lo instead of assuming a
    particular local partition ordering.
    """
    gdim = mesh.geometry().dim()
    coords_all = space.tabulate_dof_coordinates().reshape((-1, gdim))

    local_to_global = np.asarray(
        space.dofmap().tabulate_local_to_global_dofs(), dtype=np.int64
    )
    lo, hi = space.dofmap().ownership_range()
    n_owned = int(hi - lo)

    owned_mask = (local_to_global >= lo) & (local_to_global < hi)
    owned_local = np.flatnonzero(owned_mask)

    if len(owned_local) != n_owned:
        raise RuntimeError(
            "Could not identify all owned DOFs: found {}, expected {}.".format(
                len(owned_local), n_owned
            )
        )

    owned_coords = np.empty((n_owned, gdim), dtype=float)
    seen = np.zeros(n_owned, dtype=bool)

    for local_i in owned_local:
        slot = int(local_to_global[local_i] - lo)
        if slot < 0 or slot >= n_owned or seen[slot]:
            raise RuntimeError("Invalid owned DOF global-index mapping.")
        owned_coords[slot, :] = coords_all[local_i, :]
        seen[slot] = True

    if not np.all(seen):
        raise RuntimeError("Owned DOF coordinate map has missing PETSc slots.")

    return owned_coords



def canonical_key(coord):
    rounded = np.round(np.asarray(coord, dtype=float), coord_decimals)
    return tuple(float(x) for x in rounded)



def gather_function_canonical(function, label):
    local_coords = owned_dof_coordinates(function.function_space())
    local_values = np.asarray(function.vector().get_local(), dtype=float)

    if local_values.shape[0] != local_coords.shape[0]:
        local_error = (
            "{}: local coordinate/value size mismatch {} vs {}".format(
                label, local_coords.shape[0], local_values.shape[0]
            )
        )
    else:
        local_error = None

    errors = pycomm.gather(local_error, root=0)
    if rank == 0:
        error_message = next((e for e in errors if e is not None), None)
    else:
        error_message = None
    error_message = pycomm.bcast(error_message, root=0)
    if error_message is not None:
        raise RuntimeError(error_message)

    coord_parts = pycomm.gather(local_coords, root=0)
    value_parts = pycomm.gather(local_values, root=0)

    coords_out = None
    values_out = None
    error_message = None

    if rank == 0:
        try:
            coords = np.vstack(coord_parts)
            values = np.concatenate(value_parts)

            rounded = np.round(coords, coord_decimals)
            order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
            coords = coords[order]
            rounded = rounded[order]
            values = values[order]

            if coords.shape[0] != function.function_space().dim():
                raise RuntimeError(
                    "{}: gathered {} owned DOFs but global space dimension is {}.".format(
                        label, coords.shape[0], function.function_space().dim()
                    )
                )

            if rounded.shape[0] > 1:
                duplicate = np.all(rounded[1:] == rounded[:-1], axis=1)
                if np.any(duplicate):
                    raise RuntimeError(
                        "{}: duplicate coordinate keys found in checkpoint field.".format(
                            label
                        )
                    )

            coords_out = coords
            values_out = values
        except Exception as exc:
            error_message = str(exc)

    error_message = pycomm.bcast(error_message, root=0)
    if error_message is not None:
        raise RuntimeError(error_message)

    return coords_out, values_out



def restore_function_from_canonical(function, saved_coords, saved_values, label):
    local_coords = owned_dof_coordinates(function.function_space())
    request_parts = pycomm.gather(local_coords, root=0)

    send_values = None
    error_message = None

    if rank == 0:
        try:
            mapping = {}
            for idx, coord in enumerate(saved_coords):
                key = canonical_key(coord)
                if key in mapping:
                    raise RuntimeError(
                        "{}: duplicate saved coordinate key {}".format(label, key)
                    )
                mapping[key] = float(saved_values[idx])

            send_values = []
            misses = 0

            for requests in request_parts:
                vals = np.empty(requests.shape[0], dtype=float)
                for j, coord in enumerate(requests):
                    key = canonical_key(coord)
                    if key not in mapping:
                        misses += 1
                        vals[j] = np.nan
                    else:
                        vals[j] = mapping[key]
                send_values.append(vals)

            if misses != 0:
                raise RuntimeError(
                    "{}: {} regenerated DOFs were absent from the checkpoint.".format(
                        label, misses
                    )
                )
        except Exception as exc:
            error_message = str(exc)

    error_message = pycomm.bcast(error_message, root=0)
    if error_message is not None:
        raise RuntimeError(error_message)

    local_values = pycomm.scatter(send_values, root=0)

    local_bad = not np.all(np.isfinite(local_values))
    any_bad = pycomm.allreduce(local_bad, op=PY_MPI.LOR)
    if any_bad:
        raise RuntimeError(label + ": restored values contain NaN/Inf.")

    function.vector().set_local(local_values)
    function.vector().apply("insert")



def state_fingerprint(w_state, c_state):
    u_state, p_state = w_state.split(deepcopy=True)

    return {
        "fresh_volume": float(
            assemble((Constant(1.0) - c_state) * dx_measure)
        ),
        "scalar_integral": float(assemble(c_state * dx_measure)),
        "u_l2": float(norm(u_state, "L2")),
        "p_l2": float(norm(p_state, "L2")),
        "c_min": float(c_state.vector().min()),
        "c_max": float(c_state.vector().max()),
    }



def fingerprint_close(a, b):
    return abs(float(a) - float(b)) <= (
        fingerprint_atol
        + fingerprint_rtol * max(abs(float(a)), abs(float(b)), 1.0e-12)
    )



def save_checkpoint(step, time_value, expected_fresh_physical):
    """Atomically save a complete FE state to a new .npz file."""

    u_save, p_save = w_n.split(deepcopy=True)
    ux_save, uy_save, uz_save = u_save.split(deepcopy=True)

    fields = {}

    for name, function in (
        ("ux", ux_save),
        ("uy", uy_save),
        ("uz", uz_save),
        ("p", p_save),
        ("c", c_n),
    ):
        coords, values = gather_function_canonical(function, name)
        if rank == 0:
            fields[name + "_coords"] = coords
            fields[name + "_values"] = values

    fingerprint = state_fingerprint(w_n, c_n)

    if rank == 0:
        meta = {
            "restart_format_version": 4,
            "step": int(step),
            "time": float(time_value),
            "dt": float(dt),
            "T_global": float(T_global),
            "solver_schema": solver_schema,
            "mesh_recipe": mesh_recipe,
            "expected_fresh_physical": float(expected_fresh_physical),
            "mesh_vertices": int(mesh_vertices_global),
            "mesh_cells": int(mesh_cells_global),
            "L_tank": float(L_tank),
            "W_tank": float(W_tank),
            "H_tank": float(H_tank),
            "d_nozzle": float(d_nozzle),
            "Q_in": float(Q_in),
            "Q_cc_per_min": float(Q_cc_per_min),
            "nu": float(nu),
            "D": float(D),
            "g": float(g),
            "beta": float(beta),
            "rho_fresh": float(rho_fresh),
            "rho_salt": float(rho_salt),
            "inlet_perturbation": float(inlet_perturbation),
            "perturbation_period_x": float(perturbation_period_x),
            "perturbation_period_y": float(perturbation_period_y),
            "ramp_time": float(ramp_time),
            "coord_decimals": int(coord_decimals),
            "fingerprint": fingerprint,
        }

        filename = "state_step_{:09d}.npz".format(int(step))
        final_path = os.path.join(CHECKPOINT_DIR, filename)
        tmp_path = final_path + ".tmp"

        payload = dict(fields)
        payload["meta_json"] = np.asarray(json.dumps(meta, sort_keys=True))

        # np.savez appends .npz if given a string without that suffix. Use a
        # file handle so the temporary filename remains exactly tmp_path.
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **payload)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, final_path)

        # Only point latest.json at a checkpoint after the state file has
        # been fully closed and atomically renamed.
        latest_tmp = os.path.join(CHECKPOINT_DIR, "latest.json.tmp")
        latest_final = os.path.join(CHECKPOINT_DIR, "latest.json")

        with open(latest_tmp, "w") as f:
            json.dump(
                {
                    "file": filename,
                    "step": int(step),
                    "time": float(time_value),
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.flush()
            os.fsync(f.fileno())

        os.replace(latest_tmp, latest_final)

        # Retain the newest few complete checkpoints so a killed/corrupted
        # write can never destroy the previous valid state.
        complete = []
        for candidate in sorted(
            glob.glob(os.path.join(CHECKPOINT_DIR, "state_step_*.npz")),
            reverse=True,
        ):
            try:
                candidate_meta = _read_npz_metadata(candidate)
                if int(candidate_meta.get("restart_format_version", -1)) == 4:
                    complete.append(candidate)
            except Exception:
                # Never let a corrupt/high-numbered filename displace a known
                # complete checkpoint during retention cleanup.
                pass

        for old_path in complete[keep_checkpoints:]:
            try:
                os.remove(old_path)
            except OSError:
                pass

        print("")
        print(
            "CHECKPOINT SAVED: step {}  t = {:.6f} s".format(
                int(step), float(time_value)
            )
        )
        print("  ", final_path)
        print("  fresh volume =", fingerprint["fresh_volume"])
        sys.stdout.flush()

    MPI.barrier(comm)



def validate_checkpoint_metadata(meta):
    checks = [
        (int(meta.get("restart_format_version", -1)) == 4, "format version"),
        (str(meta.get("solver_schema", "")) == solver_schema, "solver schema"),
        (str(meta.get("mesh_recipe", "")) == mesh_recipe, "mesh recipe"),
        (abs(float(meta["dt"]) - dt) <= 1.0e-15, "dt"),
        (int(meta["mesh_vertices"]) == mesh_vertices_global, "mesh vertices"),
        (int(meta["mesh_cells"]) == mesh_cells_global, "mesh cells"),
        (abs(float(meta["L_tank"]) - L_tank) <= 1.0e-15, "L_tank"),
        (abs(float(meta["W_tank"]) - W_tank) <= 1.0e-15, "W_tank"),
        (abs(float(meta["H_tank"]) - H_tank) <= 1.0e-15, "H_tank"),
        (abs(float(meta["d_nozzle"]) - d_nozzle) <= 1.0e-15, "d_nozzle"),
        (abs(float(meta["Q_in"]) - Q_in) <= 1.0e-18, "Q_in"),
        (
            abs(float(meta["Q_cc_per_min"]) - Q_cc_per_min) <= 1.0e-12,
            "Q_cc_per_min",
        ),
        (abs(float(meta["nu"]) - nu) <= 1.0e-18, "nu"),
        (abs(float(meta["D"]) - D) <= 1.0e-18, "D"),
        (abs(float(meta["g"]) - g) <= 1.0e-15, "g"),
        (abs(float(meta["beta"]) - beta) <= 1.0e-15, "beta"),
        (abs(float(meta["rho_fresh"]) - rho_fresh) <= 1.0e-12, "rho_fresh"),
        (abs(float(meta["rho_salt"]) - rho_salt) <= 1.0e-12, "rho_salt"),
        (
            abs(float(meta["inlet_perturbation"]) - inlet_perturbation)
            <= 1.0e-15,
            "inlet_perturbation",
        ),
        (
            abs(float(meta["perturbation_period_x"]) - perturbation_period_x)
            <= 1.0e-15,
            "perturbation_period_x",
        ),
        (
            abs(float(meta["perturbation_period_y"]) - perturbation_period_y)
            <= 1.0e-15,
            "perturbation_period_y",
        ),
        (abs(float(meta["ramp_time"]) - ramp_time) <= 1.0e-15, "ramp_time"),
        (int(meta["coord_decimals"]) == coord_decimals, "coord_decimals"),
    ]

    failed = [name for ok, name in checks if not ok]
    if failed:
        raise RuntimeError(
            "Checkpoint is incompatible with this code: " + ", ".join(failed)
        )



def load_checkpoint(path, meta):
    validate_checkpoint_metadata(meta)

    checkpoint_read_error = None
    if rank == 0:
        try:
            with np.load(path, allow_pickle=False) as data:
                saved = {
                    name: np.array(data[name], copy=True)
                    for name in (
                        "ux_coords",
                        "ux_values",
                        "uy_coords",
                        "uy_values",
                        "uz_coords",
                        "uz_values",
                        "p_coords",
                        "p_values",
                        "c_coords",
                        "c_values",
                    )
                }
        except Exception as exc:
            saved = None
            checkpoint_read_error = (
                "Could not read checkpoint payload {}: {}".format(path, exc)
            )
    else:
        saved = None

    checkpoint_read_error = pycomm.bcast(checkpoint_read_error, root=0)
    if checkpoint_read_error is not None:
        raise RuntimeError(checkpoint_read_error)

    # Only root needs the complete saved arrays.  Create the scalar/vector
    # helper spaces TEMPORARILY from the mixed state itself, then discard
    # them after reconstruction.  This avoids keeping duplicate P2/P1
    # spaces resident throughout the production run.
    u_loaded, p_loaded = w_n.split(deepcopy=True)
    ux, uy, uz = u_loaded.split(deepcopy=True)

    for name, function in (
        ("ux", ux),
        ("uy", uy),
        ("uz", uz),
        ("p", p_loaded),
        ("c", c_n),
    ):
        if rank == 0:
            coords = saved[name + "_coords"]
            values = saved[name + "_values"]
        else:
            coords = None
            values = None

        restore_function_from_canonical(function, coords, values, name)

    assigner_vec = FunctionAssigner(
        u_loaded.function_space(),
        [ux.function_space(), uy.function_space(), uz.function_space()],
    )
    assigner_vec.assign(u_loaded, [ux, uy, uz])

    assigner_mixed = FunctionAssigner(
        W, [u_loaded.function_space(), p_loaded.function_space()]
    )
    assigner_mixed.assign(w_n, [u_loaded, p_loaded])

    w_new.assign(w_n)
    c_new.assign(c_n)

    current = state_fingerprint(w_n, c_n)
    saved_fp = meta["fingerprint"]

    bad = []
    for key in ("fresh_volume", "scalar_integral", "u_l2", "p_l2", "c_min", "c_max"):
        if not fingerprint_close(current[key], saved_fp[key]):
            bad.append(
                "{} saved={} loaded={}".format(
                    key, saved_fp[key], current[key]
                )
            )

    if bad:
        raise RuntimeError(
            "Restart state failed its post-load fingerprint check:\n  "
            + "\n  ".join(bad)
        )

    root_print("")
    root_print("Restart state reconstructed and fingerprint verified.")
    root_print("Loaded fresh volume =", current["fresh_volume"])
    root_print("Loaded c range      =", current["c_min"], "to", current["c_max"])



# INITIALISE OR RESTORE STATE


if restart_path is None:
    w_n.vector().zero()
    w_n.vector().apply("insert")
    w_new.assign(w_n)

    c_n.assign(Constant(1.0))
    c_new.assign(c_n)

    start_step = 0
    start_time = 0.0
    expected_fresh_physical = 0.0

    root_print("Initial condition: c = 1 throughout tank.")

    # Exercise the COMPLETE new checkpoint path before spending hours on
    # timestepping. This writes a step-0 state, reconstructs it by physical
    # DOF coordinates, and checks all saved state fingerprints.
    root_print("")
    root_print("Running physical-tank restart step-0 round-trip self-test...")
    save_checkpoint(0, 0.0, 0.0)

    step0_path = os.path.join(
        CHECKPOINT_DIR,
        "state_step_{:09d}.npz".format(0),
    )

    if rank == 0:
        step0_meta = _read_npz_metadata(step0_path)
    else:
        step0_meta = None
    step0_meta = pycomm.bcast(step0_meta, root=0)

    load_checkpoint(step0_path, step0_meta)
    root_print("PHYSICAL-TANK RESTART STEP-0 ROUND-TRIP SELF-TEST PASSED.")
else:
    load_checkpoint(restart_path, restart_meta)

    start_step = int(restart_meta["step"])
    start_time = float(restart_meta["time"])
    expected_fresh_physical = float(
        restart_meta.get("expected_fresh_physical", 0.0)
    )

# Exact consistency of time/step metadata.
if abs(start_time - start_step * dt) > 5.0e-12:
    raise RuntimeError(
        "Restart metadata has inconsistent step/time: {} / {}".format(
            start_step, start_time
        )
    )


def reconcile_diagnostics_csv(accepted_step):
    """Remove stale/duplicate rows beyond the accepted restart state."""
    error_message = None

    if rank == 0:
        try:
            with open(DIAGNOSTICS_PATH, "r") as f:
                lines = f.readlines()

            if not lines:
                raise RuntimeError("diagnostics.csv has no header")

            header = lines[0]
            rows_by_step = {}

            for line in lines[1:]:
                stripped = line.strip()
                if not stripped:
                    continue
                fields = stripped.split(",")
                row_step = int(fields[0])
                if row_step <= accepted_step:
                    rows_by_step[row_step] = stripped + "\n"

            tmp_path = DIAGNOSTICS_PATH + ".tmp"
            with open(tmp_path, "w") as f:
                f.write(header)
                for row_step in sorted(rows_by_step):
                    f.write(rows_by_step[row_step])
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, DIAGNOSTICS_PATH)
        except Exception as exc:
            error_message = "Could not reconcile diagnostics.csv: {}".format(exc)

    error_message = pycomm.bcast(error_message, root=0)
    if error_message is not None:
        raise RuntimeError(error_message)


write_configuration_and_diagnostics_header()
reconcile_diagnostics_csv(start_step)



# MIXED FLOW PROBLEM


(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

u_n_ufl, p_n_ufl = split(w_n)

buoyancy = as_vector(
    (
        Constant(0.0),
        Constant(0.0),
        g * beta * (Constant(1.0) - c_n),
    )
)

a_mixed = (
    (1.0 / dt) * inner(u, v) * dx_measure
    + inner(dot(grad(u), u_n_ufl), v) * dx_measure
    + nu * inner(grad(u), grad(v)) * dx_measure
    - p * div(v) * dx_measure
    - q * div(u) * dx_measure
)

L_mixed = (
    (1.0 / dt) * inner(u_n_ufl, v) * dx_measure
    + inner(buoyancy, v) * dx_measure
)

assembler = SystemAssembler(a_mixed, L_mixed, bcs_mixed)



# MIXED PRESSURE NULLSPACE


null_vec = w_new.vector().copy()
null_vec.zero()
W.sub(1).dofmap().set(null_vec, 1.0)
null_vec.apply("insert")

null_norm = null_vec.norm("l2")
if null_norm <= 0.0:
    raise RuntimeError("Pressure nullspace vector has zero norm.")
null_vec *= 1.0 / null_norm

dolfin_nullspace = VectorSpaceBasis([null_vec])
null_vec_petsc = as_backend_type(null_vec).vec()

petsc_nullspace = PETSc.NullSpace().create(
    constant=False,
    vectors=[null_vec_petsc],
    comm=PETSc.COMM_WORLD,
)



# MONOLITHIC PETSC SOLVER


ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
ksp.setOptionsPrefix("mono_")

opts = PETSc.Options()
opts["mono_ksp_type"] = "gmres"
opts["mono_ksp_rtol"] = "1.0e-10"
opts["mono_ksp_atol"] = "1.0e-12"
opts["mono_ksp_max_it"] = "500"
opts["mono_ksp_gmres_restart"] = "100"
opts["mono_ksp_initial_guess_nonzero"] = "true"
opts["mono_pc_type"] = "fieldsplit"
opts["mono_pc_fieldsplit_detect_saddle_point"] = "true"
opts["mono_pc_fieldsplit_type"] = "schur"
opts["mono_pc_fieldsplit_schur_fact_type"] = "full"
opts["mono_pc_fieldsplit_schur_precondition"] = "selfp"
opts["mono_fieldsplit_0_ksp_type"] = "preonly"
opts["mono_fieldsplit_0_pc_type"] = "hypre"
opts["mono_fieldsplit_1_ksp_type"] = "preonly"
opts["mono_fieldsplit_1_pc_type"] = "hypre"

ksp.setFromOptions()



# H(DIV) RECONSTRUCTION (FEniCS "RT", DEGREE 1; LOWEST-ORDER RT)

# Find u_H in FEniCS RT degree 1 (often called mathematical RT0), with
# prescribed normal boundary fluxes, such that
# it is the L2-nearest field to the Taylor-Hood velocity while being
# exactly divergence free in DG0:

#   (u_H, v) + (lambda, div v) = (u_TH, v)
#   (div u_H, q) = 0


(u_h, lam_h) = TrialFunctions(Z_hdiv)
(v_h, q_h) = TestFunctions(Z_hdiv)

hdiv_bottom_velocity = Expression(
    (
        "0.0",
        "0.0",
        "ramp * (((x[0]*x[0] + x[1]*x[1]) <= a*a) ? "
        "(Uin*(1.0-(x[0]*x[0] + x[1]*x[1])/(a*a))) : (-Uret))",
    ),
    degree=2,
    ramp=0.0,
    a=a_nozzle,
    Uin=hdiv_inlet_amp,
    Uret=hdiv_return_amp,
)

bcs_hdiv = [
    DirichletBC(Z_hdiv.sub(0), zero_velocity, boundaries, BOUNDARY_TOP),
    DirichletBC(Z_hdiv.sub(0), zero_velocity, boundaries, BOUNDARY_WALL),
    DirichletBC(
        Z_hdiv.sub(0), hdiv_bottom_velocity, boundaries, BOUNDARY_NOZZLE
    ),
    DirichletBC(
        Z_hdiv.sub(0), hdiv_bottom_velocity, boundaries, BOUNDARY_RETURN
    ),
]

a_hdiv = (
    inner(u_h, v_h) * dx_measure
    + lam_h * div(v_h) * dx_measure
    + q_h * div(u_h) * dx_measure
)

L_hdiv = inner(u_advect, v_h) * dx_measure

A_hdiv = assemble(a_hdiv)
for bc in bcs_hdiv:
    bc.apply(A_hdiv)

# Constant multiplier nullspace.
z_hdiv = Function(Z_hdiv)
hdiv_null_vec = z_hdiv.vector().copy()
hdiv_null_vec.zero()
Z_hdiv.sub(1).dofmap().set(hdiv_null_vec, 1.0)
hdiv_null_vec.apply("insert")

hdiv_null_norm = hdiv_null_vec.norm("l2")
if hdiv_null_norm <= 0.0:
    raise RuntimeError("H(div) multiplier nullspace vector has zero norm.")
hdiv_null_vec *= 1.0 / hdiv_null_norm

hdiv_dolfin_nullspace = VectorSpaceBasis([hdiv_null_vec])
as_backend_type(A_hdiv).set_nullspace(hdiv_dolfin_nullspace)

hdiv_petsc_nullspace = PETSc.NullSpace().create(
    constant=False,
    vectors=[as_backend_type(hdiv_null_vec).vec()],
    comm=PETSc.COMM_WORLD,
)

A_hdiv_petsc = as_backend_type(A_hdiv).mat()
A_hdiv_petsc.setNullSpace(hdiv_petsc_nullspace)

hdiv_ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
hdiv_ksp.setOptionsPrefix("hdiv_")

opts["hdiv_ksp_type"] = "gmres"
opts["hdiv_ksp_rtol"] = "1.0e-10"
opts["hdiv_ksp_atol"] = "1.0e-12"
opts["hdiv_ksp_max_it"] = "400"
opts["hdiv_ksp_gmres_restart"] = "100"
opts["hdiv_ksp_initial_guess_nonzero"] = "true"
opts["hdiv_pc_type"] = "fieldsplit"
opts["hdiv_pc_fieldsplit_detect_saddle_point"] = "true"
opts["hdiv_pc_fieldsplit_type"] = "schur"
opts["hdiv_pc_fieldsplit_schur_fact_type"] = "full"
opts["hdiv_pc_fieldsplit_schur_precondition"] = "selfp"
opts["hdiv_fieldsplit_0_ksp_type"] = "preonly"
opts["hdiv_fieldsplit_0_pc_type"] = "hypre"
opts["hdiv_fieldsplit_1_ksp_type"] = "preonly"
opts["hdiv_fieldsplit_1_pc_type"] = "hypre"

hdiv_ksp.setFromOptions()
hdiv_ksp.setOperators(A_hdiv_petsc)

u_hdiv_ufl, lambda_hdiv_ufl = split(z_hdiv)

root_print("H(div) constant reconstruction matrix assembled.")



# CONSERVATIVE DG0 SCALAR TRANSPORT USING H(DIV) FLUX


c_trial = TrialFunction(C)
s = TestFunction(C)
h_cell = CellDiameter(mesh)

un_int_hdiv = dot(avg(u_hdiv_ufl), normal("+"))

upwind_flux_int = (
    un_int_hdiv * avg(c_trial)
    + Constant(0.5) * abs(un_int_hdiv) * jump(c_trial)
)

un_bnd_hdiv = dot(u_hdiv_ufl, normal)
outflow_speed = Constant(0.5) * (un_bnd_hdiv + abs(un_bnd_hdiv))

diffusion_int = (
    D / avg(h_cell) * jump(c_trial) * jump(s)
)

# NOTE: there is deliberately NO -c*div(u) source correction here.
# The RT1 field is reconstructed to be cellwise divergence-free.
a_c = (
    c_trial * s * dx_measure
    + dt * upwind_flux_int * jump(s) * dS_measure
    + dt * outflow_speed * c_trial * s * ds
    + dt * diffusion_int * dS_measure
    + dt * D / h_cell * c_trial * s * ds(BOUNDARY_NOZZLE)
)

L_c = c_n * s * dx_measure

solver_scalar = PETScKrylovSolver("gmres", "hypre_amg")
solver_scalar.parameters["relative_tolerance"] = 1.0e-10
solver_scalar.parameters["absolute_tolerance"] = 1.0e-12
solver_scalar.parameters["maximum_iterations"] = 500



# SEGMENT OUTPUT


job_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))
segment_name = (
    "segment_full_physical_from_step_{:09d}_t_{:.3f}s_job_{}".format(
        start_step,
        start_time,
        job_id,
    ).replace(".", "p")
)

segment_dir = os.path.join(SEGMENTS_DIR, segment_name)

if rank == 0:
    os.makedirs(segment_dir, exist_ok=True)
MPI.barrier(comm)

root_print("")
root_print("Segment output directory:")
root_print(segment_dir)

u_file = XDMFFile(comm, os.path.join(segment_dir, "u_taylor_hood.xdmf"))
p_file = XDMFFile(comm, os.path.join(segment_dir, "p.xdmf"))
c_file = XDMFFile(comm, os.path.join(segment_dir, "c_dg0.xdmf"))
for output_file in (u_file, p_file, c_file):
    output_file.parameters["flush_output"] = True
    output_file.parameters["functions_share_mesh"] = True
    output_file.parameters["rewrite_function_mesh"] = False



def write_output(time_value, u_state=None, p_state=None):
    if u_state is None or p_state is None:
        u_state, p_state = w_n.split(deepcopy=True)

    u_state.rename("velocity", "u")
    p_state.rename("pressure", "p")
    c_n.rename("concentration", "c")

    # Do not project div(u) into a continuous pressure space here.
    # FEniCS project() launches an extra global PETSc solve and can fail
    # independently of the physical solver. Divergence is diagnosed directly
    # from the accepted velocity in the timestep diagnostics below.
    u_file.write(u_state, time_value)
    p_file.write(p_state, time_value)
    c_file.write(c_n, time_value)


# Initial segment output. The H(div) field is zero until the first solve; this
# is harmless at t=0/restart and avoids performing an extra expensive solve.
u_initial, p_initial = w_n.split(deepcopy=True)
write_output(start_time, u_initial, p_initial)



# RUN LIMITS


def exact_step_count(duration, label):
    raw_steps = duration / dt
    nearest = int(round(raw_steps))
    if nearest < 1 or abs(raw_steps - nearest) > 1.0e-9:
        raise ValueError(
            "{}={} s is not a positive integer multiple of dt={} s.".format(
                label,
                duration,
                dt,
            )
        )
    return nearest


max_global_step = exact_step_count(T_global, "T_GLOBAL")
steps_this_job = exact_step_count(segment_duration, "SEGMENT_DURATION")

if start_step > max_global_step:
    raise RuntimeError(
        "The checkpoint is already at step {} (t={} s), beyond the requested "
        "T_GLOBAL={} s.".format(start_step, start_time, T_global)
    )

end_step = min(max_global_step, start_step + steps_this_job)

root_print("")
root_print("================================================")
root_print("STARTING FULL RECTANGULAR PHYSICAL-TANK 3D RUN")
root_print("================================================")
root_print("dt                 =", dt)
root_print("global target time =", T_global, "s")
root_print("start time         =", start_time, "s")
root_print("advance this job   <=", segment_duration, "s")
root_print("ramp time          =", ramp_time)
root_print("nu                 =", nu)
root_print("D                  =", D)
root_print("beta               =", beta)
root_print("domain             =", L_tank, "x", W_tank, "x", H_tank, "m")
root_print("quarter domain     = False (no symmetry planes)")
root_print("flow rate          =", Q_cc_per_min, "cc/min")
root_print("inlet perturbation =", inlet_perturbation)
root_print("scalar             = conservative DG0 upwind with divergence-free RT1 flux")
root_print("start step         =", start_step)
root_print("end step this job  =", end_step)

initial_scalar_integral = volume_mesh
initial_fresh_volume = 0.0

wall_start = time.time()
last_checkpoint_wall = wall_start
last_saved_step = start_step if restart_path is not None else -1






for step in range(start_step + 1, end_step + 1):
    t_now = step * dt
    ramp_factor = smooth_ramp(t_now, ramp_time)

    tilt_x = inlet_perturbation * math.sin(
        2.0 * math.pi * t_now / perturbation_period_x
    )
    tilt_y = inlet_perturbation * math.sin(
        2.0 * math.pi * t_now / perturbation_period_y
    )

    bottom_velocity.ramp = ramp_factor
    bottom_velocity.tilt_x = tilt_x
    bottom_velocity.tilt_y = tilt_y
    hdiv_bottom_velocity.ramp = ramp_factor

    Q_target_now = ramp_factor * Q_in

    
    # MONOLITHIC TAYLOR-HOOD OSEEN SOLVE
    

    A = PETScMatrix()
    b = PETScVector()
    assembler.assemble(A, b)

    as_backend_type(A).set_nullspace(dolfin_nullspace)
    A_petsc = as_backend_type(A).mat()
    A_petsc.setNullSpace(petsc_nullspace)

    w_new.assign(w_n)

    b_petsc = as_backend_type(b).vec()
    x_petsc = as_backend_type(w_new.vector()).vec()

    petsc_nullspace.remove(b_petsc)
    petsc_nullspace.remove(x_petsc)

    ksp.setOperators(A_petsc)
    ksp.solve(b_petsc, x_petsc)

    ksp_iterations = ksp.getIterationNumber()
    ksp_reason = ksp.getConvergedReason()
    ksp_residual = ksp.getResidualNorm()

    petsc_nullspace.remove(x_petsc)
    w_new.vector().apply("insert")

    if ksp_reason <= 0:
        raise RuntimeError(
            "Monolithic solver failed at step {}: reason {}, residual {}".format(
                step, ksp_reason, ksp_residual
            )
        )

    u_sol, p_sol = w_new.split(deepcopy=True)
    u_advect.assign(u_sol)

    
    # H(DIV) RECONSTRUCTION
    

    b_hdiv = assemble(L_hdiv)
    for bc in bcs_hdiv:
        bc.apply(b_hdiv)

    z_hdiv.vector().zero()
    # A good restart-safe choice is zero rather than stale numbering/state.
    z_hdiv.vector().apply("insert")

    b_hdiv_petsc = as_backend_type(b_hdiv).vec()
    x_hdiv_petsc = as_backend_type(z_hdiv.vector()).vec()

    hdiv_petsc_nullspace.remove(b_hdiv_petsc)
    hdiv_petsc_nullspace.remove(x_hdiv_petsc)

    hdiv_ksp.solve(b_hdiv_petsc, x_hdiv_petsc)

    hdiv_iterations = hdiv_ksp.getIterationNumber()
    hdiv_reason = hdiv_ksp.getConvergedReason()
    hdiv_residual = hdiv_ksp.getResidualNorm()

    hdiv_petsc_nullspace.remove(x_hdiv_petsc)
    z_hdiv.vector().apply("insert")

    if hdiv_reason <= 0:
        raise RuntimeError(
            "H(div) reconstruction failed at step {}: reason {}, residual {}".format(
                step, hdiv_reason, hdiv_residual
            )
        )

    
    # CONSERVATIVE DG0 SCALAR SOLVE
    

    A_c = assemble(a_c)
    b_c = assemble(L_c)
    solver_scalar.set_operator(A_c)
    scalar_iterations = solver_scalar.solve(c_new.vector(), b_c)

    raw_c_min = c_new.vector().min()
    raw_c_max = c_new.vector().max()

    velocity_component_linf = u_sol.vector().norm("linf")

    if (
        not np.isfinite(velocity_component_linf)
        or not np.isfinite(raw_c_min)
        or not np.isfinite(raw_c_max)
        or velocity_component_linf > velocity_abort_threshold
        or raw_c_min < -scalar_abort_margin
        or raw_c_max > 1.0 + scalar_abort_margin
    ):
        raise RuntimeError(
            "Safety abort at step {} t={}: |u|inf={}, c=[{},{}]".format(
                step,
                t_now,
                velocity_component_linf,
                raw_c_min,
                raw_c_max,
            )
        )

    
    # SCALAR MASS ACCOUNTING
    

    fresh_volume = assemble((Constant(1.0) - c_new) * dx_measure)

    q_nozzle_hd_step = -assemble(
        dot(u_hdiv_ufl, normal) * ds(BOUNDARY_NOZZLE)
    )
    q_return_hd_step = assemble(
        dot(u_hdiv_ufl, normal) * ds(BOUNDARY_RETURN)
    )

    step_flux_tolerance = max(1.0e-13, 1.0e-6 * max(Q_target_now, Q_in))
    if abs(q_return_hd_step - q_nozzle_hd_step) > step_flux_tolerance:
        raise RuntimeError(
            "H(div) boundary flux imbalance at step {}: Qin={}, Qout={}.".format(
                step,
                q_nozzle_hd_step,
                q_return_hd_step,
            )
        )

    fresh_out_return = assemble(
        (Constant(1.0) - c_new)
        * dot(u_hdiv_ufl, normal)
        * ds(BOUNDARY_RETURN)
    )

    fresh_diffusive_in_nozzle = assemble(
        D / h_cell * c_new * ds(BOUNDARY_NOZZLE)
    )

    expected_fresh_physical += (
        q_nozzle_hd_step
        - fresh_out_return
        + fresh_diffusive_in_nozzle
    ) * dt

    
    # ACCEPT STEP
    

    w_n.assign(w_new)
    c_n.assign(c_new)

    
    
    

    if step <= start_step + 5 or step % diagnostic_every == 0:
        div_hdiv_l2 = math.sqrt(
            max(0.0, assemble(div(u_hdiv_ufl) ** 2 * dx_measure))
        )
        int_div_hdiv = assemble(div(u_hdiv_ufl) * dx_measure)

        div_th_l2 = math.sqrt(
            max(0.0, assemble(div(u_sol) ** 2 * dx_measure))
        )
        q_nozzle_th = -assemble(
            dot(u_sol, normal) * ds(BOUNDARY_NOZZLE)
        )
        q_return_th = assemble(
            dot(u_sol, normal) * ds(BOUNDARY_RETURN)
        )

        q_nozzle_hd = q_nozzle_hd_step
        q_return_hd = q_return_hd_step

        fresh_ratio = (
            fresh_volume / expected_fresh_physical
            if expected_fresh_physical > 0.0
            else 0.0
        )

        if fresh_volume > 1.0e-20:
            fresh_x_centroid = assemble(
                x_coord * (Constant(1.0) - c_n) * dx_measure
            ) / fresh_volume
            fresh_y_centroid = assemble(
                y_coord * (Constant(1.0) - c_n) * dx_measure
            ) / fresh_volume
            fresh_z_centroid = assemble(
                z_coord * (Constant(1.0) - c_n) * dx_measure
            ) / fresh_volume
        else:
            fresh_x_centroid = 0.0
            fresh_y_centroid = 0.0
            fresh_z_centroid = 0.0

        lateral_ke = 0.5 * assemble(
            (u_sol[0]**2 + u_sol[1]**2) * dx_measure
        )

        root_print("")
        root_print("================================================")
        root_print("step =", step, " t =", t_now)
        root_print("ramp factor =", ramp_factor)
        root_print("wall elapsed =", time.time() - wall_start, "s")
        root_print("")
        root_print("MONOLITHIC SOLVER")
        root_print("-----------------")
        root_print("KSP iterations =", ksp_iterations)
        root_print("KSP reason     =", ksp_reason)
        root_print("KSP residual   =", ksp_residual)
        root_print("")
        root_print("H(DIV) SCALAR-FLUX RECONSTRUCTION")
        root_print("---------------------------------")
        root_print("H(div) iterations =", hdiv_iterations)
        root_print("H(div) reason     =", hdiv_reason)
        root_print("H(div) residual   =", hdiv_residual)
        root_print("||div(u_hdiv)||_L2 =", div_hdiv_l2)
        root_print("integral div(u_hdiv) =", int_div_hdiv)
        root_print("H(div) nozzle in  =", q_nozzle_hd)
        root_print("H(div) return out =", q_return_hd)
        root_print("")
        root_print("VELOCITY / INCOMPRESSIBILITY")
        root_print("-----------------------------")
        root_print("max velocity DOF component =", velocity_component_linf, "m/s")
        root_print("||div(u_TH)||_L2 =", div_th_l2)
        root_print("||p||_L2 =", norm(p_sol, "L2"))
        root_print("")
        root_print("FLOW BALANCE")
        root_print("------------")
        root_print("Q target now =", Q_target_now)
        root_print("Q nozzle TH   =", q_nozzle_th)
        root_print("Q return TH   =", q_return_th)
        root_print("")
        root_print("CONCENTRATION -- DG0 UPWIND")
        root_print("---------------------------")
        root_print("scalar iterations =", scalar_iterations)
        root_print("min(c) =", raw_c_min)
        root_print("max(c) =", raw_c_max)
        root_print("fresh volume total =", fresh_volume)
        root_print("physical expected fresh added =", expected_fresh_physical)
        root_print("fresh / physical expected =", fresh_ratio)
        root_print("")
        root_print("FULL-3D ASYMMETRY DIAGNOSTICS")
        root_print("-----------------------------")
        root_print("inlet tilt x/y =", tilt_x, tilt_y)
        root_print(
            "fresh centroid x/y/z =",
            fresh_x_centroid,
            fresh_y_centroid,
            fresh_z_centroid,
        )
        root_print("specific lateral KE integral =", lateral_ke)

        if rank == 0:
            with open(DIAGNOSTICS_PATH, "a") as f:
                f.write(
                    (
                        "{},{:.12g},{:.12g},{:.12g},{:.12g},{:.12g},"
                        "{:.12g},{:.12g},{:.12g},{:.12g},{:.12g},{:.12g},"
                        "{:.12g},{:.12g},{:.12g},{:.12g},{:.12g},{:.12g},"
                        "{:.12g},{:.12g},{:.12g}\n"
                    ).format(
                        step,
                        t_now,
                        ramp_factor,
                        tilt_x,
                        tilt_y,
                        velocity_component_linf,
                        div_hdiv_l2,
                        div_th_l2,
                        q_nozzle_hd,
                        q_return_hd,
                        q_nozzle_th,
                        q_return_th,
                        raw_c_min,
                        raw_c_max,
                        fresh_volume,
                        expected_fresh_physical,
                        fresh_ratio,
                        fresh_x_centroid,
                        fresh_y_centroid,
                        fresh_z_centroid,
                        lateral_ke,
                    )
                )
                f.flush()

    
    # PARAVIEW OUTPUT
    

    if step % write_every == 0 or step == end_step:
        write_output(t_now, u_sol, p_sol)

    
    # CHECKPOINT
    

    if rank == 0:
        wall_checkpoint_due = (
            time.time() - last_checkpoint_wall >= checkpoint_wall_seconds
        )
    else:
        wall_checkpoint_due = None
    wall_checkpoint_due = pycomm.bcast(wall_checkpoint_due, root=0)

    if (
        step % checkpoint_every == 0
        or step == end_step
        or wall_checkpoint_due
    ):
        save_checkpoint(step, t_now, expected_fresh_physical)
        last_saved_step = step
        last_checkpoint_wall = time.time()






for output_file in (u_file, p_file, c_file):
    output_file.close()

root_print("")
root_print("================================================")
root_print("FULL RECTANGULAR PHYSICAL-TANK SEGMENT COMPLETE")
root_print("================================================")
root_print("Segment start time =", start_time, "s")
root_print("Segment final time =", end_step * dt, "s")
root_print("Global target time =", T_global, "s")
root_print("Total wall time    =", time.time() - wall_start, "s")
root_print("Latest checkpoint step =", last_saved_step)

if end_step < max_global_step:
    root_print("")
    root_print("Re-submit the SAME Slurm script to continue automatically.")
else:
    root_print("")
    root_print("GLOBAL TARGET REACHED.")

MPI.barrier(comm)
if rank == 0 and lock_handle is not None:
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    lock_handle.close()

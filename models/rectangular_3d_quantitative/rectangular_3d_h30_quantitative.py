# FULL RECTANGULAR 3D -- QUANTITATIVE COMPARISON RUN
# Legacy DOLFIN 2019.1 / PETSc / MPI.
# Domain600x300x300mm; Q20cc/min; 2mm nozzle; dt0.001s; target120s.
# MESH_LEVEL=fine|reference. Taylor-Hood flow and divergence-free RT advection.
# Consistent mixed RT/DG0 diffusion, no scalar clipping. Separate density-proxy
# and passive dye-complement fields distinguish buoyancy from PLIF.
# D_density=1.5e-9 and D_dye=4.14e-10 m^2/s are nominal room-temperature
# literature values, not measurements of the full propan-2-ol mixture.
# Distributed floor return and flush parabolic source are modelling assumptions.
# Two meshes quantify sensitivity; they do not establish convergence order.
# Supplied launcher runs small numerical verification cases before production.
# Analyser uses bottom-up epsilon0.08 height definition and separate sheet/
# whole-volume averages. This study cannot resume earlier solver checkpoints.

from dolfin import *
from petsc4py import PETSc
from mpi4py import MPI as PY_MPI

import glob
import fcntl
import json
import math
import os
import signal
import shutil
import sys
import time

import numpy as np
from mixed_scalar_transport import MixedScalarTransport

run_wall_start = time.monotonic()
stop_signal = 0

def request_checkpoint_stop(signum, frame):
    # No MPI or file I/O inside a signal handler.
    global stop_signal
    stop_signal = int(signum)

signal.signal(signal.SIGUSR1, request_checkpoint_stop)
signal.signal(signal.SIGTERM, request_checkpoint_stop)



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


L_tank = 0.600
W_tank = 0.300
H_tank = 0.300

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
if not math.isfinite(Q_cc_per_min) or Q_cc_per_min <= 0.0:
    raise ValueError("FLOW_RATE_CC_MIN must be finite and positive.")
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
D = float(os.environ.get("SCALAR_DIFFUSIVITY", "1.5e-9"))
D_dye = float(os.environ.get("DYE_DIFFUSIVITY", "4.14e-10"))
expected_dye_physical = 0.0
g = 9.81

rho_fresh = 999.3
rho_salt = 1005.5
beta = (rho_salt - rho_fresh) / rho_salt



# TIME / OUTPUT SETTINGS


dt_default = min(1.0e-3, 1.0e-3 * 20.0 / Q_cc_per_min)
dt = float(os.environ.get("DT", "{:.16g}".format(dt_default)))
if not math.isfinite(dt) or dt <= 0.0:
    raise ValueError("DT must be finite and positive.")
T_global = float(os.environ.get("T_GLOBAL", "120.0"))
ramp_time = 0.5

# Full-domain 3D is expensive. Each scheduler submission advances a bounded
# amount and writes restart files throughout. Re-submit the same job to carry
# on. The default allocation attempts the entire remaining simulation.
segment_duration = float(os.environ.get("SEGMENT_DURATION", str(T_global)))

# Convert physical output intervals to the nearest step.
# At dt=0.001 all of these intervals are represented exactly.
diagnostic_every = max(1, int(round(0.5 / dt)))
write_every = max(1, int(round(1.0 / dt)))
velocity_write_every = max(1, int(round(5.0 / dt)))
checkpoint_every = max(1, int(round(10.0 / dt)))
keep_checkpoints = 3

# Also checkpoint by wall clock. This prevents a short scheduler wall limit
# from killing an expensive full-domain job before 0.1 physical seconds have
# elapsed. The check occurs between completed timesteps.
checkpoint_wall_seconds = float(
    os.environ.get("CHECKPOINT_WALL_SECONDS", "1800.0")
)

wall_time_budget_seconds = float(
    os.environ.get("WALL_TIME_BUDGET_SECONDS", "165600.0")
)

save_margin_seconds = float(os.environ.get("SAVE_MARGIN_SECONDS", "1800.0"))
scheduler_end_time = float(os.environ.get("SLURM_JOB_END_TIME", "0"))
if not math.isfinite(scheduler_end_time) or scheduler_end_time < 0:
    raise ValueError("SLURM_JOB_END_TIME must be a nonnegative finite epoch.")
budget_relative_tolerance = 0.001
pin_checkpoint_time = float(os.environ.get("PIN_CHECKPOINT_TIME", "0"))
if (not math.isfinite(pin_checkpoint_time) or pin_checkpoint_time < 0
        or abs(pin_checkpoint_time/dt-round(pin_checkpoint_time/dt)) > 1e-9):
    raise ValueError("PIN_CHECKPOINT_TIME must be zero or a nonnegative multiple of DT.")
seed_checkpoint = os.environ.get("SEED_CHECKPOINT", "")
seed_provenance = None

# A completely new root for each flow rate. Old reduced-domain runs are
# deliberately ignored and different experimental cases cannot collide.
mesh_level = os.environ.get("MESH_LEVEL", "fine")
if mesh_level not in ("reference", "fine"):
    raise ValueError("MESH_LEVEL must be reference or fine.")
def parameter_tag(value):
    return ("{:g}".format(value)).replace(".", "p").replace("-", "m")
flow_tag = parameter_tag(Q_cc_per_min)
perturbation_tag = parameter_tag(inlet_perturbation)
dt_tag = parameter_tag(dt)
default_output_root = "results_rect3d_quant_{}_Q{}_eps{}_dt{}_D{}_Dye{}".format(
    mesh_level, flow_tag, perturbation_tag, dt_tag,
    parameter_tag(D), parameter_tag(D_dye)
)
script_directory = os.path.dirname(os.path.abspath(__file__))
requested_output_root = os.environ.get("RECT3D_OUTPUT_ROOT", default_output_root)
if os.path.isabs(requested_output_root):
    OUTPUT_ROOT = requested_output_root
else:
    OUTPUT_ROOT = os.path.join(script_directory, requested_output_root)
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoint")
SEGMENTS_DIR = os.path.join(OUTPUT_ROOT, "segments")
SEED_DIR = os.path.join(OUTPUT_ROOT, "seeds")
DIAGNOSTICS_PATH = os.path.join(OUTPUT_ROOT, "diagnostics.csv")
CONFIGURATION_PATH = os.path.join(OUTPUT_ROOT, "configuration.json")
STATUS_PATH = os.path.join(OUTPUT_ROOT, "run_status.json")
DIAGNOSTIC_COLUMNS = ["step","time_s","ramp","tilt_x","tilt_y","max_velocity_component","div_hdiv_l2","div_th_l2","q_nozzle_hdiv","q_return_hdiv","q_nozzle_th","q_return_th","c_min","c_max","fresh_volume_m3","fresh_expected_m3","fresh_ratio","x_fresh_centroid_m","y_fresh_centroid_m","z_fresh_centroid_m","specific_lateral_ke_integral","dye_min","dye_max","dye_equivalent_volume_m3","dye_expected_m3","dye_budget_relative_error","scalar_budget_relative_error","scalar_true_residual","dye_true_residual","gross_inflow_m3_s","gross_outflow_m3_s","wall_leak_m3_s","density_diffusive_gain_m3_s","dye_diffusive_gain_m3_s"]


# Safety thresholds.
velocity_abort_threshold = 1.0
scalar_abort_margin = 1.0e-3

# This changes acceptance checks only, not the state equations or the fields.
# Jobs 18989285 and 18989361 measured a local density overshoot at startup;
# it peaked at 0.252557% and returned below 0.1% at t=0.074 s. A residual
# correction at t=0.051 s changed concentration by at most 6e-11.
FINE_STARTUP_BOUNDS_POLICY = {
    "revision": "fine-startup-bounds-2026-09-18",
    "evidence_jobs": ["18989285", "18989361"],
    "end_time_s_exclusive": 0.100,
    "density_upper_margin": 0.003,
    "ordinary_scalar_margin": 0.001,
    "maximum_cells_above_ordinary_limit": 64,
    "maximum_volume_above_ordinary_limit_m3": 1.0e-10,
    "maximum_integrated_excess_above_one_m3": 2.0e-13,
    "maximum_cell_centre_radius_m": 0.0015,
    "maximum_cell_centre_height_m": 0.00025,
    "scope": "fine mesh, dt=0.001, Q=20, nominal diffusivities, zero perturbation",
    "state_equations_unchanged": True,
}


def fine_startup_case_matches(level, timestep, flow, density_D, dye_D, perturbation):
    return (level == "fine"
            and math.isclose(timestep, .001, rel_tol=0., abs_tol=1e-15)
            and math.isclose(flow, 20., rel_tol=0., abs_tol=1e-12)
            and math.isclose(density_D, 1.5e-9, rel_tol=0., abs_tol=1e-20)
            and math.isclose(dye_D, 4.14e-10, rel_tol=0., abs_tol=1e-20)
            and perturbation == 0.)


def fine_startup_allowance_active(time_s, case_matches):
    return bool(case_matches and 0. < time_s < FINE_STARTUP_BOUNDS_POLICY["end_time_s_exclusive"])


def density_upper_margin_at(time_s, case_matches):
    if fine_startup_allowance_active(time_s, case_matches):
        return FINE_STARTUP_BOUNDS_POLICY["density_upper_margin"]
    return scalar_abort_margin


def assess_startup_density(values, coordinates, volumes, communicator, mpi):
    """Check location, extent and total excess without changing concentrations."""
    values = np.asarray(values, dtype=float)
    coordinates, volumes = np.asarray(coordinates), np.asarray(volumes)
    if (values.shape != volumes.shape or coordinates.shape != (len(values), 3)
            or not np.all(np.isfinite(coordinates))
            or not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.)):
        raise ValueError("Invalid owned coordinates/volumes for the startup guard.")
    policy = FINE_STARTUP_BOUNDS_POLICY
    finite = np.isfinite(values)
    beyond = finite & (values > 1. + policy["ordinary_scalar_margin"])
    radius2 = coordinates[:, 0]**2 + coordinates[:, 1]**2
    outside_region = beyond & (
        (radius2 > policy["maximum_cell_centre_radius_m"]**2)
        | (coordinates[:, 2] > policy["maximum_cell_centre_height_m"])
        | (coordinates[:, 2] < 0.))
    local = np.array([
        np.sum(~finite), np.sum(beyond), np.sum(outside_region),
        np.sum(volumes[beyond]),
        np.dot(volumes, np.maximum(np.where(finite, values, 1.)-1., 0.)),
    ], dtype=float)
    total = communicator.allreduce(local, op=mpi.SUM)
    result = dict(
        nonfinite_cells=int(round(total[0])),
        cells_above_ordinary_limit=int(round(total[1])),
        cells_above_limit_outside_nozzle_region=int(round(total[2])),
        volume_above_ordinary_limit_m3=float(total[3]),
        integrated_excess_above_one_m3=float(total[4]),
    )
    result["extent_checks_passed"] = bool(
        result["nonfinite_cells"] == 0
        and result["cells_above_limit_outside_nozzle_region"] == 0
        and result["cells_above_ordinary_limit"] <= policy["maximum_cells_above_ordinary_limit"]
        and result["volume_above_ordinary_limit_m3"] <= policy["maximum_volume_above_ordinary_limit_m3"]
        and result["integrated_excess_above_one_m3"] <= policy["maximum_integrated_excess_above_one_m3"])
    return result


def write_startup_bounds_report(path, report, communicator, root_rank):
    error = None
    if root_rank == 0:
        try:
            with open(path+".tmp", "w") as stream:
                json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(path+".tmp", path)
        except Exception as exc:
            error = "Could not save startup-bounds report: {}".format(exc)
    error = communicator.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(error)


fine_startup_case_eligible = fine_startup_case_matches(
    mesh_level, dt, Q_cc_per_min, D, D_dye, inlet_perturbation)
fine_startup_bounds_policy = dict(FINE_STARTUP_BOUNDS_POLICY, enabled=fine_startup_case_eligible)

# Restart reconstruction tolerances.
coord_decimals = 13
fingerprint_rtol = 1.0e-8
fingerprint_atol = 1.0e-12

# Explicit state-evolution identity. Change this string whenever the weak forms,
# boundary recipe, element families/orders, or refinement recipe change.
solver_schema = "rect3d_quant_dual_scalar_mixed_diffusion_20260907a"
scalar_discretization = "dg0_upwind_mixed_rt_diffusion_dual_scalar"
mesh_recipe = {
    "reference": "box40x20x33;z14bulk_3transition_16ceiling;plume2_nozzle4",
    "fine": "box48x24x42;z18bulk_3transition_21ceiling;plume2_nozzle4",
}[mesh_level]

for parameter_name, parameter_value in (
    ("SCALAR_DIFFUSIVITY", D),
    ("DYE_DIFFUSIVITY", D_dye),
    ("DT", dt),
    ("T_GLOBAL", T_global),
    ("SEGMENT_DURATION", segment_duration),
    ("CHECKPOINT_WALL_SECONDS", checkpoint_wall_seconds),
    ("SAVE_MARGIN_SECONDS", save_margin_seconds),
):
    if not math.isfinite(parameter_value) or parameter_value <= 0.0:
        raise ValueError(parameter_name + " must be finite and positive.")
if not math.isfinite(wall_time_budget_seconds) or wall_time_budget_seconds < 0.0:
    raise ValueError("WALL_TIME_BUDGET_SECONDS must be finite and nonnegative.")

# Fail before allocating a large mesh for invalid time settings.
for parameter_name, duration in (
    ("T_GLOBAL", T_global), ("SEGMENT_DURATION", segment_duration)
):
    ratio = duration / dt
    if not math.isfinite(ratio) or ratio < 1.0 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(parameter_name + " must be an integer multiple of DT.")
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

def write_run_status(step, time_value, complete=False, stop_reason=None):
    error = None
    if rank == 0:
        try:
            status = {
                "solver_schema": solver_schema, "step": int(step),
                "time": float(time_value), "target_time_s": float(T_global),
                "complete": bool(complete), "stop_reason": stop_reason,
                "checkpoint_step": int(globals().get("last_saved_step", 0)),
                "job_id": os.environ.get("SLURM_JOB_ID", "interactive"),
                "dt": float(dt), "flow_rate_cc_per_min": Q_cc_per_min,
                "dimensions_m": [L_tank, W_tank, H_tank],
                "mesh_level": mesh_level, "density_diffusivity": D, "dye_diffusivity": D_dye,
                "seed_provenance": seed_provenance,
                "startup_bounds_policy": fine_startup_bounds_policy,
                "updated_epoch": time.time(),
            }
            temp = STATUS_PATH + ".tmp"
            with open(temp, "w") as f:
                json.dump(status, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, STATUS_PATH)
        except Exception as exc:
            error = "Could not write run status: {}".format(exc)
    error = pycomm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(error)








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
        os.makedirs(SEED_DIR, exist_ok=True)
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



def validate_checkpoint_metadata(meta, check_mesh=True):
    checks = [
        (int(meta.get("restart_format_version", -1)) == 5, "format version"),
        (str(meta.get("solver_schema", "")) == solver_schema, "solver schema"),
        (str(meta.get("mesh_recipe", "")) == mesh_recipe, "mesh recipe"),
        (abs(float(meta["dt"]) - dt) <= 1.0e-15, "dt"),
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
        (abs(float(meta["D"]) - D) <= 1.0e-20, "D"),
        (abs(float(meta["D_dye"]) - D_dye) <= 1.0e-20, "D_dye"),
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

    if check_mesh:
        checks.extend([
            (int(meta["mesh_vertices"]) == mesh_vertices_global, "mesh vertices"),
            (int(meta["mesh_cells"]) == mesh_cells_global, "mesh cells"),
        ])
    if not all(math.isfinite(float(meta[key])) for key in ("expected_fresh_physical", "expected_dye_physical")):
        raise RuntimeError("Checkpoint freshwater budget is non-finite.")

    failed = [name for ok, name in checks if not ok]
    if failed:
        raise RuntimeError(
            "Checkpoint is incompatible with this code: " + ", ".join(failed)
        )


def _read_npz_metadata(path, validate_payload=False):
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
            "dye_coords", "dye_values",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(
                "checkpoint is missing arrays: " + ", ".join(missing)
            )
        metadata = json.loads(str(data["meta_json"].item()))
        if validate_payload:
            # Reading each member also checks its compressed payload/CRC.
            for name in ("ux", "uy", "uz", "p", "c", "dye"):
                coords, values = data[name + "_coords"], data[name + "_values"]
                if coords.ndim != 2 or coords.shape[1] != 3:
                    raise RuntimeError("Invalid coordinates in checkpoint: " + name)
                if values.ndim != 1 or len(values) != len(coords):
                    raise RuntimeError("Invalid values in checkpoint: " + name)
                if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(values)):
                    raise RuntimeError("Non-finite checkpoint payload: " + name)
        return metadata



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
            meta = _read_npz_metadata(path, validate_payload=True)
            if int(meta.get("restart_format_version", -1)) != 5:
                continue
            return path, meta
        except Exception:
            continue

    if all_state_paths:
        raise RuntimeError(
            "Checkpoint files exist, but none has a complete compatible "
            "restart-format-5 payload. Refusing to start clean over them."
        )

    return None, None


def load_seed_root(path):
    """Start a separate, finer-time-step branch without modifying its seed."""
    path = os.path.abspath(path)
    meta = _read_npz_metadata(path, validate_payload=True)
    original_dt = float(meta["dt"])
    physical_time = float(meta["time"])
    original_step = int(meta["step"])
    if (not math.isfinite(original_dt) or original_dt <= 0
            or not math.isfinite(physical_time) or physical_time <= 0
            or original_step < 0 or abs(physical_time-original_step*original_dt) > 5e-12):
        raise RuntimeError("Invalid seed step/time metadata.")
    if dt >= original_dt:
        raise RuntimeError("SEED_CHECKPOINT requires a strictly smaller timestep.")
    new_step = int(round(physical_time/dt))
    if abs(new_step*dt-physical_time) > 5e-12 or physical_time >= T_global:
        raise RuntimeError("Seed time must lie on the new timestep grid and precede T_GLOBAL.")
    adapted = dict(meta)
    adapted.update(dt=dt, step=new_step, T_global=T_global)
    # Only time-step indexing changes; mesh, physical model and field values do not.
    validate_checkpoint_metadata(adapted, check_mesh=False)
    adapted["seed_provenance"] = {
        "source_checkpoint": path, "source_dt_s": original_dt,
        "source_step": original_step, "physical_time_s": physical_time,
        "branch_dt_s": dt,
        "scope": "Shared earlier history; subsequent timestep sensitivity only",
    }
    return path, adapted


if rank == 0:
    try:
        restart_path, restart_meta = find_latest_checkpoint_root()
        if restart_path is None and seed_checkpoint:
            restart_path, restart_meta = load_seed_root(seed_checkpoint)
        elif restart_path is not None and seed_checkpoint:
            provenance = restart_meta.get("seed_provenance") or {}
            if provenance.get("source_checkpoint") != os.path.abspath(seed_checkpoint):
                raise RuntimeError("Existing branch was not started from the requested seed.")
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
seed_provenance = (restart_meta or {}).get("seed_provenance")

# Validate physical identity before allocating any mesh. A completed case exits
# here so unused jobs in a submitted chain take only the startup/checkpoint read.
if restart_path is not None:
    validate_checkpoint_metadata(restart_meta, check_mesh=False)
    saved_step = int(restart_meta["step"])
    saved_time = float(restart_meta["time"])
    if (saved_step < 0 or not math.isfinite(saved_time)
            or abs(saved_time - saved_step * dt) > 5e-12):
        raise RuntimeError("Checkpoint step/time metadata are inconsistent.")
    requested_final_step = int(round(T_global / dt))
    if saved_step > requested_final_step:
        raise RuntimeError("Checkpoint is beyond the requested T_GLOBAL.")
    if saved_step == requested_final_step:
        last_saved_step = saved_step
        write_run_status(saved_step, saved_time, complete=True)
        root_print("GLOBAL TARGET ALREADY REACHED at", saved_time, "s; exiting.")
        MPI.barrier(comm)
        if rank == 0 and lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
        sys.exit(0)

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

# Both meshes cover identical geometry and the same source-refinement regions.
# Fine reduces horizontal, lower vertical and ceiling vertical spacing.
if mesh_level == "reference":
    nx_base, ny_base, bulk_layers, ceiling_layers = 40, 20, 14, 16
else:
    nx_base, ny_base, bulk_layers, ceiling_layers = 48, 24, 18, 21
z_layers = np.concatenate((
    np.linspace(0.0, 0.210, bulk_layers + 1),
    np.array([0.222, 0.231, 0.237]),
    np.linspace(0.237, H_tank, ceiling_layers + 1)[1:],
))
nz_base = len(z_layers) - 1
if (abs(z_layers[0]) > 1e-15 or abs(z_layers[-1]-H_tank) > 1e-15
        or np.any(np.diff(z_layers) <= 0)):
    raise RuntimeError("Invalid graded vertical mesh.")
if not np.allclose(np.diff(z_layers)[-ceiling_layers:], 0.063/ceiling_layers,
                   atol=1e-14, rtol=0):
    raise RuntimeError("Ceiling layers do not have the prescribed spacing.")

mesh = BoxMesh.create(
    comm,
    [Point(x_min, y_min, z_min), Point(x_max, y_max, z_max)],
    [nx_base, ny_base, nz_base], CellType.Type.tetrahedron,
)
# Apply the same deterministic map to owned and ghost vertices on every rank.
mesh.coordinates()[:, 2] = np.interp(
    mesh.coordinates()[:, 2], np.linspace(0.0, H_tank, nz_base + 1), z_layers
)
mesh.bounding_box_tree().build(mesh)
root_print("Base cells before local refinement =", nx_base * ny_base * nz_base * 6)
root_print("Ceiling base vertical spacing =", np.diff(z_layers)[-1], "m")
root_print("Ceiling refinement begins at z =", z_layers[-ceiling_layers-1], "m")
print_mesh_info(mesh, "INITIAL GRADED RECTANGULAR MESH")

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
ds = Measure("ds", domain=mesh, subdomain_data=boundaries,
             metadata={"quadrature_degree": 8})
dx_measure = Measure("dx", domain=mesh)
dS_measure = Measure("dS", domain=mesh)
x_coord, y_coord, z_coord = SpatialCoordinate(mesh)

volume_mesh = assemble(Constant(1.0) * dx_measure)
if not math.isclose(volume_mesh, L_tank * W_tank * H_tank, rel_tol=1e-10, abs_tol=1e-13):
    raise RuntimeError("Graded mesh volume does not match the 0.054 m^3 box.")
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
    Calibrate the discrete trace using its total gross inflow and outflow.

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

    if not all(math.isfinite(value) and value > 0
               for value in (inlet_amp, return_amp)):
        raise RuntimeError(label + " calibration has nonpositive amplitudes.")
    full = Function(space)
    full.vector().zero()
    full.vector().axpy(float(inlet_amp), unit_in.vector())
    full.vector().axpy(float(return_amp), unit_ret.vector())
    full.vector().apply("insert")
    un = dot(full, normal)
    gross_in = assemble(0.5 * (abs(un)-un) * ds)
    if not math.isfinite(gross_in) or gross_in <= 0:
        raise RuntimeError(label + " calibration has no positive gross inflow.")

    # Net flux on a midpoint-marked nozzle need not equal true total inflow.
    # Rescale both amplitudes together: retain zero net flux and set gross Q.
    factor = Q_in / gross_in
    inlet_amp *= factor
    return_amp *= factor
    full_vector = full.vector()
    full_vector *= factor
    full.vector().apply("insert")
    q_in = assemble(0.5 * (abs(un)-un) * ds)
    q_out = assemble(0.5 * (abs(un)+un) * ds)
    flux_tolerance = max(1e-14, 1e-8*Q_in)
    if (abs(q_in-Q_in) > flux_tolerance or abs(q_out-Q_in) > flux_tolerance
            or abs(q_in-q_out) > flux_tolerance):
        raise RuntimeError(
            "{} gross-flux calibration failed: Qin={}, Qout={}, target={}".format(
                label, q_in, q_out, Q_in
            )
        )
    root_print(label, "gross inflow/outflow =", q_in, q_out)
    root_print(label, "inflow on nominal return facets =",
               assemble(0.5*(abs(un)-un)*ds(BOUNDARY_RETURN)))
    return float(inlet_amp), float(return_amp), float(q_in), float(q_out)


th_inlet_amp, th_return_amp, th_q_nozzle, th_q_return = (
    calibrated_boundary_pair(V, "Taylor-Hood P2")
)

root_print("")
root_print("================================================")
root_print("FUNCTION SPACES")
root_print("================================================")
root_print("Mixed velocity-pressure DOFs =", W.dim())
root_print("Concentration DOFs           =", C.dim())
root_print("Total flow + two scalar DOFs  =", W.dim() + 2*C.dim())
root_print("Q target                      =", Q_in)
root_print("analytical mean inlet speed   =", Q_in / (math.pi * a_nozzle**2))
root_print("analytical peak inlet speed   =", 2.0 * Q_in / (math.pi * a_nozzle**2))
root_print("P2 inlet amplitude            =", th_inlet_amp)
root_print("P2 return amplitude           =", th_return_amp)
root_print("P2 gross boundary inflow       =", th_q_nozzle)
root_print("P2 gross boundary outflow      =", th_q_return)
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
root_print("H(div) gross inflow  =", hdiv_q_nozzle)
root_print("H(div) gross outflow =", hdiv_q_return)
root_print("H(div) net full =", hdiv_q_return - hdiv_q_nozzle)

mesh_hmin_global = MPI.min(comm, mesh.hmin())
mesh_hmax_global = MPI.max(comm, mesh.hmax())

def write_configuration_and_diagnostics_header():
    error_message = None

    if rank == 0:
        try:
            configuration = {
                "model": "Rectangular quantitative comparison with separate density and dye transport",
                "startup_bounds_policy": fine_startup_bounds_policy,
                "mesh_level": mesh_level,
                "scalar_discretization": scalar_discretization,
                "gravity_m_per_s2": g,
                "ramp_time_s": ramp_time,
                "perturbation_period_x_s": perturbation_period_x,
                "perturbation_period_y_s": perturbation_period_y,
                "dye_diffusivity_m2_per_s": D_dye,
                "diffusivity_basis": "Nominal literature values; full propan-2-ol mixture is approximated",
                "source_profile": "flush parabolic, total inflow calibrated",
                "seed_provenance": seed_provenance,
                "pin_checkpoint_time_s": pin_checkpoint_time,
                "legacy_fenics": "DOLFIN 2019.1",
                "solver_schema": solver_schema,
                "mesh_recipe": mesh_recipe,
                "base_vertical_nodes_m": z_layers.tolist(),
                "ceiling_base_dz_m": float(np.diff(z_layers)[-1]),
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
                "wall_time_budget_seconds": wall_time_budget_seconds,
                "save_margin_seconds": save_margin_seconds,
                "output_interval_s": write_every * dt,
                "velocity_output_interval_s": velocity_write_every * dt,
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
                    f.write(",".join(DIAGNOSTIC_COLUMNS) + "\n")
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
# changes the instantaneous jet direction by at most sqrt(2) times the
# inlet_perturbation in radians
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
c_dye_n = Function(C)
c_dye_new = Function(C)
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



def state_fingerprint(w_state, c_state, dye_state):
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
        "dye_integral": float(assemble(dye_state * dx_measure)),
        "dye_min": float(dye_state.vector().min()),
        "dye_max": float(dye_state.vector().max()),
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
        ("dye", c_dye_n),
    ):
        coords, values = gather_function_canonical(function, name)
        if rank == 0:
            fields[name + "_coords"] = coords
            fields[name + "_values"] = values

    fingerprint = state_fingerprint(w_n, c_n, c_dye_n)

    checkpoint_error = None
    if rank == 0:
        try:
            meta = {
                "restart_format_version": 5,
                "step": int(step),
                "time": float(time_value),
                "dt": float(dt),
                "T_global": float(T_global),
                "startup_bounds_policy": fine_startup_bounds_policy,
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
                "D_dye": float(D_dye),
                "expected_dye_physical": float(expected_dye_physical),
                "seed_provenance": seed_provenance,
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

            if pin_checkpoint_time > 0 and abs(time_value-pin_checkpoint_time) <= 5e-12:
                seed_name = "state_time_{}s.npz".format(parameter_tag(pin_checkpoint_time))
                seed_path = os.path.join(SEED_DIR, seed_name)
                with open(final_path, "rb") as source_file, open(seed_path+".tmp", "wb") as seed_file:
                    shutil.copyfileobj(source_file, seed_file)
                    seed_file.flush()
                    os.fsync(seed_file.fileno())
                os.replace(seed_path+".tmp", seed_path)
                root_print("PINNED TIMESTEP-CHECK SEED:", seed_path)

            # Retain the newest few complete checkpoints so a killed/corrupted
            # write can never destroy the previous valid state.
            complete = []
            for candidate in sorted(
                glob.glob(os.path.join(CHECKPOINT_DIR, "state_step_*.npz")),
                reverse=True,
            ):
                try:
                    candidate_meta = _read_npz_metadata(candidate)
                    if int(candidate_meta.get("restart_format_version", -1)) == 5:
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
        except Exception as exc:
            checkpoint_error = "Checkpoint write failed: {}".format(exc)

    checkpoint_error = pycomm.bcast(checkpoint_error, root=0)
    if checkpoint_error is not None:
        raise RuntimeError(checkpoint_error)

    MPI.barrier(comm)






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
                        "dye_coords",
                        "dye_values",
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
        ("dye", c_dye_n),
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
    c_dye_new.assign(c_dye_n)

    current = state_fingerprint(w_n, c_n, c_dye_n)
    saved_fp = meta["fingerprint"]

    bad = []
    for key in current:
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
    root_print("Loaded dye range    =", current["dye_min"], "to", current["dye_max"])



# INITIALISE OR RESTORE STATE


if restart_path is None:
    w_n.vector().zero()
    w_n.vector().apply("insert")
    w_new.assign(w_n)

    c_n.assign(Constant(1.0))
    c_dye_n.assign(Constant(1.0))
    c_new.assign(c_n)
    c_dye_new.assign(c_dye_n)

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

    step0_meta, step0_error = None, None
    if rank == 0:
        try:
            step0_meta = _read_npz_metadata(step0_path)
        except Exception as exc:
            step0_error = "Step-0 checkpoint read failed: {}".format(exc)
    step0_error = pycomm.bcast(step0_error, root=0)
    if step0_error is not None:
        raise RuntimeError(step0_error)
    step0_meta = pycomm.bcast(step0_meta, root=0)

    load_checkpoint(step0_path, step0_meta)
    root_print("PHYSICAL-TANK RESTART STEP-0 ROUND-TRIP SELF-TEST PASSED.")
else:
    load_checkpoint(restart_path, restart_meta)

    start_step = int(restart_meta["step"])
    start_time = float(restart_meta["time"])
    expected_fresh_physical = float(restart_meta["expected_fresh_physical"])
    expected_dye_physical = float(restart_meta["expected_dye_physical"])

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

            column_count = len(header.rstrip().split(","))
            for line_index, line in enumerate(lines[1:], start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                fields = stripped.split(",")
                try:
                    if len(fields) != column_count or not line.endswith("\n"):
                        raise ValueError("incomplete diagnostics row")
                    row_step = int(fields[0])
                    if not all(math.isfinite(float(value)) for value in fields):
                        raise ValueError("non-finite diagnostics row")
                except ValueError:
                    if line_index == len(lines) - 1:
                        root_print("Discarding an incomplete final diagnostics row.")
                        continue
                    raise
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

# HYPRE intermittently returned PETSc error 76 in the production flow solve
# on Blue Pebble, on several nodes and from several valid checkpoints.  PETSc
# block Jacobi with a local ILU solve on each MPI subdomain is therefore the
# production default.  This changes only the linear preconditioner; the
# assembled equations, tolerances and accepted state are unchanged.  Retain
# the alternatives as explicit overrides for controlled comparisons.
mono_block_pc = os.environ.get("RECT3D_MONO_BLOCK_PC", "bjacobi").strip().lower()
if mono_block_pc not in ("bjacobi", "hypre", "gamg"):
    raise ValueError("RECT3D_MONO_BLOCK_PC must be bjacobi, hypre or gamg.")

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
opts["mono_fieldsplit_0_pc_type"] = mono_block_pc
opts["mono_fieldsplit_1_ksp_type"] = "preonly"
opts["mono_fieldsplit_1_pc_type"] = mono_block_pc
if mono_block_pc == "bjacobi":
    opts["mono_fieldsplit_0_sub_ksp_type"] = "preonly"
    opts["mono_fieldsplit_0_sub_pc_type"] = "ilu"
    opts["mono_fieldsplit_1_sub_ksp_type"] = "preonly"
    opts["mono_fieldsplit_1_sub_pc_type"] = "ilu"

ksp.setFromOptions()
root_print(
    "MONOLITHIC FLOW BLOCK PRECONDITIONER:", mono_block_pc,
    "(local ILU enabled)" if mono_block_pc == "bjacobi" else ""
)



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


# Each scalar has its own algebraic state; both reuse the RT1 x DG0 FE space.
scalar_transport = MixedScalarTransport(
    mesh, C, c_n, u_hdiv_ufl, D, dt, boundaries, mixed_space=Z_hdiv
)
dye_transport = MixedScalarTransport(
    mesh, C, c_dye_n, u_hdiv_ufl, D_dye, dt, boundaries, mixed_space=Z_hdiv
)
un_bnd_hdiv = dot(u_hdiv_ufl, normal)
outflow_speed = 0.5 * (un_bnd_hdiv + abs(un_bnd_hdiv))
inflow_speed = 0.5 * (un_bnd_hdiv - abs(un_bnd_hdiv))


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

segment_directory_error = None
if rank == 0:
    try:
        os.makedirs(segment_dir, exist_ok=True)
    except Exception as exc:
        segment_directory_error = "Could not create segment directory: {}".format(exc)
segment_directory_error = pycomm.bcast(segment_directory_error, root=0)
if segment_directory_error is not None:
    raise RuntimeError(segment_directory_error)
MPI.barrier(comm)

root_print("")
root_print("Segment output directory:")
root_print(segment_dir)

u_file = XDMFFile(comm, os.path.join(segment_dir, "u_taylor_hood.xdmf"))
p_file = XDMFFile(comm, os.path.join(segment_dir, "p.xdmf"))
c_file = XDMFFile(comm, os.path.join(segment_dir, "c_dg0.xdmf"))
dye_file = XDMFFile(comm, os.path.join(segment_dir, "dye_complement_dg0.xdmf"))
for output_file in (u_file, p_file, c_file, dye_file):
    output_file.parameters["flush_output"] = True
    output_file.parameters["functions_share_mesh"] = True
    output_file.parameters["rewrite_function_mesh"] = False



def write_output(time_value, u_state=None, p_state=None, include_velocity=True):
    if u_state is None or p_state is None:
        u_state, p_state = w_n.split(deepcopy=True)

    u_state.rename("velocity", "u")
    p_state.rename("pressure", "p")
    c_n.rename("concentration", "c")
    c_dye_n.rename("dye_complement", "c_dye")

    # Do not project div(u) into a continuous pressure space here.
    # FEniCS project() launches an extra global PETSc solve and can fail
    # independently of the physical solver. Divergence is diagnosed directly
    # from the accepted velocity in the timestep diagnostics below.
    if include_velocity:
        u_file.write(u_state, time_value)
        p_file.write(p_state, time_value)
    c_file.write(c_n, time_value)
    dye_file.write(c_dye_n, time_value)


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
root_print("STARTING 30 CM RECTANGULAR THESIS RUN")
root_print("================================================")
root_print("dt                 =", dt)
root_print("global target time =", T_global, "s")
root_print("start time         =", start_time, "s")
root_print("advance this job   <=", segment_duration, "s")
root_print("ramp time          =", ramp_time)
root_print("nu                 =", nu)
root_print("D density / dye    =", D, D_dye)
root_print("beta               =", beta)
root_print("domain             =", L_tank, "x", W_tank, "x", H_tank, "m")
root_print("quarter domain     = False (no symmetry planes)")
root_print("flow rate          =", Q_cc_per_min, "cc/min")
root_print("inlet perturbation =", inlet_perturbation)
root_print("scalars            = conservative DG0 upwind + consistent mixed RT diffusion")
root_print("start step         =", start_step)
root_print("end step this job  =", end_step)

initial_scalar_integral = volume_mesh
initial_fresh_volume = 0.0

startup_window_end_step = int(round(FINE_STARTUP_BOUNDS_POLICY["end_time_s_exclusive"] / dt))
startup_bounds_report = None
if fine_startup_case_eligible and start_step < startup_window_end_step:
    startup_owned_coords = owned_dof_coordinates(C)
    startup_owned_volumes = assemble(TestFunction(C) * dx_measure).get_local()
    local_layout_ok = int(
        startup_owned_coords.shape == (len(startup_owned_volumes), 3)
        and len(startup_owned_volumes) == len(c_n.vector().get_local())
        and np.all(np.isfinite(startup_owned_coords))
        and np.all(np.isfinite(startup_owned_volumes))
        and np.all(startup_owned_volumes > 0.))
    if not pycomm.allreduce(local_layout_ok, op=PY_MPI.MIN):
        raise RuntimeError("Startup guard coordinates and owned DG0 volumes do not align.")
    startup_total_volume = pycomm.allreduce(float(np.sum(startup_owned_volumes)), op=PY_MPI.SUM)
    if not math.isclose(startup_total_volume, volume_mesh, rel_tol=1e-10, abs_tol=1e-14):
        raise RuntimeError("Startup guard volumes do not sum to the tank volume.")
    startup_job_tag = os.environ.get("SLURM_JOB_ID", "")
    if not startup_job_tag.isdigit():
        startup_job_tag = "local_" + str(os.getpid())
    startup_bounds_path = os.path.join(OUTPUT_ROOT, "startup_bounds_"+startup_job_tag+".json")
    startup_bounds_report = dict(
        policy=fine_startup_bounds_policy, job_id=startup_job_tag,
        case_output_root=OUTPUT_ROOT, segment_start_step=int(start_step),
        segment_start_time_s=float(start_time), steps=[],
        window_completed=False,
        note="Accepted production steps only; all raw concentration values are retained.")
    root_print("FINE STARTUP BOUNDS: upper density margin 0.003 only before 0.100 s; "
               "nozzle location, cell-count, volume and integral checks enabled.")
    root_print("Startup-bounds report:", startup_bounds_path)

wall_start = time.time()
last_checkpoint_wall = wall_start
last_saved_step = start_step
completed_step = start_step
stop_reason = None
write_run_status(start_step, start_time, complete=start_step >= max_global_step)






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

    # Warm-start from the preceding solve within this job. The Function is
    # initially zero after startup/restart; no old parallel numbering is loaded.
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
    

    scalar_iterations = scalar_transport.solve(c_new)
    dye_iterations = dye_transport.solve(c_dye_new)

    raw_c_min = c_new.vector().min()
    raw_c_max = c_new.vector().max()
    raw_dye_min = c_dye_new.vector().min()
    raw_dye_max = c_dye_new.vector().max()

    velocity_component_linf = u_sol.vector().norm("linf")
    startup_allowance = fine_startup_allowance_active(t_now, fine_startup_case_eligible)
    density_upper_margin = density_upper_margin_at(t_now, fine_startup_case_eligible)
    startup_metrics = None
    if startup_bounds_report is not None and step <= startup_window_end_step:
        startup_metrics = assess_startup_density(
            c_new.vector().get_local(), startup_owned_coords, startup_owned_volumes, pycomm, PY_MPI)

    if (
        not np.isfinite(velocity_component_linf)
        or not np.isfinite(raw_c_min)
        or not np.isfinite(raw_c_max)
        or velocity_component_linf > velocity_abort_threshold
        or raw_c_min < -scalar_abort_margin
        or raw_c_max > 1.0 + density_upper_margin
        or (startup_allowance and (startup_metrics is None or not startup_metrics["extent_checks_passed"]))
        or not math.isfinite(raw_dye_min) or not math.isfinite(raw_dye_max)
        or raw_dye_min < -scalar_abort_margin
        or raw_dye_max > 1.0 + scalar_abort_margin
    ):
        raise RuntimeError(
            "Safety abort at step {} t={}: |u|inf={}, c=[{},{}], dye=[{},{}]; "
            "density upper margin={}; startup metrics={}".format(
                step,
                t_now,
                velocity_component_linf,
                raw_c_min,
                raw_c_max,
                raw_dye_min, raw_dye_max,
                density_upper_margin, startup_metrics,
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

    gross_in = -assemble(inflow_speed * ds)
    gross_out = assemble(outflow_speed * ds)
    wall_leak = assemble(abs(un_bnd_hdiv) * (ds(BOUNDARY_TOP)+ds(BOUNDARY_WALL)))
    if (not all(math.isfinite(v) for v in (gross_in, gross_out, wall_leak))
            or abs(gross_in-Q_target_now) > step_flux_tolerance
            or abs(gross_out-gross_in) > step_flux_tolerance
            or wall_leak > step_flux_tolerance):
        raise RuntimeError("Gross boundary throughput/continuity check failed at step {}.".format(step))

    # All exterior inflow is source concentration zero for each complement.
    # Whole-boundary accounting also handles source support on crossing facets.
    fresh_out = assemble(outflow_speed * (1.0-c_new) * ds)
    fresh_diffusive_gain = assemble(dot(scalar_transport.diffusive_flux, normal) * ds)
    dye_out = assemble(outflow_speed * (1.0-c_dye_new) * ds)
    dye_diffusive_gain = assemble(dot(dye_transport.diffusive_flux, normal) * ds)
    expected_fresh_physical += dt * (gross_in-fresh_out+fresh_diffusive_gain)
    expected_dye_physical += dt * (gross_in-dye_out+dye_diffusive_gain)
    dye_volume = assemble((1.0-c_dye_new) * dx_measure)
    budget_relative_error = abs(fresh_volume-expected_fresh_physical) / max(
        abs(expected_fresh_physical), Q_in*dt, 1e-30
    )
    dye_budget_relative_error = abs(dye_volume-expected_dye_physical) / max(
        abs(expected_dye_physical), Q_in*dt, 1e-30
    )
    if (not all(math.isfinite(v) for v in (budget_relative_error, dye_budget_relative_error))
            or (t_now >= 1.0 and max(budget_relative_error, dye_budget_relative_error)
                > budget_relative_tolerance)):
        raise RuntimeError(
            "Scalar budget error exceeds {} at t={}: density={}, dye={}.".format(
                budget_relative_tolerance, t_now, budget_relative_error, dye_budget_relative_error
            )
        )

    
    # ACCEPT STEP
    

    w_n.assign(w_new)
    c_n.assign(c_new)
    c_dye_n.assign(c_dye_new)
    completed_step = step
    startup_window_just_completed = bool(fine_startup_case_eligible and step == startup_window_end_step)
    if startup_metrics is not None:
        startup_row = dict(
            startup_metrics, step=int(step), time_s=float(t_now),
            allowance_active=startup_allowance, density_upper_margin=density_upper_margin,
            density_min=float(raw_c_min), density_max=float(raw_c_max),
            dye_min=float(raw_dye_min), dye_max=float(raw_dye_max),
            density_budget_relative_error=float(budget_relative_error),
            dye_budget_relative_error=float(dye_budget_relative_error),
            density_true_residual=float(scalar_transport.last_true_residual),
            dye_true_residual=float(dye_transport.last_true_residual))
        startup_bounds_report["steps"].append(startup_row)
        startup_bounds_report["window_completed"] = startup_window_just_completed
        startup_bounds_report["last_accepted_step"] = int(step)
        write_startup_bounds_report(startup_bounds_path, startup_bounds_report, pycomm, rank)
        if step % 10 == 0:
            root_print("FINE STARTUP BOUNDS: t={:.3f}; density maximum={:.9f}; "
                       "cells above 1.001={}; volume={:.3e} m^3; integrated excess={:.3e} m^3.".format(
                           t_now, raw_c_max, startup_metrics["cells_above_ordinary_limit"],
                           startup_metrics["volume_above_ordinary_limit_m3"],
                           startup_metrics["integrated_excess_above_one_m3"]))
    if startup_window_just_completed:
        root_print("FINE STARTUP BOUNDS CHECK PASSED at t=0.100 s; standard 0.001 limits are active.")

    # All ranks must take the same final-output/checkpoint branch.
    collective_signal = pycomm.allreduce(stop_signal, op=PY_MPI.MAX)
    if rank == 0:
        stop_reason = None
        if collective_signal:
            stop_reason = "signal {}".format(collective_signal)
        elif scheduler_end_time > 0 and scheduler_end_time - time.time() <= save_margin_seconds:
            stop_reason = "scheduler end-time margin"
        elif wall_time_budget_seconds > 0 and time.monotonic() - run_wall_start >= wall_time_budget_seconds:
            stop_reason = "soft wall budget"
    stop_reason = pycomm.bcast(stop_reason, root=0)
    stop_for_wall_budget = stop_reason is not None

    
    
    

    if step <= start_step + 5 or step % diagnostic_every == 0 or startup_window_just_completed:
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
        root_print("scalar budget relative error =", budget_relative_error)
        root_print("dye range =", raw_dye_min, raw_dye_max)
        root_print("dye equivalent source volume =", dye_volume)
        root_print("dye budget relative error =", dye_budget_relative_error)
        root_print("mixed scalar iterations density/dye =", scalar_iterations, dye_iterations)
        root_print("mixed scalar true residuals =", scalar_transport.last_true_residual, dye_transport.last_true_residual)
        root_print("gross boundary inflow/outflow/wall leak =", gross_in, gross_out, wall_leak)
        if step - start_step >= 50:
            seconds_per_step = (time.time() - wall_start) / (step - start_step)
            root_print("recent job mean wall seconds/step =", seconds_per_step)
            root_print("estimated remaining compute hours =", seconds_per_step * (max_global_step-step)/3600.0)
        write_run_status(step, t_now)

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

        diagnostics_write_error = None
        if rank == 0:
            try:
                with open(DIAGNOSTICS_PATH, "a") as f:
                    row = [
                        step, t_now, ramp_factor, tilt_x, tilt_y, velocity_component_linf,
                        div_hdiv_l2, div_th_l2, q_nozzle_hd, q_return_hd, q_nozzle_th, q_return_th,
                        raw_c_min, raw_c_max, fresh_volume, expected_fresh_physical, fresh_ratio,
                        fresh_x_centroid, fresh_y_centroid, fresh_z_centroid, lateral_ke,
                        raw_dye_min, raw_dye_max, dye_volume, expected_dye_physical,
                        dye_budget_relative_error, budget_relative_error,
                        scalar_transport.last_true_residual, dye_transport.last_true_residual,
                        gross_in, gross_out, wall_leak, fresh_diffusive_gain, dye_diffusive_gain,
                    ]
                    if len(row) != len(DIAGNOSTIC_COLUMNS) or not all(math.isfinite(v) for v in row):
                        raise RuntimeError("Invalid diagnostics row.")
                    f.write(str(int(step)) + "," + ",".join("{:.12g}".format(v) for v in row[1:]) + "\n")
                    f.flush()
            except Exception as exc:
                diagnostics_write_error = "Diagnostics append failed: {}".format(exc)
        diagnostics_write_error = pycomm.bcast(diagnostics_write_error, root=0)
        if diagnostics_write_error is not None:
            raise RuntimeError(diagnostics_write_error)

    
    # PARAVIEW OUTPUT
    

    if step % write_every == 0 or step == end_step or stop_for_wall_budget:
        write_output(
            t_now, u_sol, p_sol,
            include_velocity=(step % velocity_write_every == 0 or step == end_step or stop_for_wall_budget)
        )

    
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
        or startup_window_just_completed
        or stop_for_wall_budget
        or (pin_checkpoint_time > 0 and abs(t_now-pin_checkpoint_time) <= 5e-12)
    ):
        save_checkpoint(step, t_now, expected_fresh_physical)
        last_saved_step = step
        last_checkpoint_wall = time.time()

    if stop_for_wall_budget:
        root_print("Accepted state saved; stopping for", stop_reason, ". Re-submit to resume.")
        break






for output_file in (u_file, p_file, c_file, dye_file):
    output_file.close()

root_print("")
root_print("================================================")
root_print("30 CM RECTANGULAR THESIS-RUN SEGMENT COMPLETE")
root_print("================================================")
root_print("Segment start time =", start_time, "s")
root_print("Segment final time =", completed_step * dt, "s")
write_run_status(completed_step, completed_step * dt, complete=completed_step >= max_global_step, stop_reason=stop_reason)
root_print("Global target time =", T_global, "s")
root_print("Time-loop wall time =", time.time() - wall_start, "s")
root_print("Total wall time     =", time.monotonic() - run_wall_start, "s")
root_print("Latest checkpoint step =", last_saved_step)

if completed_step < max_global_step:
    root_print("")
    root_print("Re-submit the SAME Slurm script to continue automatically.")
else:
    root_print("")
    root_print("GLOBAL TARGET REACHED.")

MPI.barrier(comm)
if rank == 0 and lock_handle is not None:
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    lock_handle.close()

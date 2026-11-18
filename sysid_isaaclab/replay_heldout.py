"""Replay recorded trajectories in Isaac Lab with FIXED identified parameters (no optimization).

Use it to validate a fit on recordings that were not used by ``my_fit.py`` (e.g. the held-out
``datasets/b2/sideways_14.pt``) and compare the error with the training recordings and with a nominal model.

Every (parameter set, dataset) pair gets its own environment. The simulated joint positions are compared with
the recorded ones exactly as the fit does (error = sim - real - bias), so ``score`` is comparable with the
``Min score`` printed during the fit.

Example (needs the pixi ``isaaclab`` environment; ``--repo`` is the checkout holding the robot task, config.py
and the datasets, by default the repository this file lives in):

    pixi run -e isaaclab python sysid_isaaclab/replay_heldout.py \
        --params sysid_isaaclab/pace-sim2real/logs/pace/b2_sim/<run>/mean_143.pt \
        --dataset datasets/b2/sideways_14.pt datasets/b2/sideways_1.pt --baseline
"""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Replay recordings with fixed identified parameters.")
parser.add_argument("--params", type=str, required=True, help="mean_*.pt file written by my_fit.py (49 sim params).")
parser.add_argument(
    "--dataset",
    nargs="+",
    default=["datasets/b2/sideways_14.pt"],
    metavar="PATH",
    help="Recordings (.pt) to replay, relative to --repo unless absolute. Default: the held-out sideways_14.",
)
parser.add_argument("--task", type=str, default="IsaacLab-Pace-b2", help="Name of the registered task.")
parser.add_argument(
    "--repo",
    type=str,
    default=str(Path(__file__).resolve().parents[1]),
    help="Checkout holding config.py, sysid_isaaclab/tasks and the datasets.",
)
parser.add_argument(
    "--baseline",
    action="store_true",
    help="Also replay a nominal model (armature 0.1, viscous damping 1, friction 0, no bias, no delay).",
)
parser.add_argument("--out", type=str, default=None, help="JSON file for the results (default: next to --params).")
add_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()
preset_overrides = [arg for arg in unknown_args if not arg.startswith("-")]
if len(preset_overrides) != len(unknown_args):
    parser.error(f"unrecognized arguments: {' '.join(arg for arg in unknown_args if arg.startswith('-'))}")

repo = Path(args_cli.repo).resolve()
for path in (repo / "sysid_isaaclab", repo):
    sys.path.insert(0, str(path))

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import pace_sim2real.tasks  # noqa: E402,F401
from tasks import register_my_tasks  # noqa: E402,F401

NUM_JOINTS = 12
ARMATURE, DAMPING, FRICTION, BIAS = (slice(i * NUM_JOINTS, (i + 1) * NUM_JOINTS) for i in range(4))
DELAY = 4 * NUM_JOINTS


def load_dataset(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def nominal_params():
    params = torch.zeros(4 * NUM_JOINTS + 1)
    params[ARMATURE] = 0.1  # MJCF default armature
    params[DAMPING] = 1.0  # MJCF default joint damping
    return params


def apply_params(articulation, joint_ids, sim_params, initial_position):
    """Same writes as the fit (BackendAgnosticSimulatorMixin.update_simulator) for fixed params."""
    env_ids = torch.arange(sim_params.shape[0], device=joint_ids.device)
    articulation.write_joint_armature_to_sim_index(armature=sim_params[:, ARMATURE], joint_ids=joint_ids, env_ids=env_ids)
    kwargs = dict(joint_friction_coeff=sim_params[:, FRICTION], joint_viscous_friction_coeff=sim_params[:, DAMPING])
    if hasattr(articulation, "write_joint_dynamic_friction_coefficient_to_sim_index"):
        kwargs["joint_dynamic_friction_coeff"] = sim_params[:, FRICTION]  # separate_dynamic_friction is False
    articulation.write_joint_friction_coefficient_to_sim_index(joint_ids=joint_ids, env_ids=env_ids, **kwargs)
    articulation.write_joint_position_to_sim_index(position=initial_position + sim_params[:, BIAS], joint_ids=joint_ids)
    articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(initial_position), joint_ids=joint_ids)
    for drive_type in articulation.actuators.keys():
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            drive_indices = torch.arange(joint_ids.shape[0], device=joint_ids.device)[drive_indices]
        match = joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0)
        drive_joint_idx = torch.argmax(match.int(), dim=0)
        articulation.actuators[drive_type].update_encoder_bias(sim_params[:, BIAS][:, drive_joint_idx])
        articulation.actuators[drive_type].update_time_lags(sim_params[:, DELAY].to(torch.int))
        articulation.actuators[drive_type].reset(env_ids)


def main(env_cfg, data_files, param_sets, set_names):
    device = env_cfg.sim.device
    datasets = [load_dataset(f) for f in data_files]
    K, S = len(datasets), len(param_sets)
    for gain in ("kp", "kd"):
        values = [torch.as_tensor(d[gain]) for d in datasets if gain in d]
        if any(not torch.equal(v, values[0]) for v in values[1:]):
            raise ValueError(f"The recordings were made with different `{gain}`; the task has one value.")
    num_steps = max(d["time"].shape[0] for d in datasets)

    def padded(key):
        out = []
        for d in datasets:
            v = torch.as_tensor(d[key], dtype=torch.float32)
            out.append(torch.cat([v, v[-1:].expand(num_steps - v.shape[0], -1)]))
        return torch.stack(out).to(device)

    target, measured = padded("des_dof_pos"), padded("dof_pos")
    recorded = torch.stack([torch.arange(num_steps) < d["time"].shape[0] for d in datasets]).to(device)

    env_cfg.scene.num_envs = S * K
    env = gym.make(args_cli.task, cfg=env_cfg)
    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order
    joint_ids = torch.tensor([articulation.joint_names.index(n) for n in joint_order], device=env.unwrapped.device)

    # env = s * K + k replays dataset k with parameter set s
    sim_params = torch.stack(param_sets).to(device).repeat_interleave(K, dim=0)
    target_env = target.repeat(S, 1, 1)
    measured_env = measured.repeat(S, 1, 1)
    recorded_env = recorded.repeat(S, 1)

    env.reset()
    apply_params(articulation, joint_ids, sim_params, measured_env[:, 0])

    sim_trace = torch.zeros_like(measured_env)
    with torch.inference_mode():
        for t in range(num_steps):
            sim_trace[:, t] = articulation.data.joint_pos.torch[:, joint_ids]
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, joint_ids] = target_env[:, t]
            env.step(actions)
    env.close()

    err = sim_trace - measured_env - sim_params[:, BIAS].unsqueeze(1)  # (envs, T, J)
    err = err * recorded_env.unsqueeze(-1)
    steps = recorded_env.sum(dim=1).clamp_min(1)
    score = err.square().sum(dim=(1, 2)) / steps  # the fit score: mean over steps of the sum over joints
    rms = (err.square().sum(dim=(1, 2)) / (steps * NUM_JOINTS)).sqrt()
    rms_joint = (err.square().sum(dim=1) / steps.unsqueeze(-1)).sqrt()
    peak = err.abs().amax(dim=(1, 2))

    results = {}
    print("\nset        dataset                 score        rms[rad]   max|err|[rad]")
    for s in range(S):
        for k in range(K):
            i = s * K + k
            name = Path(data_files[k]).stem
            print(f"{set_names[s]:10s} {name:22s} {score[i].item():.6f}   {rms[i].item():.6f}   {peak[i].item():.4f}")
            results.setdefault(set_names[s], {})[name] = {
                "score": score[i].item(),
                "rms_rad": rms[i].item(),
                "max_abs_err_rad": peak[i].item(),
                "rms_per_joint_rad": dict(zip(joint_order, rms_joint[i].tolist())),
            }
    if S == 2:
        print("\nfitted / baseline rms ratio:")
        for k in range(K):
            print(f"  {Path(data_files[k]).stem:22s} {rms[k].item() / rms[K + k].item():.3f}")
    return results


if __name__ == "__main__":
    params_path = Path(args_cli.params).resolve()
    fitted = torch.load(params_path, map_location="cpu", weights_only=False).float().flatten()
    if fitted.numel() != 4 * NUM_JOINTS + 1:
        raise ValueError(f"{params_path} has {fitted.numel()} values, expected {4 * NUM_JOINTS + 1}.")
    param_sets, set_names = [fitted], ["fitted"]
    if args_cli.baseline:
        param_sets.append(nominal_params())
        set_names.append("baseline")
    print(f"[INFO]: effective delay of the fitted set: {int(fitted[DELAY].item())} steps (raw {fitted[DELAY].item():.3f})")

    data_files = [Path(p) if Path(p).is_absolute() else repo / p for p in args_cli.dataset]
    missing = [str(p) for p in data_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Recordings not found: {missing}")

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=len(param_sets) * len(data_files), overrides=preset_overrides
    )
    with launch_simulation(env_cfg, args_cli):
        results = main(env_cfg, data_files, param_sets, set_names)

    out = Path(args_cli.out) if args_cli.out else params_path.parent / f"replay_{params_path.stem}.json"
    out.write_text(json.dumps({"params": str(params_path), "datasets": [str(p) for p in data_files], **results}, indent=1))
    print(f"[INFO]: results written to {out}")

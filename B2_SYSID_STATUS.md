# B2 SysID: Current Status

Updated 2026-11-18

## Current setup

- Project: `/home/gdessy-iit.local/mujoco_dev/sim2real-robot-identification`.
- B2 is reached over the wired interface at `192.168.123.161` from host interface `enp98s0` (`192.168.123.99`). The neighbor entry was reachable and ping stopped when the cable was unplugged.
- ROS 2 discovery worked after selecting Cyclone DDS and the wired interface. The robot publishes Unitree `LowState` on `/lowstate`; ROS graph inspection found no `/blind_state`.
- The host Conda environment is `aliengo_pacc_deploy_ros2_env`. Its Python is 3.11, whereas the system ROS Jazzy install and generated Unitree message type support are Python 3.12 builds. For the working low-level SDK2 state probe, use the SDK2 environment/setup documented in the repo. A ROS `topic echo` attempt failed due to that Python/type-support mismatch; this does not mean the robot state stream is absent.
- The first collector state timeout was caused by SDK2 selecting loopback (`lo`) instead of `enp98s0`. A read-only probe on the wired interface later received 20 states in 0.038 s, about 502 Hz. The measured SDK motor order is FR, FL, RR, RL, each hip/thigh/calf.

## Rig and simulation

- Physical rig: a main box approximately 77 cm long, 47 cm high, and 35 cm wide; a smaller 30 x 20 x 15 cm block supports the belly near its center. Reference photos are in `b2_setup/`.
- MuJoCo rig scene models those boxes, with the B2 base origin at 0.695 m. The torso rests on the small support. Headless simulation and contact checks work; the viewer has crashed in the current Wayland/GLFW environment, so use headless runs for this screen.
- The current candidate is the slow sideways profile. In `b2_collect_sdk2.py`, each leg moves from hip abduction magnitude 0.45 rad toward 0.65 rad, with thigh/calf endpoint 0.9/-2.0 rad. The planned lowest foot is 0.277 m below the base origin. The speed setting used for the candidate is 0.3; the collector ramps from REST B2 to its start pose at no more than 0.10 rad/s, then requires a separate ARM B2 confirmation before trajectory execution.
- Headless MuJoCo screened seeds 1–20 against the modeled boxes: all were clear. Largest simulated tracking error was 0.0566 rad; largest joint speed was 0.9694 rad/s. Seed 12 measured 0.5551 rad/s, so it is a middle-speed example, not the fastest. Seed 14 was fastest at 0.9694 rad/s. These are simulation results, not a hardware safety guarantee.
- The seed 12 full simulation trajectory completed for 24 s / 4800 samples without box contact. Headless simulation is the supported check in this environment.

## Real data and fitting

- One real sideways collection, seed 12, was completed and saved as `datasets/b2/sideways_1.npz`; it was converted to `datasets/b2/sideways_1.pt`.
- A quick one-iteration MuJoCo fit on that real recording completed. Objective decreased from 0.01321 to 0.004753. Report: `/tmp/b2_seed12_fit/report.html`.
- This is an initial fit check, not a converged identification or a held-out validation. Next data step is to collect several additional collision-screened seeds (the working plan is 15 total, with seeds checked in simulation first), then fit the combined recordings and reserve at least one trajectory for validation. Isaac B2 fitting is deferred until the datasets are ready and a B2 Isaac task is available.
- The real low-level command run, final tracking behavior, and an independent emergency-stop path have not been validated. Enter/Ctrl+C stops this program’s command publishing; do not assume that this is a verified e-stop or that it forces damping.

## Latest MuJoCo fit

- There are 21 valid B2 sideways recordings, 24 s each, with unique seeds 0–21 except seed 3. All use `kp=100`, `kd=5`, speed scale 0.3. `.pt` conversions are available beside the `.npz` files.
- A held-out check trained on 20 seeds and reserved the fastest recording, seed 14. Normalized residual RMS changed from 0.0004731 to 0.0001894 on seed 14 (about 60% lower), comparable to the training set. This supports generalization to that trajectory.
- The final all-data MuJoCo fit then used all 21 available seeds, including seed 14. Its objective decreased from 0.2862 to 0.0440; the objective was essentially flat by iteration 8. The fitted model was exported and reloaded successfully.
- Fitted fixed-base MJCF: `sysid_mujoco/results/b2/final_all21/b2_fitted.xml`. Named fit parameters and fit metadata: `sysid_mujoco/results/b2/final_all21/metrics.json`. The held-out metrics and its 20-seed parameter vector are in `sysid_mujoco/results/b2/heldout_seed14/metrics.json`.
- This fit identifies per-joint armature, friction loss, and damping over the nominal model. It does not identify link inertias. Isaac B2 fitting is the next backend step; no Isaac B2 task has been implemented yet.

## Clearance and operating reminders

- `--rest-foot-clearance` means measured vertical clearance from the floor to the lowest foot in the robot’s initial/rest configuration. It is not the height of the belly support. Measure it for the actual rig before using that gate; 0.47 m is only a rough estimate if the feet are on the main box top.
- In earlier hardware attempts, the robot was on a 40 cm box and a state timeout occurred while DDS selected `lo`; state reception later succeeded with the correct wired interface. Do not interpret that timeout as a trajectory or robot-motion failure.
- User operates the robot commands. Never launch collector, trajectory, or other robot-control code when connected to the robot. Read-only status inspection is the safe boundary for assistant actions during a connected session.

## Relevant files

- Collector and safety prompts: `b2_collect_sdk2.py`
- Box-model headless simulator: `b2_fixed_base_sim.py`
- Existing broader notes and historical setup: `B2_SYSID_NOTES.md`

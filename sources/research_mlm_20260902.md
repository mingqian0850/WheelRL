# Source record: MLM whole-body loco-manipulation

Retrieved: 2026-09-02

Query: assess whether *MLM: Learning Multi-task Loco-Manipulation Whole-Body
Control for Quadruped Robot with Arm* can serve as the training blueprint for
B2-W + Z1 six-DoF TCP tracking.

## Primary sources

- arXiv: <https://arxiv.org/abs/2508.10538>
- IEEE DOI: <https://doi.org/10.1109/LRA.2025.3632087>
- Author publication page: <https://yding25.com/>
- User-provided PDF inspected locally on 2026-09-02:
  `/mnt/c/Users/chenm/Zotero/storage/CS6S64NI/Liu et al. - 2025 - MLM Learning Multi-task Loco-Manipulation Whole-Body Control for Quadruped Robot with Arm.pdf`

## Verified facts

- MLM trains one PPO policy for multi-task whole-body loco-manipulation on a
  Go2 carrying a six-DoF Airbot Play arm.
- The actor consumes five frames of quadruped proprioception, arm position and
  velocity, the previous 18-dimensional action, and a short TCP pose reference
  spanning past/current/future samples. A pose is represented by translation
  and the first six elements of a rotation matrix rather than Euler-angle
  subtraction.
- The policy outputs 12 leg and 6 arm joint-position offsets, which are tracked
  by PD control. The critic additionally receives simulation-only privileged
  information.
- The Trajectory-Velocity Prediction network predicts unavailable future TCP
  references and base velocity during historical-only teleoperation. If an
  upstream diffusion policy already provides a future trajectory, the future
  predictor is bypassed.
- Training uses NVIDIA Isaac Gym with 4,096 agents. The paper reports an RTX
  3090 Ti and about eight hours for 10,000 iterations. The whole-body policy is
  deployed at 50 Hz on a Jetson Orin NX.
- Reported simulation mean position errors are approximately 0.66--1.32 cm and
  orientation errors approximately 0.05--0.15 rad across tasks. Selected
  real-world tasks report approximately 1.02--1.39 cm and 0.036--0.054 rad.
  These results do not include perception/calibration error for a B2-W + Z1
  door system.
- The demonstrations include everyday manipulation trajectories, but not the
  complete contact mechanics of rotating a lever, releasing a latch, and
  pulling a hinged door while regulating force.

## Code and data availability audit

No official project page, implementation, checkpoint, Isaac Gym environment,
or the paper's task-trajectory library was found from the arXiv record, DOI,
author publication page, author public GitHub, exact-title searches, arXiv-ID
searches, or DOI searches. This is a dated negative search result, not proof
that the authors cannot release code later.

## Relevance assessment

MLM is the stronger blueprint for continuous six-DoF TCP trajectory tracking:
use its reference window, state history, asymmetric training, curriculum, and
domain randomization. It is not a drop-in implementation for B2-W because it
has no wheels, no released code, no hard whole-body constraints, and no
contact-force door controller. For a planned door trajectory, future reference
samples are already known, so the trajectory predictor is optional initially.

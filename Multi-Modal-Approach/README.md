# Multi-Modal-Approach

Local application skeleton separating visual detection, pizza geometry, recipe
planning, camera/workplane registration, and robot manipulation. Detector training
is intentionally a separate project; no real-food detection model is bundled.

## Current status

Implemented: typed component contracts, strict structured-recipe parsing,
dependency-injected dry-run wiring with inventory/count/camera-ID checks, and
hardware-disabled executor. No images are automatically detected, no geometric
planner or camera transform is implemented, and no robot can be moved by this
skeleton. Adapters must not substitute fake detections or identity transforms.

From this directory, using Python 3.12+ (no extra dependencies):

```bash
python3 app.py --recipe recipe.example.json
python3 -m unittest -v test_app
```

The CLI validates the recipe and explicitly reports missing adapters. It does
not claim to have generated a layout. Labels are extensible: olives, pepperoni,
cheese, mushrooms, etc. Natural-language requests such as "three olives and eight
pepperoni slices" will later be translated into this structured format and
confirmed before planning. Cheese distribution by area/weight requires a future
recipe schema; `count` means discrete instances only.

## Components and wiring

| Component | Responsibility / future adapter |
| --- | --- |
| `pizza_detection.py` | Trained segmentation/detection: pizza center, boundary, confidence |
| `ingredient_detection.py` | Real-food labels, locations, confidence; verification using a new frame |
| `layout_planner.py` | Geometry-only target generation; counts, topping footprints, physical margins and spacing |
| `coordinate_calibration.py` | Versioned camera-to-workplane registration, camera ID and robot serial, explicit millimeters |
| `recipe_parser.py` | Strict structured recipe boundary; later add confirmed natural-language interpretation |
| `robot_executor.py` | Future deterministic or hybrid manipulation adapter; all hardware execution currently refused |
| `app.py` | Compose injected adapters into a dry-run plan; never invoke the executor |
| `models.py` | Shared pixel/robot coordinate types and observation/plan metadata |

Flow:

```text
recipe -> structured quantities
workspace frame -> pizza + ingredient detectors
camera registration + pizza geometry + recipe -> layout planner
layout + detected sources -> manipulation backend (NOT IMPLEMENTED)
new workspace frame -> placement verification -> accept/retry/stop
```

The composition function is integration wiring, not a safety validator. Production
adapters must validate confidence, image freshness, camera identity/pose/resolution,
boundary geometry and ingredient footprints, feasible layouts, reachable poses,
calibration provenance, source uniqueness, and collision-free approach/lift paths.
The count checks alone do not establish correct or safe placement.

## Station capability finding

The installed NormaCore Station Python client and examples expose serial-targeted
ST3215 register writes/sync-writes and telemetry. No Cartesian XYZ/IK command was
found in those surfaces. `send_commands` is a generic transport, not proof of a
Cartesian controller. Consequently:

- **Option A: deterministic control** needs a validated robot kinematic model,
  joint directions/offsets/limits, gripper/tool geometry, inverse kinematics,
  collision checking, and supervised pick/lift/place primitives. Alternatively,
  separately validated taught joint-space primitives may serve fixed stations;
  they are not arbitrary Cartesian control.
- **Option B: hybrid SmolVLA** needs demonstrations on the actual calibrated
  follower and an explicit way to condition manipulation on the planned target
  (e.g. a verified target-relative observation). A task-string-only checkpoint
  must not be assumed to follow arbitrary geometric XY targets.

Both options consume the same layout contract and require camera-based placement
verification. Neither adapter is selected or implemented by default.

## Calibration and development sequence

1. Use [guided joint calibration](../elrobot/docs/calibration.md) and save both
   serial-bound profiles. Preserve Station metadata and motor EEPROM.
2. Fix the workspace camera pose. Collect measured pixel/robot correspondences
   on the pizza workplane and validate error on held-out correspondences. A planar
   homography does not supply ingredient height or a collision-free 3D path.
3. Integrate separately trained detection adapters and test on recorded images.
4. Implement/test geometry and constraints with no hardware attached.
5. Choose deterministic or target-conditioned hybrid manipulation after verifying
   robot kinematics and API integration. Keep dry-run default and guarded execution.
6. Validate each primitive under supervision, then add closed-loop verification.

Local images, registration files, and generated plans belong in ignored `local/`.
Never commit credentials, robot-local configuration, or large collected datasets
by default.

# 🔜 Roadmap

← back to the [README](../README.md)

## More Evaluation Protocols

Place recognition is evaluated under several incompatible protocols. `inlier eval` implements two of them today — [cross-session](helipr-benchmark.md#running-the-evaluation) and [online-lcd](cli.md#online-loop-closure-detection). Coming next:

- 📍 **`online-global`** — online localization against a fixed prior map, deciding at a *fixed* threshold with no post-hoc selection, and reporting first-fix latency and per-frame cost.
- 🗺️ **`multi-session`** — N sessions all-vs-all, aggregated into one benchmark table.
- ▶️ **`inlier run`** — produce loop closures and 6-DoF poses on data with **no ground truth**, which is what a deployment actually has.

## ROS2 Support

- 🤖 We are also planning to release ROS2 nodes to support front-end agnostic loop closures, including a GTSAM based back-end optimization.

## Front-End Integrations

- 🧩 We also intend to provide integrations with **[KISS-ICP](https://github.com/PRBonn/kiss-icp)**, **[FAST-LIO2](https://github.com/hku-mars/FAST_LIO)** and **[GLIM](https://github.com/koide3/glim)**, so that InLiER can plug into the odometry front-end you already run — taking its deskewed scans and pose estimates, and returning loop closures with 6-DoF constraints for the pose graph.

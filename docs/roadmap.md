# 🔜 Roadmap

← back to the [README](../README.md)

## More Evaluation Protocols

Place recognition is evaluated under several incompatible protocols, and `inlier eval` currently implements one of them ([cross-session](helipr-benchmark.md#running-the-evaluation)). Coming next:

- 🔁 **`online-lcd`** — single-session online loop closure detection. The database grows as the query streams and candidates are restricted to frames older than an exclusion window, reported with the SLAM convention (F1max, max-recall at 100% precision) rather than Recall@N.
- 📍 **`online-global`** — online localization against a fixed prior map, deciding at a *fixed* threshold with no post-hoc selection, and reporting first-fix latency and per-frame cost.
- 🗺️ **`multi-session`** — N sessions all-vs-all, aggregated into one benchmark table.
- ▶️ **`inlier run`** — produce loop closures and 6-DoF poses on data with **no ground truth**, which is what a deployment actually has.

## ROS2 Support

- 🤖 We are also planning to release ROS2 nodes to support front-end agnostic loop closures, including a GTSAM based back-end optimization.

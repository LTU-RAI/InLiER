# 🔜 Roadmap

← back to the [README](../README.md)

## More Evaluation Protocols

Place recognition is evaluated under several incompatible protocols. `inlier eval` implements two of them today — [cross-session](helipr-benchmark.md#running-the-evaluation) and [online-lcd](cli.md#online-loop-closure-detection). Coming next:

- ▶️ **`inlier run`** — produce loop closures and 6-DoF poses on data with **no ground truth**, which is what a deployment actually has.

### Two protocols we decided not to add

Earlier drafts of this page listed two more. Both are gone, for opposite
reasons — one was already implemented under another name, the other would have
produced a number we do not believe in.

#### `online-global` — already here

Online global localization against a fixed prior map, deciding at a fixed
threshold. It *is* [cross-session](helipr-benchmark.md#running-the-evaluation)
run with `--threshold`.

A prior map is a database built from one session and finalized. Because it never
changes while the queries run, arrival order cannot affect what is retrievable,
so every query still sees the whole database — which is exactly what
cross-session does. Same database, same candidate set, same scoring; the
retrieval is identical, down to the candidate lists. The only thing a separate
protocol would have added is the promise not to pick the operating threshold
from the run being scored, and `--threshold X` already makes that promise
(`--threshold-policy fixed`).

So: **to evaluate global localization against a prior map, run `inlier eval
cross-session` with a threshold you fixed beforehand** — chosen on a different
sequence, not this one. The two metrics that protocol would have added
uniquely, first-fix latency and a per-frame cost distribution, are tracked as
possible additions to cross-session's output rather than as a reason to
duplicate it.

#### `multi-session` — the table would not be the evidence

N sessions all-vs-all, aggregated into one benchmark table. The mechanics are
easy — it is N(N−1) cross-session runs, which you can already produce today by
looping `inlier eval cross-session` over the pairs and collecting the JSONs.
That is exactly the problem: a table of pairwise retrieval scores is not what
anyone actually wants to know about a multi-session system.

The real question is whether the map stays consistent as sessions accumulate,
and that does not decompose into pairs. It depends on the order the sessions
arrive in, on what the back-end does with the constraints InLiER hands it, on
how the map is merged and pruned between runs, and on drift that only shows up
several sessions later. Averaging pairwise F1 over a grid would look like a
measurement of that and would not be one.

So it is left to field evaluation, where those variables are real rather than
assumed. If you want the pairwise grid anyway, the loop above gives it to you
without a protocol that implies more than it measures.

## ROS2 Support

- 🤖 We are also planning to release ROS2 nodes to support front-end agnostic loop closures, including a GTSAM based back-end optimization.

## Front-End Integrations

- 🧩 We also intend to provide integrations with **[KISS-ICP](https://github.com/PRBonn/kiss-icp)**, **[FAST-LIO2](https://github.com/hku-mars/FAST_LIO)** and **[GLIM](https://github.com/koide3/glim)**, so that InLiER can plug into the odometry front-end you already run — taking its deskewed scans and pose estimates, and returning loop closures with 6-DoF constraints for the pose graph.

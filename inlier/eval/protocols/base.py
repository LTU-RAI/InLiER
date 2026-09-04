"""What every protocol produces.

A protocol decides *which* database entries a query may match and *what counts
as correct*; the stages, the metrics, and the artifacts are shared.  Keeping
the result shape common is what lets ``inlier report`` and ``inlier compare``
treat a cross-session run and an online loop-closure run the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RunResult:
    """One completed run -- an evaluation, or an ``inlier run`` deployment.

    Every metric ``summary()`` reports is optional, because ``inlier run``
    produces none: it has no labels to score against.
    """

    protocol: str
    results: Dict[str, Any]                 # the results JSON payload
    output_dir: Path
    artifacts: Dict[str, Path] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"protocol : {self.protocol}", f"output   : {self.output_dir}"]
        conf = self.results.get("confusion") or {}
        if conf:
            lines.append(
                f"{conf.get('stage', '?')} @ thr={conf.get('threshold')}: "
                f"TP={conf.get('TP')} FP={conf.get('FP')} "
                f"FN={conf.get('FN')} TN={conf.get('TN')}  "
                f"P={conf.get('precision')} R={conf.get('recall')}"
            )
        lc = self.results.get("loop_closure") or {}
        if lc:
            lines.append(
                f"loop closure: R@1={lc.get('recall_at_1')}  "
                f"F1max={lc.get('f1_max')}  "
                f"max recall @ 100% precision="
                f"{lc.get('max_recall_at_full_precision')}")
        cl = self.results.get("closures") or {}
        if cl:
            lines.append(
                f"closures : {cl.get('n_closures')} at threshold "
                f"{(self.results.get('config') or {}).get('threshold')} "
                f"across {cl.get('n_queries_with_a_closure')} frame(s), "
                f"{cl.get('n_refined')} GICP-refined")
        for stage in ("stage1", "stage2", "combined", "verify"):
            block = self.results.get(stage)
            if block:
                r1 = block["recall_at_n"].get("1")
                lines.append(f"{stage:<9}: R@1={r1}  PR-AUC={block['pr_auc']}")
        return "\n".join(lines)

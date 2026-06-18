#!/usr/bin/env python3
"""
评估脚本：跑 30 条 test_samples.json，对比旧版/新版 prompt 的准确率。

用法：
    python run_eval.py                      # 默认 mock 模式跑两版对比
    python run_eval.py --real               # 用真实 OpenAI API
    python run_eval.py --output results/    # 指定输出目录
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "improved"))
sys.path.insert(0, str(ROOT / "eval"))

from classifier import (  # type: ignore  noqa: E402
    Config, batch_classify, make_openai_caller, with_retry,
)
from mock_llm import make_mock_caller  # type: ignore  noqa: E402

LABELS = ["退款退货", "物流查询", "账号问题", "商品咨询", "投诉建议", "其他"]


def evaluate(results_path: Path, samples_path: Path) -> dict:
    truth = {s["id"]: s["label"] for s in json.loads(samples_path.read_text(encoding="utf-8"))}
    preds = json.loads(results_path.read_text(encoding="utf-8"))

    correct = 0
    errors: list[dict] = []
    confusion: dict[str, Counter] = defaultdict(Counter)  # true_label -> Counter(pred_label)
    per_class_total: Counter = Counter()
    per_class_correct: Counter = Counter()

    for p in preds:
        qid = p["id"]
        true_lbl = truth.get(qid)
        pred_lbl = p["predicted_category"]
        per_class_total[true_lbl] += 1
        confusion[true_lbl][pred_lbl] += 1
        if pred_lbl == true_lbl:
            correct += 1
            per_class_correct[true_lbl] += 1
        else:
            errors.append({
                "id": qid,
                "question": p["question"],
                "true": true_lbl,
                "pred": pred_lbl,
                "confidence": p.get("confidence", ""),
                "reason": p.get("reason", ""),
            })

    return {
        "total": len(preds),
        "correct": correct,
        "accuracy": correct / len(preds) if preds else 0.0,
        "errors": errors,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "per_class": {
            lbl: {
                "total": per_class_total[lbl],
                "correct": per_class_correct[lbl],
                "recall": per_class_correct[lbl] / per_class_total[lbl] if per_class_total[lbl] else 0.0,
            }
            for lbl in LABELS
        },
    }


def render_report(eval_v1: dict, eval_v2: dict) -> str:
    lines: list[str] = []
    lines.append("# 评估报告：旧版 vs 新版 Prompt\n")
    lines.append("## 总体准确率\n")
    lines.append("| 版本 | 正确 | 总数 | 准确率 |")
    lines.append("|------|------|------|--------|")
    lines.append(f"| v1（旧版） | {eval_v1['correct']} | {eval_v1['total']} | "
                 f"**{eval_v1['accuracy']:.1%}** |")
    lines.append(f"| v2（新版） | {eval_v2['correct']} | {eval_v2['total']} | "
                 f"**{eval_v2['accuracy']:.1%}** |")
    delta = (eval_v2['accuracy'] - eval_v1['accuracy']) * 100
    lines.append(f"\n**提升：+{delta:.1f} 个百分点**\n")

    lines.append("## 各类别召回率（recall = 该类样本中被正确分类的比例）\n")
    lines.append("| 类别 | v1 召回 | v2 召回 | Δ |")
    lines.append("|------|--------|--------|----|")
    for lbl in LABELS:
        r1 = eval_v1["per_class"][lbl]["recall"]
        r2 = eval_v2["per_class"][lbl]["recall"]
        total = eval_v1["per_class"][lbl]["total"]
        lines.append(f"| {lbl} | {r1:.0%} ({eval_v1['per_class'][lbl]['correct']}/{total}) "
                     f"| {r2:.0%} ({eval_v2['per_class'][lbl]['correct']}/{total}) "
                     f"| {(r2 - r1) * 100:+.0f}pp |")

    lines.append("\n## v1 错误样本（这些都是新版修复的）\n")
    for e in eval_v1["errors"]:
        fixed = "🟢 新版修复" if all(
            ne["id"] != e["id"] for ne in eval_v2["errors"]
        ) else "🔴 新版仍错"
        lines.append(f"- **id={e['id']}** `{e['question']}` → "
                     f"真:{e['true']} / v1:{e['pred']}  {fixed}")

    if eval_v2["errors"]:
        lines.append("\n## v2 仍存在的错误\n")
        for e in eval_v2["errors"]:
            lines.append(f"- **id={e['id']}** `{e['question']}` → "
                         f"真:{e['true']} / v2:{e['pred']} (conf={e['confidence']}) "
                         f"— {e['reason']}")
    else:
        lines.append("\n## v2 仍存在的错误\n\n无。")

    lines.append("\n## 混淆矩阵（v2，行=真实，列=预测）\n")
    header = "| 真↓ \\ 预→ | " + " | ".join(LABELS) + " |"
    sep = "|---" * (len(LABELS) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for true_lbl in LABELS:
        row = [true_lbl]
        for pred_lbl in LABELS:
            v = eval_v2["confusion"].get(true_lbl, {}).get(pred_lbl, 0)
            row.append(str(v) if v else "·")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=str(ROOT / "original" / "test_samples.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "eval" / "results"))
    parser.add_argument("--real", action="store_true",
                        help="使用真实 OpenAI API（默认 mock）")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = Path(args.samples)

    # 跑 v1（旧 prompt）
    print("→ 跑 v1（旧 prompt）...")
    v1_llm = (
        with_retry(make_openai_caller(Config()), Config())
        if args.real else make_mock_caller("v1")
    )
    v1_out = out_dir / "v1_results.json"
    if v1_out.exists():
        v1_out.unlink()  # 评估总是从头跑
    batch_classify(args.samples, str(v1_out), v1_llm, concurrency=args.concurrency)

    # 跑 v2（新 prompt）
    print("\n→ 跑 v2（新 prompt）...")
    v2_llm = (
        with_retry(make_openai_caller(Config()), Config())
        if args.real else make_mock_caller("v2")
    )
    v2_out = out_dir / "v2_results.json"
    if v2_out.exists():
        v2_out.unlink()
    batch_classify(args.samples, str(v2_out), v2_llm, concurrency=args.concurrency)

    # 评估对比
    eval_v1 = evaluate(v1_out, samples_path)
    eval_v2 = evaluate(v2_out, samples_path)

    (out_dir / "v1_metrics.json").write_text(
        json.dumps(eval_v1, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "v2_metrics.json").write_text(
        json.dumps(eval_v2, ensure_ascii=False, indent=2), encoding="utf-8")

    report = render_report(eval_v1, eval_v2)
    (out_dir / "comparison.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(report)
    print(f"详细结果已写入: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Regenerate experiment tables from packaged numerical evidence (stdlib only)."""

from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "evidence" / name).read_text())


def table(headers, rows):
    return (
        "\n".join(
            ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
            + ["|" + "|".join(str(x) for x in row) + "|" for row in rows]
        )
        + "\n"
    )


def render():
    chunks = [
        "# 自动生成的实验数值表\n\n由 `python scripts/render_evidence.py` 从随包 JSON 重建，不能跨面板直接比较。\n"
    ]
    data = read("architecture.json")["per_source"]
    rows = []
    for mode in ("img65", "img85", "txt", "txt_joint"):
        cells = [r for r in data if r["mode"] == mode]
        if not cells:
            continue
        for arm in ("SI1", "AI0", "AI1"):
            rows.append(
                [
                    mode,
                    arm,
                    len(cells),
                    f"{sum(r[arm]['L_side'] for r in cells) / len(cells):.5f}",
                    f"{sum(r[arm]['correct_minus_all_wrong'] for r in cells) / len(cells):.5f}",
                ]
            )
    chunks += [
        "## 注入架构\n\n3内容×5画师；txt_joint仅1内容×5画师。L=相对无风格的风格余弦增益；D=正确与其他画师输出的相对差。\n",
        table(["模式", "组", "内容数", "L", "D"], rows),
    ]
    comparison = read("zeroshot_comparison.json")["comparisons"]
    rows = []
    for arm in ("Z0", "Z1", "Z2", "Z3"):
        value = read(arm + "_evaluation.json")
        cohort = next(x for x in value["summary"] if not x["adapter_seen"])
        for branch in ("single", "blankstyle"):
            result = cohort["arms"][branch]
            match = next(
                (x for x in comparison if x["stage"] == arm and x["arm"] == branch and not x["adapter_seen"]),
                None,
            )
            if match is None:
                raise ValueError(f"Missing comparison {arm}/{branch}")
            m = match["metrics"]
            ci = m["hit1"]["ci95"]
            rows.append(
                [
                    arm + "/" + branch,
                    f"{result['hit1'] * 100:.3f}%",
                    f"{m['style']['delta']:+.5f}",
                    f"{m['content']['delta']:+.5f}",
                    f"{m['hit1']['delta'] * 100:+.3f} [{ci[0] * 100:+.3f},{ci[1] * 100:+.3f}]",
                    match.get("passes", False),
                ]
            )
    chunks += [
        "## Z0–Z3 训练消融\n\n8位适配器未见画师×4案例。Δ相对AI0 single，95% CI按画师聚类。\n",
        table(["组/分支", "Hit@1", "Δ风格", "Δ内容", "Δ命中pp [95% CI]", "通过旧门槛"], rows),
    ]
    tuning = read("z3_tuning.json")["summary"]
    csv_rows = []
    for cohort in tuning:
        rows = []
        for name, item in cohort["arms"].items():
            m = item["means"]
            delta = item["versus_Z3_single"]
            ci = delta["style"]["ci95"] if delta else None
            rows.append(
                [
                    name,
                    f"{m['style']:.5f}",
                    f"{m['hit1'] * 100:.3f}%",
                    f"{m['content']:.5f}",
                    f"{delta['style']['delta']:+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}]" if delta else "基线",
                ]
            )
            csv_rows.append(dict(adapter_seen=cohort["adapter_seen"], arm=name, **m))
        chunks += [
            f"## Z3 推理参数扫描：{'已见' if cohort['adapter_seen'] else '未见'}画师\n\n各8位画师×4案例，复用原始基线；同一个已看过的开发面板。\n",
            table(["设置", "风格余弦", "Hit@1", "内容余弦", "Δ风格 [95% CI]"], rows),
        ]
    rows = []
    for item in read("eval_v2.json")["expanded_evaluations"]:
        rows.append(
            [
                item["step"],
                f"{item['csd_margin']:.6f}",
                f"{item['csd_similarity']:.6f}",
                f"{item['fm_relative']:.6f}",
                f"{item['condition_gain'] * 100:.4f}%",
                item["generation_cases"],
                item["fm_cases"],
            ]
        )
    chunks += [
        "## 后期独立评估 v2\n\n16画师、192成图案例、288固定FM案例；不与早期DINO余弦混比。\n",
        table(["step", "CSD margin", "CSD similarity", "FM / 无条件FM", "条件收益", "成图数", "FM数"], rows),
    ]
    moss = read("moss_diagnosis.json")
    rows = [
        [
            x["step"],
            f"{x['native']:.8f}",
            f"{x['correct']:.8f}",
            f"{x['wrong']:.8f}",
            f"{x['correct_improvement_percent']:.4f}%",
        ]
        for x in moss["validation"]
        if x["arm"] == "extend"
    ]
    chunks += [
        "## Moss 固定FM诊断\n\n本轮使用64个固定噪声案例；与更早256步demo不同seed的FM不可混比。\n",
        table(["step", "无style", "正确参考", "错误参考", "相对基模改善"], rows),
    ]
    rows = []
    for arm in ("step256", "step1024"):
        for mode, m in moss["images"][arm]["dino"].items():
            rows.append([arm, mode, f"{m['style_cos']:.5f}", f"{m['hits']}/{m['n']}"])
    chunks += [
        "## Moss 生成诊断\n\n4画师×2提示词×2seed；48画师原型图库。\n",
        table(["检查点", "条件", "DINO风格余弦", "Hit@1"], rows),
    ]
    return "\n".join(chunks), csv_rows


if __name__ == "__main__":
    output, rows = render()
    (ROOT / "docs/Z3_TABLES.md").write_text(output)
    with (ROOT / "evidence/z3_tuning.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

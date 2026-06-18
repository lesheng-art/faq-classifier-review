"""
Mock LLM —— 用于离线评估、CI 烟雾测试、回归测试。

基于对 30 条样本的人工 review，模拟两版 prompt 的失败模式：
  - v1（旧 prompt）：缺类别定义、缺多意图规则、缺边界 case 处理 → 容易被关键词带偏
  - v2（新 prompt）：注入定义 + few-shot + JSON 约束后绝大多数误判被修复

预测映射是确定性的，所以跑出的准确率（63.3% → 96.7%）**反映 prompt 设计的预期效果**，
而不是真实 LLM 的实际输出。如需真实模型评估，请用 `run_eval.py --real` 调用 API。
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal


PromptVersion = Literal["v1", "v2"]


# 基于人工 review 30 条样本得到的 fixture 映射
# 格式：id -> (v1_label, v2_label, v1_confidence, v2_confidence)
_PREDICTIONS: dict[int, tuple[str, str, str, str]] = {
    1:  ("退款退货", "退款退货", "high",   "high"),
    2:  ("物流查询", "物流查询", "high",   "high"),
    3:  ("账号问题", "账号问题", "high",   "high"),
    4:  ("商品咨询", "商品咨询", "high",   "high"),
    5:  ("投诉建议", "投诉建议", "high",   "high"),
    6:  ("物流查询", "退款退货", "high",   "high"),   # 旧错：'到账'触发物流
    7:  ("物流查询", "物流查询", "high",   "high"),
    8:  ("账号问题", "账号问题", "high",   "high"),
    9:  ("商品咨询", "商品咨询", "high",   "high"),
    10: ("其他",     "投诉建议", "low",    "high"),   # 旧错：'举报'歧义
    11: ("退款退货", "退款退货", "high",   "high"),
    12: ("物流查询", "物流查询", "high",   "high"),
    13: ("账号问题", "账号问题", "high",   "high"),
    14: ("商品咨询", "商品咨询", "high",   "high"),
    15: ("物流查询", "投诉建议", "medium", "high"),   # 旧错：'配送'关键词
    16: ("退款退货", "退款退货", "high",   "high"),
    17: ("物流查询", "物流查询", "high",   "high"),
    18: ("账号问题", "账号问题", "high",   "high"),
    19: ("商品咨询", "商品咨询", "high",   "high"),
    20: ("投诉建议", "投诉建议", "high",   "high"),
    21: ("物流查询", "退款退货", "medium", "medium"), # 旧错：'快递没寄回'触发物流
    22: ("物流查询", "物流查询", "high",   "high"),
    23: ("退款退货", "投诉建议", "high",   "high"),   # 旧错：'退货'关键词
    24: ("物流查询", "物流查询", "medium", "medium"), # 双错：双意图边界
    25: ("退款退货", "物流查询", "medium", "high"),   # 旧错：'寄错地址'误判换货
    26: ("退款退货", "退款退货", "high",   "high"),
    27: ("其他",     "商品咨询", "low",    "high"),   # 旧错：'带上飞机'歧义判其他
    28: ("商品咨询", "其他",     "low",    "high"),   # 旧错：闲聊无规则
    29: ("商品咨询", "其他",     "low",    "high"),   # 旧错：纯标点无规则
    30: ("商品咨询", "其他",     "low",    "high"),   # 旧错：纯问候无规则
}

_REASONS_V2: dict[int, str] = {
    6: "主诉是退款进度", 10: "举报含具体投诉", 15: "明确给平台建议",
    21: "主诉是取消退货", 23: "语气抱怨流程", 24: "双意图，主诉退款（边界）",
    25: "诉求是改派送", 27: "咨询商品使用场景", 28: "闲聊致谢",
    29: "纯标点无信息", 30: "纯问候",
}


@lru_cache(maxsize=1)
def _question_to_id() -> dict[str, int]:
    """O(1) lookup，文件只读一次（修复旧版每条样本都重读文件的 O(n²) bug）"""
    samples_path = Path(__file__).resolve().parent.parent / "original" / "test_samples.json"
    if not samples_path.exists():
        return {}
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    return {s["question"]: s["id"] for s in samples}


def make_mock_caller(prompt_version: PromptVersion = "v2"):
    """
    返回符合 classifier.LLMFn 协议的可调用对象。
    再次提醒：这只是 fixture，不是真实模型。
    """
    q2id = _question_to_id()

    def call(question: str) -> tuple[str, int]:
        time.sleep(0.005)  # 模拟微小延迟
        qid = q2id.get(question)
        if qid is None or qid not in _PREDICTIONS:
            output = json.dumps(
                {"label": "其他", "confidence": "low", "reason": "fixture 未配置该样本"},
                ensure_ascii=False,
            )
            return output, 5

        v1_lbl, v2_lbl, v1_conf, v2_conf = _PREDICTIONS[qid]
        if prompt_version == "v1":
            # mock 只模拟「分类结果差异」，不模拟「输出格式差异」——
            # 输出格式是真实 LLM 才能暴露的失败模式，mock 不该假装能测。
            output = json.dumps(
                {"label": v1_lbl, "confidence": v1_conf, "reason": ""},
                ensure_ascii=False,
            )
            return output, 5
        output = json.dumps(
            {"label": v2_lbl, "confidence": v2_conf, "reason": _REASONS_V2.get(qid, "")},
            ensure_ascii=False,
        )
        return output, 5

    return call

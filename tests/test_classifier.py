"""
单元测试 —— 覆盖最容易回归的工程路径：
  - parse_and_normalize（解析/归一化）
  - _is_retryable（retry 白名单）
  - batch_classify 的 JSONL 增量落盘 + 断点续传
  - few-shot 是否真的不在测试集（数据泄漏回归门）
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from classifier import (
    VALID_LABELS, FALLBACK_LABEL, FEW_SHOT,
    parse_and_normalize, _is_retryable,
    batch_classify, _jsonl_path,
)


# ---------- parse_and_normalize ----------

class TestParseAndNormalize:
    def test_valid_json(self):
        raw = '{"label": "退款退货", "confidence": "high", "reason": "test"}'
        label, conf, reason = parse_and_normalize(raw)
        assert label == "退款退货"
        assert conf == "high"
        assert reason == "test"

    def test_invalid_json_does_not_guess_label(self):
        """关键回归：W3 修复 —— 不再从乱文里猜标签"""
        raw = '这不是退款问题，应该归到投诉建议'
        label, conf, reason = parse_and_normalize(raw)
        assert label == FALLBACK_LABEL  # 必须兜底为"其他"
        assert conf == "low"
        assert "JSON 解析失败" in reason
        # 即便原文里出现了"退款退货""投诉建议"也不应被采纳
        assert label not in {"退款退货", "投诉建议"}

    def test_illegal_label(self):
        raw = '{"label": "退款", "confidence": "high"}'  # 缺"退货"
        label, conf, _ = parse_and_normalize(raw)
        assert label == FALLBACK_LABEL
        assert conf == "low"

    def test_empty_input(self):
        label, conf, reason = parse_and_normalize("")
        assert label == FALLBACK_LABEL
        assert "空" in reason

    def test_non_object_json(self):
        label, conf, _ = parse_and_normalize('["退款退货"]')
        assert label == FALLBACK_LABEL

    def test_invalid_confidence_normalized(self):
        raw = '{"label": "其他", "confidence": "VERY_SURE"}'
        _, conf, _ = parse_and_normalize(raw)
        assert conf == "low"


# ---------- retry 白名单 ----------

class TestIsRetryable:
    def test_value_error_not_retryable(self):
        assert _is_retryable(ValueError("bad")) is False

    def test_key_error_not_retryable(self):
        """配置错误不该重试"""
        assert _is_retryable(KeyError("OPENAI_API_KEY")) is False

    def test_timeout_retryable_by_name(self):
        """没装 openai 也能按名字识别瞬时错误"""
        class RateLimitError(Exception): pass
        assert _is_retryable(RateLimitError()) is True

        class TimeoutError(Exception): pass
        assert _is_retryable(TimeoutError()) is True


# ---------- batch_classify JSONL + 断点续传 ----------

class TestBatchClassify:
    def _make_llm(self):
        """返回一个可控的假 LLM"""
        calls: list[str] = []

        def llm(q: str) -> tuple[str, int]:
            calls.append(q)
            return f'{{"label": "其他", "confidence": "high", "reason": "{q[:10]}"}}', 1

        return llm, calls

    def test_jsonl_incremental_write(self, tmp_path: Path):
        """W2 修复回归：每条立即 append 到 .jsonl"""
        input_file = tmp_path / "in.json"
        input_file.write_text(json.dumps(
            [{"id": i, "question": f"q{i}"} for i in range(1, 6)],
            ensure_ascii=False,
        ))
        output_file = tmp_path / "out.json"

        llm, _ = self._make_llm()
        batch_classify(str(input_file), str(output_file), llm, concurrency=2)

        # .jsonl 应该有 5 行
        jsonl = _jsonl_path(output_file)
        assert jsonl.exists()
        lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
        assert len(lines) == 5
        # 每行都是合法 JSON
        for line in lines:
            json.loads(line)

        # 最终 .json 也写好了，且有序
        results = json.loads(output_file.read_text(encoding="utf-8"))
        assert [r["id"] for r in results] == [1, 2, 3, 4, 5]

    def test_resume_skips_done(self, tmp_path: Path):
        """断点续传：已 jsonl 落盘的不重跑"""
        input_file = tmp_path / "in.json"
        input_file.write_text(json.dumps(
            [{"id": i, "question": f"q{i}"} for i in range(1, 4)],
            ensure_ascii=False,
        ))
        output_file = tmp_path / "out.json"

        # 预先写入 id=1 已完成的 jsonl
        jsonl = _jsonl_path(output_file)
        jsonl.write_text(json.dumps({
            "id": 1, "question": "q1", "predicted_category": "其他",
            "confidence": "high", "reason": "pre-existing", "raw_output": "",
            "latency_ms": 0, "error": None,
        }, ensure_ascii=False) + "\n")

        llm, calls = self._make_llm()
        batch_classify(str(input_file), str(output_file), llm, concurrency=1)

        # llm 只被调了 id=2、3 两次
        assert len(calls) == 2
        assert "q1" not in calls

    def test_truncated_jsonl_tolerated(self, tmp_path: Path):
        """jsonl 最后一行被截断不应崩"""
        input_file = tmp_path / "in.json"
        input_file.write_text(json.dumps([{"id": 1, "question": "q1"}]))
        output_file = tmp_path / "out.json"
        jsonl = _jsonl_path(output_file)
        jsonl.write_text('{"id": 99, "question": "q99", "predicted_categ')  # 损坏

        llm, calls = self._make_llm()
        # 不该抛异常
        batch_classify(str(input_file), str(output_file), llm, concurrency=1)
        # id=1 应被处理（损坏行被忽略）
        assert len(calls) == 1


# ---------- 数据泄漏回归门 ----------

class TestNoDataLeak:
    def test_few_shot_not_in_test_samples(self):
        """B2 修复回归：few-shot 例子不能出现在 test_samples.json 中"""
        samples_path = (
            Path(__file__).resolve().parent.parent / "original" / "test_samples.json"
        )
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
        test_questions = {s["question"] for s in samples}

        # 从 FEW_SHOT 文本中粗略抽取"输入：xxx"行
        import re
        few_shot_inputs = re.findall(r"输入：(.+)", FEW_SHOT)
        assert len(few_shot_inputs) >= 5

        leaked = [q for q in few_shot_inputs if q.strip() in test_questions]
        assert leaked == [], f"数据泄漏！few-shot 含测试集样本：{leaked}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

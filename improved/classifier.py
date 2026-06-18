#!/usr/bin/env python3
"""
客服 FAQ 自动分类脚本 v2.0

工程化改进相比 v1.0：
  1. API key 走环境变量，不再硬编码
  2. 结构化 JSON 输出 + 解析校验 + 标签归一化兜底（防止 LLM 偏离 6 类）
  3. retry + 指数退避（处理瞬时网络/限流错误）
  4. 增量落盘 + 断点续传（中途崩溃不丢前面已分类的结果）
  5. 并发批量调用（ThreadPoolExecutor）
  6. 结构化日志（每条记录延迟、token、错误）
  7. mock 模式（用于 CI、回归、离线评估）
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

VALID_LABELS = {"退款退货", "物流查询", "账号问题", "商品咨询", "投诉建议", "其他"}
FALLBACK_LABEL = "其他"

logger = logging.getLogger("classifier")


@dataclass
class Config:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    seed: int = 42
    max_retries: int = 3
    retry_base_delay: float = 1.0
    concurrency: int = 5
    request_timeout: float = 30.0
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class ClassifyResult:
    id: int
    question: str
    predicted_category: str
    confidence: str = "high"
    reason: str = ""
    raw_output: str = ""
    latency_ms: int = 0
    error: Optional[str] = None


# ---------- Prompt ----------

SYSTEM_PROMPT = """你是一个电商客服 FAQ 分类助手。你的任务是把用户消息严格归类到下列 6 个标签之一。

# 标签定义

1. 退款退货 —— 用户要求退款、退货、换货，或咨询退款进度、退货流程、退货条件、退货邮费等。
   - 典型："我要退货"、"钱什么时候到账"、"退货邮费谁出"、"七天无理由怎么退"
   - 注意：即使提到"快递""到账"，只要核心诉求是"退款的钱/退货流程"，归此类。

2. 物流查询 —— 用户询问包裹当前位置、配送状态、派送地址、快递柜、签收异常等正向物流问题。
   - 典型："快递到哪了"、"显示签收但没收到"、"能改派送地址吗"、"放错快递柜了"
   - 注意：不包括"退款到账时间"——那属于"退款退货"。

3. 账号问题 —— 登录、密码、账号安全、绑定信息、账号状态等。
   - 典型："忘记密码"、"账号被冻结"、"修改手机号"、"异地登录提醒"

4. 商品咨询 —— 售前/售中咨询商品规格、材质、库存、价格、使用场景、兼容性等。
   - 典型："有蓝色吗"、"尺码怎么选"、"是真皮的吗"、"能带上飞机吗"

5. 投诉建议 —— 对服务/商品/流程的不满表达，或对平台的改进建议。语气通常带有抱怨、批评、举报、建议。
   - 典型："服务太差"、"什么破质量"、"建议增加 XX 功能"、"流程太麻烦"
   - 注意：即使含"退货"二字，只要主要诉求是抱怨流程或质量，归此类。
   - "举报"含具体投诉内容时归此类。

6. 其他 —— 闲聊、问候、致谢、纯标点、无意义内容、上述类别都不沾边的内容。
   - 典型："你好"、"嗯嗯好的谢谢"、"？？？"、纯辱骂（无具体投诉内容）

# 决策规则

- 一条消息同时涉及多个类别时，按用户**主要诉求**（句子主干、首要意图、明确指向的动作）归类。
- 含"退款"+"快递"双关键词时，主诉是钱→退款退货；主诉是包裹位置→物流查询。
- 含"退货"+"抱怨语气"时，主诉是流程不满→投诉建议；主诉是怎么操作→退款退货。
- 拿不准时归"其他"并把 confidence 标为 low，绝不要硬猜其他 5 类。

# 输出格式

严格输出以下 JSON，不要任何额外文字、不要 Markdown 代码块：

{"label": "<6个标签之一>", "confidence": "high|medium|low", "reason": "<不超过20字的判断依据>"}"""

FEW_SHOT = """示例 1
输入：钱啥时候能退到我账户
输出：{"label": "退款退货", "confidence": "high", "reason": "退款进度查询"}

示例 2
输入：希望可以推出会员专属优惠
输出：{"label": "投诉建议", "confidence": "high", "reason": "向平台提建议"}

示例 3
输入：我想取消订单同时问问发货了没
输出：{"label": "退款退货", "confidence": "medium", "reason": "主诉取消订单，物流为辅"}

示例 4
输入：你们退款审核怎么这么慢，等了一周了
输出：{"label": "投诉建议", "confidence": "high", "reason": "对退款流程的不满抱怨"}

示例 5
输入：啊好的明白了
输出：{"label": "其他", "confidence": "high", "reason": "闲聊确认"}
"""


def build_user_message(question: str) -> str:
    return (
        f"{FEW_SHOT}\n\n请对下面这条用户消息分类。\n\n"
        f"用户消息：{question}\n\n按 system 中定义的 JSON 格式输出。"
    )


# ---------- LLM 调用 ----------

LLMFn = Callable[[str], tuple[str, int]]  # (raw_output, latency_ms)


def make_openai_caller(cfg: Config) -> LLMFn:
    import openai

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"环境变量 {cfg.api_key_env} 未设置。请先 export {cfg.api_key_env}=sk-..."
        )

    client = openai.OpenAI(api_key=api_key, timeout=cfg.request_timeout)

    def call(question: str) -> tuple[str, int]:
        start = time.time()
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(question)},
            ],
            temperature=cfg.temperature,
            seed=cfg.seed,
            response_format={"type": "json_object"},
        )
        latency_ms = int((time.time() - start) * 1000)
        return resp.choices[0].message.content or "", latency_ms

    return call


def _is_retryable(exc: BaseException) -> bool:
    """
    只重试这些瞬时错误：限流、超时、连接中断、5xx。
    永远不重试：401/403（认证）、400（请求非法）、404（模型名错）等。
    这些错误重试 100 次也不会变，等待只会浪费时间并掩盖配置问题。
    """
    try:
        import openai
    except ImportError:  # 测试环境没装 openai 时退化为基于名字判断
        name = type(exc).__name__
        return name in {
            "RateLimitError", "APITimeoutError", "APIConnectionError",
            "InternalServerError", "TimeoutError", "ConnectionError",
        }
    return isinstance(exc, (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    ))


def _parse_retry_after(exc: BaseException) -> Optional[float]:
    """RateLimitError 通常携带 Retry-After header；尊重它而不是固定指数退避。"""
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", None) or {}
    try:
        ra = headers.get("retry-after") or headers.get("Retry-After")
        return float(ra) if ra else None
    except (TypeError, ValueError):
        return None


def with_retry(fn: LLMFn, cfg: Config) -> LLMFn:
    def wrapper(question: str) -> tuple[str, int]:
        last_err: Optional[BaseException] = None
        for attempt in range(cfg.max_retries):
            try:
                return fn(question)
            except BaseException as e:  # noqa: BLE001
                if not _is_retryable(e):
                    # 不可重试错误立即抛出，让调用方拿到清晰的失败信号
                    logger.error("不可重试错误（%s），立即放弃：%s", type(e).__name__, e)
                    raise
                last_err = e
                # 优先尊重 Retry-After，否则指数退避 + 小抖动
                from random import uniform
                ra = _parse_retry_after(e)
                delay = ra if ra is not None else (
                    cfg.retry_base_delay * (2 ** attempt) + uniform(0, 0.5)
                )
                logger.warning(
                    "LLM 瞬时错误 %s attempt=%d/%d，%.1fs 后重试",
                    type(e).__name__, attempt + 1, cfg.max_retries, delay,
                )
                time.sleep(delay)
        raise RuntimeError(f"重试 {cfg.max_retries} 次仍失败: {last_err}") from last_err

    return wrapper


# ---------- 输出解析 + 归一化 ----------

def parse_and_normalize(raw: str) -> tuple[str, str, str]:
    """
    解析 LLM 输出，返回 (label, confidence, reason)。
    任何解析失败、标签非法都归为 (其他, low, <原因>) 并保留原始信号。
    永远不抛异常，但**不再尝试从乱文里猜标签** —— 这种"宽容兜底"会把
    明显该报警的 prompt 失败，伪装成"看起来分对了"的脏数据。
    """
    raw_stripped = (raw or "").strip()
    if not raw_stripped:
        return FALLBACK_LABEL, "low", "LLM 返回空"

    try:
        obj = json.loads(raw_stripped)
    except json.JSONDecodeError as e:
        snippet = raw_stripped[:80].replace("\n", " ")
        return FALLBACK_LABEL, "low", f"JSON 解析失败: {e.msg} | 原文: {snippet}"

    if not isinstance(obj, dict):
        return FALLBACK_LABEL, "low", f"输出不是 JSON 对象: {type(obj).__name__}"

    label = str(obj.get("label", "")).strip()
    confidence = str(obj.get("confidence", "low")).strip().lower()
    reason = str(obj.get("reason", "")).strip()

    if label not in VALID_LABELS:
        return FALLBACK_LABEL, "low", f"非法标签 '{label}'，已兜底"
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return label, confidence, reason


# ---------- 单条分类 ----------

def classify_one(item: dict, llm: LLMFn) -> ClassifyResult:
    qid = item["id"]
    question = item["question"]
    try:
        raw, latency = llm(question)
        label, conf, reason = parse_and_normalize(raw)
        return ClassifyResult(
            id=qid, question=question,
            predicted_category=label, confidence=conf, reason=reason,
            raw_output=raw, latency_ms=latency,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("样本 id=%s 分类失败: %s", qid, e)
        return ClassifyResult(
            id=qid, question=question,
            predicted_category=FALLBACK_LABEL, confidence="low",
            reason="调用异常", error=str(e),
        )


# ---------- 批量分类 + 增量落盘 ----------

def _jsonl_path(out_path: Path) -> Path:
    """同名 .jsonl 作为崩溃容忍的增量日志"""
    return out_path.with_suffix(out_path.suffix + ".jsonl")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 容忍最后一行被截断
    return rows


def batch_classify(
    input_file: str,
    output_file: str,
    llm: LLMFn,
    concurrency: int = 5,
    resume: bool = True,
) -> list[ClassifyResult]:
    in_path = Path(input_file)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = _jsonl_path(out_path)

    items: list[dict] = json.loads(in_path.read_text(encoding="utf-8"))

    # 断点续传：优先从 .jsonl 增量日志读（更可靠），再降级到 .json 最终输出
    done: dict[int, ClassifyResult] = {}
    if resume:
        for row in _read_jsonl(jsonl_path):
            if row.get("error") is None:
                done[row["id"]] = ClassifyResult(**row)
        if not done and out_path.exists():
            try:
                for row in json.loads(out_path.read_text(encoding="utf-8")):
                    if row.get("error") is None:
                        done[row["id"]] = ClassifyResult(**row)
            except Exception as e:  # noqa: BLE001
                logger.warning("读取已有 .json 失败，将全部重跑: %s", e)
        if done:
            logger.info("断点续传：跳过 %d 条已完成的样本", len(done))

    pending = [it for it in items if it["id"] not in done]
    results: list[ClassifyResult] = list(done.values())
    write_lock = __import__("threading").Lock()

    def append_jsonl(res: ClassifyResult) -> None:
        """成功完成一条就 append 一行 —— 崩溃最多丢 1 条"""
        with write_lock:
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")

    def finalize_json() -> None:
        """所有任务跑完后，把 .jsonl 整理成有序 .json 数组（用户期望的输出形态）"""
        ordered = sorted(results, key=lambda r: r.id)
        out_path.write_text(
            json.dumps([asdict(r) for r in ordered], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(classify_one, it, llm): it for it in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            append_jsonl(res)  # 每完成一条立即落盘
            logger.info(
                "[%d/%d] id=%s label=%s conf=%s latency=%dms%s",
                i, len(pending), res.id, res.predicted_category,
                res.confidence, res.latency_ms,
                f" ERROR={res.error}" if res.error else "",
            )

    finalize_json()
    return sorted(results, key=lambda r: r.id)


# ---------- CLI ----------

def main() -> int:
    parser = argparse.ArgumentParser(description="FAQ 客服分类器 v2.0")
    parser.add_argument("input", help="输入 JSON（list of {id, question}）")
    parser.add_argument("output", help="输出 JSON")
    parser.add_argument("--mock", action="store_true", help="使用 mock LLM（不调真实 API）")
    parser.add_argument("--mock-prompt-version", choices=["v1", "v2"], default="v2",
                        help="mock 模式下模拟旧/新 prompt 行为（默认 v2）")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--no-resume", action="store_true", help="忽略已有结果，全部重跑")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = Config(model=args.model, concurrency=args.concurrency)
    if args.mock:
        # 延迟导入，使生产环境不需要 mock 模块
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
        from mock_llm import make_mock_caller  # type: ignore
        llm = make_mock_caller(prompt_version=args.mock_prompt_version)
        logger.info("使用 MOCK 模式 (prompt=%s)", args.mock_prompt_version)
    else:
        llm = with_retry(make_openai_caller(cfg), cfg)
        logger.info("使用真实 OpenAI API model=%s", cfg.model)

    results = batch_classify(
        args.input, args.output, llm,
        concurrency=args.concurrency, resume=not args.no_resume,
    )
    failed = sum(1 for r in results if r.error)
    print(f"分类完成 共 {len(results)} 条，失败 {failed} 条 → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

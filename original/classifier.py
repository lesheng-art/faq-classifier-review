#!/usr/bin/env python3
"""
客服 FAQ 自动分类脚本
用途：对用户发来的问题进行自动分类，分配到对应的客服组
（这是原始版本，已知存在问题，仅作为对比基线保留）
"""

import json
import openai

# API 配置
openai.api_key = "sk-REPLACE-WITH-YOUR-OPENAI-KEY"  # 占位符，原版硬编码真实 key（Code Review 指出的 P0 问题）
MODEL = "gpt-4o-mini"


def classify_question(question: str) -> str:
    """对单条用户问题进行分类"""
    prompt = f"""你是一个客服分类助手。请对以下用户问题进行分类。

分类类别：退款退货、物流查询、账号问题、商品咨询、投诉建议、其他

用户问题：{question}

请直接回复分类结果，只回复类别名称。"""

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    result = response.choices[0].message.content.strip()
    return result


def batch_classify(input_file: str, output_file: str):
    """批量分类"""
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    results = []
    for item in questions:
        question = item['question']
        category = classify_question(question)
        results.append({
            'id': item['id'],
            'question': question,
            'predicted_category': category
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"分类完成，共处理 {len(results)} 条问题")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python classifier.py <输入文件> <输出文件>")
        sys.exit(1)

    batch_classify(sys.argv[1], sys.argv[2])

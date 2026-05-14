import math
from matplotlib import pyplot as plt
from openai import OpenAI

from scripts.load_prompt import load_data, inject_data

client = OpenAI(base_url="https://api.apiyi.com/v1", api_key="sk-KwztT3ut4XQa4QbcEe5eB820D829476896C769784d64F0E8")

def compute_entropy(probs):
    return -sum(p * math.log(p + 1e-12) for p in probs)


# def get_entropy_trace(prompt, model="deepseek-v4-flash", top_k=5):
#     response = client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=512,
#         logprobs=True,
#         top_logprobs=top_k,
#         temperature=0
#     )

#     results = []

#     tokens = response.choices[0].logprobs.content

#     for token_info in tokens:
#         token = token_info.token  # ✅ 当前生成 token

#         probs = []
#         for t in token_info.top_logprobs:
#             probs.append(math.exp(t.logprob))

#         # 归一化
#         Z = sum(probs)
#         probs = [p / Z for p in probs]

#         entropy = compute_entropy(probs)

#         results.append({
#             "token": token,
#             "entropy": entropy
#         })

#     return results

import math

def get_entropy_trace(prompt, model="deepseek-v4-flash", top_k=5):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        logprobs=True,
        top_logprobs=top_k,
        temperature=0,
        stream=True   # ✅ 必须开启
    )

    results = []

    for chunk in response:
        choice = chunk.choices[0]

        # 有些 chunk 没有内容（比如 role）
        if not hasattr(choice, "delta") or choice.delta is None:
            continue

        logprobs = getattr(choice, "logprobs", None)
        if logprobs is None:
            continue

        # 每个 chunk 可能包含多个 token
        content = getattr(logprobs, "content", None)
        if content is None:
            continue

        for token_info in content:
            token = token_info.token

            probs = []
            for t in token_info.top_logprobs:
                probs.append(math.exp(t.logprob))

            if len(probs) == 0:
                continue

            # 归一化
            Z = sum(probs)
            probs = [p / Z for p in probs]

            entropy = compute_entropy(probs)

            results.append({
                "token": token,
                "entropy": entropy
            })

    return results


def sliding_window_stats(results, window_size=10):
    stats = []

    entropies = [r["entropy"] for r in results]

    for i in range(len(results)):
        start = max(0, i - window_size + 1)
        window = entropies[start:i+1]

        mu = sum(window) / len(window)
        sigma = (sum((x - mu) ** 2 for x in window) / len(window)) ** 0.5

        stats.append({
            "step": i,
            "token": results[i]["token"],
            "entropy": results[i]["entropy"],
            "mu": mu,
            "sigma": sigma
        })

    return stats

# Step   0 | Token='999' | H=0.0009 | μ=0.0009 | σ=0.0000
# Step   1 | Token='.' | H=0.0574 | μ=0.0292 | σ=0.0282
# Step   2 | Token='00' | H=0.0000 | μ=0.0195 | σ=0.0269


if __name__ == "__main__":
    dic = load_data()
    # prompt = f"""
    # 你是一个助手，分析网页，给出第一步应该做什么，{dic['task']},
    # {dic['clean']}
    # """
    prompt = f"""
    你是一个助手，分析网页，给出第一步应该做什么，{dic['task']},
    {dic['clean']}
    """

    results = get_entropy_trace(prompt)
    stats = sliding_window_stats(results)
    clean = []
    for s in stats:
        clean.append(s['entropy'])
        print(
            f"Step {s['step']:3d} | "
            f"Token='{s['token']}' | "
            f"H={s['entropy']:.4f} | "
            f"μ={s['mu']:.4f} | "
            f"σ={s['sigma']:.4f}"
        )
    

    inj_dic = inject_data(dic)
    prompt = f"""
    你是一个助手，分析网页，给出第一步应该做什么，{inj_dic['task']},
    {inj_dic['clean']}
    """

    results = get_entropy_trace(prompt)
    stats = sliding_window_stats(results)
    inject = []
    for s in stats:
        inject.append(s['entropy'])
        print(
            f"Step {s['step']:3d} | "
            f"Token='{s['token']}' | "
            f"H={s['entropy']:.4f} | "
            f"μ={s['mu']:.4f} | "
            f"σ={s['sigma']:.4f}"
        )
    # 创建1行2列的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 第一个子图 - 数组1
    ax1.plot(clean, 'o-', color='blue', linewidth=2, markersize=6)
    ax1.set_title('数组1', fontsize=14)
    ax1.set_xlabel('索引')
    ax1.set_ylabel('值')
    ax1.grid(True, alpha=0.3)

    # 第二个子图 - 数组2
    ax2.plot(inject, 's-', color='red', linewidth=2, markersize=6)
    ax2.set_title('数组2', fontsize=14)
    ax2.set_xlabel('索引')
    ax2.set_ylabel('值')
    ax2.grid(True, alpha=0.3)

    # 自动调整布局
    plt.tight_layout()
    plt.show()
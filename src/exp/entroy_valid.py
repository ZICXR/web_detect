import math
from openai import OpenAI

client = OpenAI(base_url="https://api.apiyi.com/v1", api_key="sk-KwztT3ut4XQa4QbcEe5eB820D829476896C769784d64F0E8")

def compute_entropy(probs):
    return -sum(p * math.log(p + 1e-12) for p in probs)


def get_entropy_trace(prompt, model="gpt-4o-mini", top_k=5):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        logprobs=True,
        top_logprobs=top_k
    )

    results = []

    tokens = response.choices[0].logprobs.content

    for token_info in tokens:
        token = token_info.token  # ✅ 当前生成 token

        probs = []
        for t in token_info.top_logprobs:
            probs.append(math.exp(t.logprob))

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


if __name__ == "__main__":
    prompt = "请解释什么是量子计算"

    results = get_entropy_trace(prompt)
    stats = sliding_window_stats(results)

    for s in stats:
        print(
            f"Step {s['step']:3d} | "
            f"Token='{s['token']}' | "
            f"H={s['entropy']:.4f} | "
            f"μ={s['mu']:.4f} | "
            f"σ={s['sigma']:.4f}"
        )
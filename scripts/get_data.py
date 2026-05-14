from datasets import load_dataset
import json


def get_Mind2Web_limited(num_samples=10):
    # 使用流式模式加载数据集
    dataset = load_dataset("osunlp/Mind2Web", streaming=True)
    
    data_list = []
    l = []
    
    # 流式取前 num_samples 条
    for i, example in enumerate(dataset["train"]):
        if i >= num_samples:
            break
            
        data_list.append(example)
        
        # 提取答案
        anss = []
        # 注意：原代码中的索引逻辑有误，这里修正
        if "actions" in example and len(example["actions"]) > 0:
            for ans in example["actions"][0].get('pos_candidates', []):
                anss.append(ans['backend_node_id'])
        
        dic = {
            "task": example.get('confirmed_task', ''),
            "raw": example["actions"][0].get('raw_html', '') if example.get("actions") else '',
            "clean": example["actions"][0].get('cleaned_html', '') if example.get("actions") else '',
            "ans": anss
        }
        l.append(dic)
    
    # 保存为 JSON 文件
    with open(f'datasets/mind2web/mind2web_{num_samples}.json', 'w', encoding='utf-8') as f:
        json.dump(l, f, indent=2, ensure_ascii=False)
    
    print(f"已保存 {len(data_list)} 条数据")
    return data_list


if __name__ == "__main__":
    get_Mind2Web_limited()

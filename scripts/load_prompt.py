import json

def del_id(content):
    import re
    a = re.sub(r'( backend_node_id=\"\d+\")', "", content)
    return a

def load_data():
    with open("datasets/mind2web/mind2web_10.json", 'r', encoding='utf-8') as f:
        dic = json.load(f)
    dic[0]['clean'] = del_id(dic[0]['clean'])
    return dic[0]

def inject_data(dic):
    import re
    data = []
    with open('datasets/webinjectbench/text/malicious/popup.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    a = data[79]
    a = a['text']
    inj_text = f"""
    <div">
        {a}
        <button onclick="this.parentElement.style.display='none'">确定</button>
    </div>
    """
    dic['clean'] = re.sub(r'(<body[^>]*>)', r'\1' + inj_text, dic['clean'], flags=re.IGNORECASE)
    print(dic['clean'])
    return dic
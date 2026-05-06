import json
import re

def build_prompt(template, sentences):
    numbered = "\n".join([f"{i}. {s}" for i, s in enumerate(sentences, start=1)])
    return template.replace("{sentences}", numbered)

def post_process_output(raw_text):
    """استخراج 10 جمل نظيفة من مخرج النموذج."""
    text = raw_text.strip()
    # إزالة أي tokens خاصة (Llama, Mistral, etc.)
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
    text = re.sub(r'(Human|Assistant|User|System)\s*:', '', text, flags=re.IGNORECASE)

    # 1. محاولة JSON
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list):
                sentences = [s.strip() for s in arr if isinstance(s, str) and s.strip()]
                while len(sentences) < 10:
                    sentences.append("")
                return sentences[:10]
        except:
            pass

    # 2. خطة بديلة: قائمة مرقمة
    lines = text.split('\n')
    candidates = []
    for line in lines:
        if re.match(r'^\s*(Human|Assistant|Note|ملاحظة|الخ)', line, re.IGNORECASE):
            continue
        m = re.match(r'^\s*(\d{1,2})\.?\s+(.+)', line)
        if m:
            seq = int(m.group(1))
            content = m.group(2).strip()
            content = re.split(r'\s*(Human|Assistant|Note|ملاحظة)\s*:', content)[0].strip()
            if content and not re.match(r'^\d+$', content):
                candidates.append((seq, content))
        else:
            stripped = line.strip().strip('"').strip(',')
            if stripped and not re.match(r'^[\[\]{}]', stripped) and not re.match(r'^\d+$', stripped):
                if not re.search(r'(مثال|الجمل|أخرج|أجب)', stripped, re.IGNORECASE):
                    candidates.append((len(candidates)+1, stripped))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        sentences = []
        for _, txt in candidates:
            if len(sentences) >= 10:
                break
            if txt not in sentences:
                sentences.append(txt)
        while len(sentences) < 10:
            sentences.append("")
        return sentences[:10]

    # 3. فشل تام
    return [""] * 10

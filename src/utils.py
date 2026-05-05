import json
import re

def build_prompt(template, sentences):
    """بناء البرومبت النهائي بترقيم الجمل المعطاة."""
    numbered = "\n".join([f"{i}. {s}" for i, s in enumerate(sentences, start=1)])
    return template.replace("{sentences}", numbered)

def post_process_output(raw_text):
    """استخراج قائمة من 10 جمل من مخرج النموذج (JSON أولاً، ثم خطة بديلة)."""
    # 1. محاولة استخراج JSON
    match = re.search(r'\[.*\]', raw_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                # ترشيح السلاسل النصية فقط
                sentences = [str(s).strip() for s in parsed if isinstance(s, str) or (isinstance(s, int) and s != '') ]
                # إزالة أي عناصر وهمية مثل "__" أو أرقام مفردة أتت بدون نص
                sentences = [s for s in sentences if s and not re.match(r'^_+$', s) and not re.match(r'^\d+$', s)]
                if len(sentences) >= 10:
                    return sentences[:10]
                else:
                    # إن كان عددها أقل، نملأ بفراغات
                    return sentences + [""] * (10 - len(sentences))
        except (json.JSONDecodeError, TypeError):
            pass
    
    # 2. خطة بديلة: البحث عن جمل مرقمة
    lines = raw_text.split('\n')
    sentences = []
    for line in lines:
        m = re.match(r'^\s*\d+\.\s*(.*)', line)
        if m:
            sentences.append(m.group(1).strip())
    sentences = [s for s in sentences if s and not re.match(r'^_+$', s) and not re.match(r'^\d+$', s)]
    if len(sentences) >= 10:
        return sentences[:10]
    
    # 3. فشل كل شيء: نعيد النص الخام كجملة أولى مع تنبيه
    return [raw_text.strip()] + [""] * 9

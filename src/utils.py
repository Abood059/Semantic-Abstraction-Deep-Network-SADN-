import json
import re

def build_prompt(template, sentences):
    """بناء البرومبت النهائي بترقيم الجمل المعطاة."""
    numbered = "\n".join([f"{i}. {s}" for i, s in enumerate(sentences, start=1)])
    return template.replace("{sentences}", numbered)

def post_process_output(raw_text):
    """
    تستخرج بالضبط 10 جمل من مخرج النموذج، مهما كان تنسيقه.
    تعالج JSON، وقوائم مرقمة، وتزيل الثرثرة الحوارية.
    """
    # --- تنظيف أولي ---
    text = raw_text.strip()
    # إزالة أي tokens خاصة بـ Llama (أو غيرها) قد تتسرب
    text = re.sub(r'<\|.*?\|>', '', text)
    # إزالة أي "Human:" أو "Assistant:" ظاهرية
    text = re.sub(r'\b(Human|Assistant|User|System)\s*:', '', text, flags=re.IGNORECASE)

    # --- 1. محاولة JSON صريحة ---
    # نبحث عن أول [ وآخر ]
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        json_candidate = text[start:end+1]
        try:
            arr = json.loads(json_candidate)
            if isinstance(arr, list):
                # استخراج السلاسل النصية غير الفارغة
                sentences = [str(s).strip() for s in arr if str(s).strip()]
                # نأخذ أول 10
                while len(sentences) < 10:
                    sentences.append("")
                return sentences[:10]
        except (json.JSONDecodeError, TypeError):
            # إذا فشل التحليل، ننتقل للخطة البديلة
            pass

    # --- 2. خطة بديلة: قائمة مرقمة (مع تجاهل الحوار) ---
    lines = text.split('\n')
    candidates = []
    for line in lines:
        # تجاهل الأسطر التي تبدأ بعبارات حوارية أو تعليمات
        if re.match(r'^\s*(Human|Assistant|Note|ملاحظة|لاحظ|الجمل المعطاة|المخرج|Output|[\(\)\{\}\[\]])', line, re.IGNORECASE):
            continue
        # نمط "رقم. نص"
        m = re.match(r'^\s*(\d{1,2})\.?\s+(.+)', line)
        if m:
            seq_num = int(m.group(1))
            content = m.group(2).strip()
            # إزالة أي حواشٍ في نفس السطر (مثل "Human:" بعد الجملة)
            content = re.split(r'\s*(?:Human|Assistant|Note|ملاحظة)\s*:', content)[0].strip()
            if content and not re.match(r'^\d+$', content):
                candidates.append((seq_num, content))
        else:
            # أسطر غير مرقمة قد تكون جملًا مفيدة
            stripped = line.strip().strip('"').strip(',').strip()
            if stripped and not re.match(r'^[\[\]\{\}]', stripped) and not re.match(r'^\d+$', stripped):
                # تجاهل الأسطر التي تبدو تعليمات
                if not re.search(r'(مثال|الجمل|أخرج|أجب|تعليمات|المطلوب)', stripped, re.IGNORECASE):
                    candidates.append((len(candidates)+1, stripped))

    if candidates:
        # ترتيب حسب الرقم
        candidates.sort(key=lambda x: x[0])
        sentences = []
        for _, txt in candidates:
            if len(sentences) >= 10:
                break
            if txt not in sentences:  # تجنب التكرار
                sentences.append(txt)
        while len(sentences) < 10:
            sentences.append("")
        return sentences[:10]

    # --- 3. فشل تام: إرجاع 10 فراغات ---
    return [""] * 10

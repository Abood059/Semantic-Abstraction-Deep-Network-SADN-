import re

def build_prompt(template, sentences):
    """بناء البرومبت النهائي باستبدال {sentences}."""
    # ترقيم الجمل من 1 إلى 10
    numbered = "\n".join([f"{i}. {s}" for i, s in enumerate(sentences, start=1)])
    return template.replace("{sentences}", numbered)

def post_process_output(raw_text):
    """استخراج الجمل العشر من مخرجات النموذج."""
    # نتوقع تنسيقًا مثل: 1. جملة ...\n2. جملة ...
    # نبحث عن أنماط "رقم. جملة"
    pattern = r'^\s*\d+\.\s*(.*)'
    lines = raw_text.split('\n')
    sentences = []
    for line in lines:
        match = re.match(pattern, line)
        if match:
            sentences.append(match.group(1).strip())
    # إذا لم نجد 10 جمل، حاول استخدام كل النص كجملة واحدة (احتياط)
    if len(sentences) < 10:
        # ربما أخرج النموذج جمل بدون ترقيم، نقسم على نقاط نهاية الجمل
        alt_sentences = re.split(r'(?<=[.!؟])\s+', raw_text.strip())
        sentences = [s.strip() for s in alt_sentences if s.strip()][:10]
    return sentences

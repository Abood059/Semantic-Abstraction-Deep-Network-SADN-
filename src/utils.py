def post_process_output(raw_text):
    """استخراج قائمة من 10 جمل من مخرجات النموذج، بأي تنسيق."""
    # 1. محاولة JSON أولاً (تبقى موجودة إن عاد النموذج لاستخدامها)
    match = re.search(r'\[.*\]', raw_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                sentences = [str(s).strip() for s in parsed if str(s).strip()]
                if len(sentences) >= 10:
                    return sentences[:10]
        except (json.JSONDecodeError, TypeError):
            pass

    # 2. البحث عن جمل مرقمة (مثل: 1. جملة... 2. جملة...)
    # نبحث عن أرقام تبدأ من 1 وتتزايد، مع نص بعدها
    # نستخدم regex لالتقاط "رقم. نص" عبر السطور
    pattern = r'(?:^|\n)\s*(\d{1,3})\.?\s+(.+?)(?=\n\s*\d{1,3}\.?\s+|$)'
    matches = re.findall(pattern, raw_text, re.DOTALL)
    if matches:
        # نأخذ أول 10 جمل حسب الترقيم
        sentences = [m[1].strip() for m in matches if m[1].strip()]
        if len(sentences) >= 10:
            return sentences[:10]
        # إذا كانت بعض الأرقام ناقصة فنملأ الباقي
        while len(sentences) < 10:
            sentences.append("")
        return sentences[:10]

    # 3. إذا لم نجد شيئاً، نعيد النص الخام كجملة واحدة (مع 9 فراغات)
    return [raw_text.strip()] + [""] * 9

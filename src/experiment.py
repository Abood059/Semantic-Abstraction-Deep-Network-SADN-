import torch
import logging
import json
import os
from src.database import add_results, get_inputs_for_node
from src.utils import build_prompt, post_process_output

def generate_layer_batch(config, model, tokenizer, prompts):
    """
    prompts: قائمة من النصوص (طولها num_nodes)
    تُعيد قائمة من القوائم (num_nodes × 10 جمل)
    """
    # ترميز الدفعة مع تبطين (padding) تلقائي
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048   # كافٍ جداً
    ).to(model.device)
    
    # استخدام inference_mode الأسرع من no_grad
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config['max_new_tokens'],
            temperature=config['temperature'],
            do_sample=config.get('do_sample', True),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True   # تسريع إضافي
        )
    
    # استخراج الردود لكل عقدة
    all_sentences = []
    for i, out in enumerate(outputs):
        prompt_len = inputs['input_ids'][i].size(0)  # طول برومبت هذه العقدة
        generated_ids = out[prompt_len:]  # نأخذ فقط الجزء المولّد
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        sentences = post_process_output(response)
        # ضمان 10 جمل بالضبط
        while len(sentences) < 10:
            sentences.append("")
        all_sentences.append(sentences[:10])
    return all_sentences

def run_experiment(config, model, tokenizer):
    num_layers = config['num_layers']
    num_nodes = config['num_nodes_per_layer']
    prompt_template = config['prompt_template']

    # تحميل الجمل الأولية
    json_path = os.path.join("data", "initial_sentences.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            initial_sentences = json.load(f)
        logging.info(f"تم تحميل {len(initial_sentences)} جملة أولية من {json_path}")
    else:
        logging.warning(f"الملف {json_path} غير موجود. سيتم توليد جمل عشوائية.")
        import random
        random.seed(42)
        initial_sentences = [f"الجملة العشوائية رقم {i+1} لمحاكاة الفوضى." for i in range(100)]

    for layer in range(1, num_layers + 1):
        logging.info(f"===== بدء الطبقة {layer} =====")
        # بناء المطالبات لكل العقد دفعة واحدة
        prompts = []
        for node in range(1, num_nodes + 1):
            if layer == 1:
                start_idx = (node - 1) * 10
                end_idx = start_idx + 10
                input_sentences = initial_sentences[start_idx:end_idx]
            else:
                input_sentences = get_inputs_for_node(layer, node, num_nodes)
            prompt = build_prompt(prompt_template, input_sentences)
            prompts.append(prompt)
        
        try:
            batch_sentences = generate_layer_batch(config, model, tokenizer, prompts)
            for node_idx, sentences in enumerate(batch_sentences, start=1):
                add_results(layer, node_idx, sentences)
            logging.info(f"الطبقة {layer} تمت بنجاح.")
        except Exception as e:
            logging.error(f"خطأ في الطبقة {layer}: {e}")
            for node in range(1, num_nodes + 1):
                add_results(layer, node, [f"ERROR: {str(e)}"] * 10)

import torch
import logging
import json
import os
from src.database import add_results, get_inputs_for_node
from src.utils import build_prompt, post_process_output

def generate_layer_batch(config, model, tokenizer, prompts, num_sentences):
    """
    prompts: قائمة نصوص (طولها num_nodes)
    تُعيد قائمة من القوائم (num_nodes × num_sentences جمل)
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config['max_new_tokens'],
            temperature=config['temperature'],
            do_sample=config.get('do_sample', True),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    all_sentences = []
    for i, out in enumerate(outputs):
        prompt_len = inputs['input_ids'][i].size(0)
        generated_ids = out[prompt_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        sentences = post_process_output(response, num_sentences=num_sentences)
        all_sentences.append(sentences[:num_sentences])
    return all_sentences

def run_experiment(config, model, tokenizer):
    num_layers = config['num_layers']
    num_nodes = config['num_nodes_per_layer']
    num_sentences = config.get('num_output_sentences', 5)
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
        initial_sentences = [f"Random sentence number {i+1} simulating chaos." for i in range(num_nodes * num_sentences)]

    # توزيع الجمل على عقد الطبقة الأولى: كل عقدة تستلم num_sentences جمل
    for layer in range(1, num_layers + 1):
        logging.info(f"===== بدء الطبقة {layer} =====")
        prompts = []
        for node in range(1, num_nodes + 1):
            if layer == 1:
                start_idx = (node - 1) * num_sentences
                end_idx = start_idx + num_sentences
                input_sentences = initial_sentences[start_idx:end_idx]
            else:
                input_sentences = get_inputs_for_node(layer, node, num_nodes)
            prompt = build_prompt(prompt_template, input_sentences)
            prompts.append(prompt)
        
        try:
            batch_sentences = generate_layer_batch(config, model, tokenizer, prompts, num_sentences)
            for node_idx, sentences in enumerate(batch_sentences, start=1):
                add_results(layer, node_idx, sentences)
            logging.info(f"الطبقة {layer} تمت بنجاح.")
        except Exception as e:
            logging.error(f"خطأ في الطبقة {layer}: {e}")
            for node in range(1, num_nodes + 1):
                add_results(layer, node, [f"ERROR: {str(e)}"] * num_sentences)

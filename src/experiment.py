import logging
from src.database import add_results, get_inputs_for_node
from src.utils import build_prompt, post_process_output

def generate_sentences(config, model, tokenizer, prompt):
    """استدعاء النموذج وإرجاع قائمة من 10 جمل."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config['max_new_tokens'],
            temperature=config['temperature'],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    # فك التشفير وإزالة البرومبت الأصلي
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # إزالة البرومبت من البداية (قد يختلف حسب النموذج)
    response = full_response[len(prompt):].strip()
    sentences = post_process_output(response)
    return sentences

def run_experiment(config, model, tokenizer):
    num_layers = config['num_layers']
    num_nodes = config['num_nodes_per_layer']
    prompt_template = config['prompt_template']
    
    # توليد الجمل الأولية عشوائيًا (أو تحميلها من ملف)
    # هنا سنقوم بإنشاء 100 جملة عشوائية ثابتة للتجربة
    import random
    random.seed(42)  # بذرة للتكرار
    initial_sentences = [
        f"الجملة العشوائية رقم {i+1} لمحاكاة الفوضى."
        for i in range(100)
    ]
    # يمكنك استبدال السطر أعلاه بتحميل من ملف JSON لاحقًا
    
    # حفظ الجمل الأولية كأنها طبقة 0 عقد وهمية؟ 
    # سنخصص توزيعها على عقد الطبقة 1 مباشرة:
    # لكل عقدة 10 جمل مختلفة (مثلاً عقدة i تأخذ الجمل من i*10 إلى (i+1)*10-1)
    # هذا يتوافق مع فكرة أن كل عقدة تحصل على 10 جمل مختلفة تمامًا.
    
    for layer in range(1, num_layers + 1):
        logging.info(f"===== بدء الطبقة {layer} =====")
        for node in range(1, num_nodes + 1):
            logging.info(f"معالجة العقدة {node} في الطبقة {layer}")
            
            if layer == 1:
                # الطبقة 1: خذ 10 جمل من القائمة الأولية الموزعة
                start_idx = (node - 1) * 10
                end_idx = start_idx + 10
                input_sentences = initial_sentences[start_idx:end_idx]
            else:
                # الطبقات الأخرى: اجمع الجملة رقم (node) من كل عقدة سابقة
                input_sentences = get_inputs_for_node(layer, node, num_nodes)
            
            # بناء البرومبت
            prompt = build_prompt(prompt_template, input_sentences)
            
            # توليد الجمل
            try:
                generated = generate_sentences(config, model, tokenizer, prompt)
                # التحقق من عدد الجمل
                if len(generated) < 10:
                    # تعبئة الفارق بجمل فارغة
                    generated += [""] * (10 - len(generated))
                elif len(generated) > 10:
                    generated = generated[:10]
                
                add_results(layer, node, generated)
                logging.info(f"تم إنتاج 10 جمل للعقدة {node}.")
            except Exception as e:
                logging.error(f"خطأ في الطبقة {layer} العقدة {node}: {e}")
                # في حالة الخطأ، نخزن جمل خطأ
                add_results(layer, node, [f"ERROR: {str(e)}"] * 10)
                # يمكن أن نوقف التجربة أو نستمر
                continue

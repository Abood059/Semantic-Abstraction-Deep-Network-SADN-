import argparse
import yaml
import logging
import os
from pathlib import Path
from src.model_loader import load_model
from src.database import init_db, add_results, get_inputs_for_node
from src.experiment import run_experiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(config_path):
    # تحميل الإعدادات
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logging.info("بدء التجربة بالإعدادات:")
    logging.info(f"الطبقات: {config['num_layers']} | العقد لكل طبقة: {config['num_nodes_per_layer']}")
    logging.info(f"النموذج: {config['model_name']}")

    # عرض المسار الحالي لضمان الشفافية
    logging.info(f"مجلد العمل الحالي: {os.getcwd()}")
    # حساب وعرض المسار المطلق لقاعدة البيانات
    db_path = config['db_path']
    if not os.path.isabs(db_path):
        db_path = os.path.abspath(db_path)
    logging.info(f"المسار المطلق لقاعدة البيانات: {db_path}")

    # تحميل النموذج
    model, tokenizer = load_model(config)
    
    # تجهيز قاعدة البيانات (ستستخدم المسار من config وتجعله مطلقاً داخل init_db)
    init_db(config['db_path'])
    
    # تشغيل التجربة
    run_experiment(config, model, tokenizer)
    
    logging.info("اكتملت التجربة بنجاح!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Distiller experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)

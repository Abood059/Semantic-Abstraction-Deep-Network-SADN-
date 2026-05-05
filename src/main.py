import argparse
import yaml
import logging
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

    # تحميل النموذج
    model, tokenizer = load_model(config)
    
    # تجهيز قاعدة البيانات
    init_db(config['db_path'])
    
    # تشغيل التجربة
    run_experiment(config, model, tokenizer)
    
    logging.info("اكتملت التجربة بنجاح!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Distiller experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)

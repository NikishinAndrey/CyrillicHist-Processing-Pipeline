#!/usr/bin/env python3
"""
Скрипт для запуска пайплайна обработки исторических рукописей
"""

import argparse
import cv2
import os
import sys
from pathlib import Path

# Добавление пути к src
sys.path.insert(0, str(Path(__file__).parent))

from src import InferencePipeline
from src.segmentation_visualizer import SegmentationVisualizer

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline for processing historical manuscripts"
    )
    
    parser.add_argument(
        "--image", 
        type=str, 
        required=True,
        help="Path to input manuscript image"
    )
    
    parser.add_argument(
        "--detect-model", 
        type=str, 
        default="models/v1/yolo_detect_corpus/detect_n_model/weights/best.pt",
        help="Path to text block detection model"
    )
    
    parser.add_argument(
        "--segment-model", 
        type=str, 
        default="models/v1/yolo_segment_lines/segment_n_model/weights/best.pt",
        help="Path to line segmentation model"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Directory for output results"
    )
    
    parser.add_argument(
        "--target-size", 
        type=int, 
        default=1280,
        help="Target image size for processing"
    )
    
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Don't save visualization, only return results"
    )
    
    args = parser.parse_args()
    
    # Проверка существования файла
    if not os.path.exists(args.image):
        print(f"❌ Файл не найден: {args.image}")
        return 1
    
    # Проверка моделей
    for model_path in [args.detect_model, args.segment_model]:
        if not os.path.exists(model_path):
            print(f"⚠️  Модель не найдена: {model_path}")
            print("Пожалуйста, укажите правильные пути к моделям.")
    
    try:
        # Инициализация пайплайна
        print("🚀 Инициализация пайплайна...")
        pipeline = InferencePipeline(
            detect_model_path=args.detect_model,
            segment_model_path=args.segment_model,
            target_size=args.target_size
        )
        
        # Запуск пайплайна
        print(f"📄 Обработка изображения: {args.image}")
        
        if args.no_save:
            # Только инференс
            results = pipeline.run(args.image)
            print(f"✅ Обработка завершена")
            print(f"   Найдено текстовых блоков: {len(results['detect_corpuses'])}")
            
            total_lines = sum(len(corpus["segment_lines"]) 
                            for corpus in results["detect_corpuses"])
            print(f"   Найдено строк: {total_lines}")
        else:
            # Инференс с сохранением
            results = pipeline.run_and_save(
                img_path=args.image,
                output_dir=args.output_dir,
                base_filename=Path(args.image).stem
            )
            print(f"✅ Результаты сохранены в: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Ошибка при обработке: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
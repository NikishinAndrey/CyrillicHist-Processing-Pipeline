import cv2
import os
from typing import Dict, Any, Optional
from .corpus_detector import CorpusDetector
from .line_segmenter import LineSegmenter
from .segmentation_visualizer import SegmentationVisualizer

class InferencePipeline:
    """Основной пайплайн инференса для обработки исторических рукописей"""
    
    def __init__(self, detect_model_path: str, 
                 segment_model_path: str, 
                 target_size: int = 1280):
        """
        Инициализация пайплайна
        
        Args:
            detect_model_path: Путь к модели детекции корпусов
            segment_model_path: Путь к модели сегментации строк
            target_size: Размер изображения для обработки
        """
        self.detector = CorpusDetector(detect_model_path, target_size)
        self.segmenter = LineSegmenter(segment_model_path, target_size)
        self.target_size = target_size
    
    def run(self, img_path: str) -> Dict[str, Any]:
        """
        Полный пайплайн обработки изображения
        
        Args:
            img_path: Путь к изображению рукописи
            
        Returns:
            Словарь с результатами детекции и сегментации
            
        Raises:
            ValueError: Если изображение не загружается
        """
        # Загрузка изображения
        original_img = cv2.imread(img_path)
        if original_img is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")
        
        original_height, original_width = original_img.shape[:2]
        
        # Детекция корпусов
        print("🔍 Детекция корпусов текста...")
        bboxes_original = self.detector.detect(original_img)
        
        if not bboxes_original:
            return {
                "img_path": img_path,
                "image_size": {"width": original_width, "height": original_height},
                "detect_text_block": []
            }
        
        print(f"✅ Найдено корпусов: {len(bboxes_original)}")
        
        # Сегментация строк для каждого корпуса
        results = {
            "img_path": img_path,
            "image_size": {"width": original_width, "height": original_height},
            "detect_text_block": []
        }
        
        for i, bbox_original in enumerate(bboxes_original):
            print(f"📝 Обработка корпуса {i+1}/{len(bboxes_original)}...")
            
            x1, y1, x2, y2 = map(int, bbox_original)
            corpus_img = original_img[y1:y2, x1:x2]
            
            if corpus_img.size == 0:
                continue
            
            # Сегментация строк
            segment_lines = self.segmenter.segment_corpus(
                corpus_img, bbox_original, original_width, original_height
            )
            
            results["detect_text_block"].append({
                "number_text_block": i + 1,
                "bbox": bbox_original,
                "segment_lines": segment_lines
            })
            
            print(f"   Найдено строк: {len(segment_lines)}")
        
        total_lines = sum(len(corpus["segment_lines"]) for corpus in results["detect_text_block"])
        print(f"🎯 Всего обработано строк: {total_lines}")
        
        return results
    
    def run_and_save(self, img_path: str, 
                    output_dir: str = "results",
                    base_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Полный пайплайн с сохранением результатов
        
        Args:
            img_path: Путь к изображению
            output_dir: Директория для сохранения результатов
            base_filename: Базовое имя файлов (по умолчанию - имя файла изображения)
            
        Returns:
            Результаты обработки
        """
        # Определение базового имени
        if base_filename is None:
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Запуск пайплайна
        results = self.run(img_path)
        
        # Загрузка изображения для визуализации
        original_img = cv2.imread(img_path)
        
        # Сохранение результатов
        os.makedirs(output_dir, exist_ok=True)
        
        # Визуализация
        SegmentationVisualizer.save_segmentation_results(
            results, output_dir, original_img, base_filename
        )
        
        # JSON аннотация
        json_path = os.path.join(output_dir, f"{base_filename}_annotation.json")
        SegmentationVisualizer.create_json_annotation(results, json_path)
        
        # Отдельная детекция (опционально)
        detect_dir = os.path.join(output_dir, "detection")
        self.detector.save_detection_results(
            original_img, [corpus["bbox"] for corpus in results["detect_text_block"]],
            detect_dir, base_filename
        )
        
        return results
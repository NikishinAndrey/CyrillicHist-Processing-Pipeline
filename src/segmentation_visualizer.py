import cv2
import numpy as np
import os
from typing import Dict, List, Any, Optional
from .image_processor import ImageProcessor

class SegmentationVisualizer:
    """Класс для визуализации и сохранения результатов сегментации"""
    
    @staticmethod
    def save_single_line(original_img: np.ndarray,
                        corpus_bbox: List[float],
                        line_mask: List[List[float]],
                        line_number: int,
                        corpus_number: int,
                        output_dir: str,
                        base_filename: str) -> Optional[str]:
        """
        Сохранение отдельной строки с бинаризацией
        
        Args:
            original_img: Исходное изображение
            corpus_bbox: Bbox корпуса
            line_mask: Маска строки
            line_number: Номер строки
            corpus_number: Номер корпуса
            output_dir: Директория для сохранения
            base_filename: Базовое имя файла
            
        Returns:
            Путь к сохраненному файлу или None
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Создание bounding box для строки
        mask_array = np.array(line_mask, dtype=np.float32)
        x_min, y_min = mask_array.min(axis=0)
        x_max, y_max = mask_array.max(axis=0)
        
        # Добавление паддинга
        padding = 5
        x_min = max(0, int(x_min) - padding)
        y_min = max(0, int(y_min) - padding)
        x_max = min(original_img.shape[1], int(x_max) + padding)
        y_max = min(original_img.shape[0], int(y_max) + padding)
        
        # Вырезка области строки
        line_region = original_img[y_min:y_max, x_min:x_max].copy()
        
        if line_region.size == 0:
            return None
        
        # Создание маски для строки
        mask_region = np.zeros((line_region.shape[0], line_region.shape[1]), dtype=np.uint8)
        mask_points_relative = mask_array - [x_min, y_min]
        cv2.fillPoly(mask_region, [mask_points_relative.astype(np.int32)], 255)
        
        # Наложение маски
        initial_white_bg = line_region.copy()
        initial_white_bg[mask_region == 0] = 255
        
        # Бинаризация
        gray_scale = cv2.cvtColor(initial_white_bg, cv2.COLOR_RGB2GRAY)
        threshold = ImageProcessor.calculate_threshold(gray_scale)
        
        _, thresh_img = cv2.threshold(gray_scale, threshold, 255, cv2.THRESH_BINARY)
        closed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, 
                                     np.ones((3, 3), np.uint8), iterations=1)
        
        # Создание финального изображения
        closed_mask = np.stack([closed_img] * 3, axis=-1)
        white_background = np.ones_like(initial_white_bg) * 255
        result_img = np.where(closed_mask == 255, white_background, initial_white_bg)
        
        # Сохранение
        filename = f"{base_filename}_corpus_{corpus_number}_line_{line_number}.jpg"
        img_path = os.path.join(output_dir, filename)
        cv2.imwrite(img_path, result_img)
        
        return img_path
    
    @staticmethod
    def save_segmentation_results(results: Dict[str, Any],
                                 output_dir: str,
                                 original_img: np.ndarray,
                                 base_filename: str = "segmentation") -> None:
        """
        Сохранение полных результатов сегментации
        
        Args:
            results: Результаты пайплайна
            output_dir: Директория для сохранения
            original_img: Исходное изображение
            base_filename: Базовое имя файла
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Визуализация
        vis_img = original_img.copy()
        
        for corpus in results["detect_corpuses"]:
            # Отрисовка корпусов
            x1, y1, x2, y2 = map(int, corpus["bbox"])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(vis_img, f"Corpus {corpus['number_corpus']}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Отрисовка масок строк
            for line in corpus["segment_lines"]:
                mask_points = np.array(line["mask"], dtype=np.int32)
                cv2.polylines(vis_img, [mask_points], True, (255, 0, 0), 2)
                cv2.putText(vis_img, f"L{line['number_line']}",
                           (mask_points[0][0], mask_points[0][1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Сохранение визуализации
        vis_filename = f"{base_filename}_visualization.jpg"
        img_path = os.path.join(output_dir, vis_filename)
        cv2.imwrite(img_path, vis_img)
        
        # Сохранение отдельных строк
        lines_dir = os.path.join(output_dir, "lines")
        os.makedirs(lines_dir, exist_ok=True)
        
        for corpus in results["detect_corpuses"]:
            for line in corpus["segment_lines"]:
                SegmentationVisualizer.save_single_line(
                    original_img=original_img,
                    corpus_bbox=corpus["bbox"],
                    line_mask=line["mask"],
                    line_number=line["number_line"],
                    corpus_number=corpus["number_corpus"],
                    output_dir=lines_dir,
                    base_filename=base_filename
                )
        
        print(f"Результаты сегментации сохранены в: {output_dir}")
    
    @staticmethod
    def create_json_annotation(results: Dict[str, Any], 
                             output_path: str) -> None:
        """
        Создание JSON аннотации в формате для публикации
        
        Args:
            results: Результаты пайплайна
            output_path: Путь для сохранения JSON файла
        """
        import json
        
        annotation = {
            "image_path": results["img_path"],
            "image_size": {
                "width": None,
                "height": None
            },
            "corpuses": []
        }
        
        # Получение размеров изображения
        try:
            img = cv2.imread(results["img_path"])
            if img is not None:
                h, w = img.shape[:2]
                annotation["image_size"]["width"] = int(w)
                annotation["image_size"]["height"] = int(h)
        except:
            pass
        
        # Заполнение данных о корпусах
        for corpus in results["detect_corpuses"]:
            corpus_data = {
                "number_corpus": corpus["number_corpus"],
                "bbox": corpus["bbox"],
                "lines": []
            }
            
            for line in corpus["segment_lines"]:
                line_data = {
                    "number_line": line["number_line"],
                    "mask": line["mask"]
                }
                corpus_data["lines"].append(line_data)
            
            annotation["corpuses"].append(corpus_data)
        
        # Сохранение JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)
        
        print(f"JSON аннотация сохранена: {output_path}")
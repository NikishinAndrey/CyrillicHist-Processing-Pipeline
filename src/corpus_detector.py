import cv2
import numpy as np
import os
from typing import List, Tuple
from ultralytics import YOLO
from .image_processor import ImageProcessor, ImageMetadata

class CorpusDetector:
    """Класс для детекции корпусов текста в исторических рукописях"""
    
    def __init__(self, model_path: str, target_size: int = 1280):
        """
        Инициализация детектора
        
        Args:
            model_path: Путь к модели YOLO
            target_size: Размер изображения для инференса
        """
        self.model = YOLO(model_path)
        self.target_size = target_size
    
    def detect(self, image: np.ndarray) -> List[List[float]]:
        """
        Детекция корпусов текста на изображении
        
        Args:
            image: Входное изображение (RGB или BGR)
            
        Returns:
            Список bounding boxes в формате [[x1, y1, x2, y2], ...]
        """
        # Ресайз изображения
        resized_img, metadata = ImageProcessor.resize_image_with_padding(image, self.target_size)
        
        # Инференс модели
        detect_results = self.model(resized_img, imgsz=self.target_size, conf=0.45, iou=0.5)
        
        # Проверка результатов
        if len(detect_results) == 0 or detect_results[0].boxes is None:
            return []
        
        # Извлечение bounding boxes
        bboxes_resized = detect_results[0].boxes.xyxy.cpu().numpy()
        scores = detect_results[0].boxes.conf.cpu().numpy()
        
        # Фильтрация по размеру
        size_indices = self._filter_by_size(bboxes_resized, min_size_ratio=0.65)
        bboxes_resized = bboxes_resized[size_indices]
        scores = scores[size_indices]
        
        # Конвертация координат в исходную систему
        original_height, original_width = image.shape[:2]
        bboxes_original = []
        
        for bbox in bboxes_resized:
            bbox_original = self._convert_bbox_to_original_coords(
                bbox, metadata, original_width, original_height
            )
            bboxes_original.append(bbox_original)
        
        # Сортировка по координате X
        bboxes_original = sorted(bboxes_original, key=lambda bbox: bbox[0])
        
        return bboxes_original
    
    def _filter_by_size(self, boxes: np.ndarray, min_size_ratio: float = 0.65) -> List[int]:
        """
        Фильтрация bounding boxes по размеру
        
        Args:
            boxes: Массив bounding boxes
            min_size_ratio: Минимальное отношение размера к максимальному
            
        Returns:
            Индексы валидных боксов
        """
        if len(boxes) == 0:
            return []
        
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        
        max_width = np.max(widths) if len(widths) > 0 else 0
        max_height = np.max(heights) if len(heights) > 0 else 0
        
        keep_indices = []
        for i, box in enumerate(boxes):
            width = box[2] - box[0]
            height = box[3] - box[1]
            
            if width >= max_width * min_size_ratio and height >= max_height * min_size_ratio:
                keep_indices.append(i)
        
        return keep_indices
    
    def _convert_bbox_to_original_coords(self, bbox_resized: np.ndarray, 
                                        metadata: ImageMetadata,
                                        original_width: int, 
                                        original_height: int) -> List[float]:
        """
        Конвертация координат bounding box из ресайзнутого изображения в исходное
        
        Args:
            bbox_resized: Bbox в координатах ресайзнутого изображения
            metadata: Метаданные ресайза
            original_width: Ширина исходного изображения
            original_height: Высота исходного изображения
            
        Returns:
            Bbox в координатах исходного изображения
        """
        x1_res, y1_res, x2_res, y2_res = bbox_resized
        
        # Обратное преобразование координат
        x1_abs = (x1_res - metadata.x_offset) / metadata.scale
        y1_abs = (y1_res - metadata.y_offset) / metadata.scale
        x2_abs = (x2_res - metadata.x_offset) / metadata.scale
        y2_abs = (y2_res - metadata.y_offset) / metadata.scale
        
        # Добавление паддинга
        bbox_width = x2_abs - x1_abs
        bbox_height = y2_abs - y1_abs
        
        padding_x = bbox_width * 0.1
        padding_y = bbox_height * 0.03
        
        # Обрезка по границам изображения
        x1_original = max(0, min(original_width, x1_abs - padding_x))
        y1_original = max(0, min(original_height, y1_abs - padding_y))
        x2_original = max(0, min(original_width, x2_abs + padding_x))
        y2_original = max(0, min(original_height, y2_abs + padding_y))
        
        return [float(x1_original), float(y1_original), 
                float(x2_original), float(y2_original)]
    
    def save_detection_results(self, image: np.ndarray, 
                              bboxes: List[List[float]], 
                              output_folder: str,
                              base_filename: str = "detection") -> None:
        """
        Сохранение результатов детекции
        
        Args:
            image: Исходное изображение
            bboxes: Список bounding boxes
            output_folder: Папка для сохранения
            base_filename: Базовое имя файла
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Визуализация
        vis_img = image.copy()
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(vis_img, f"Corpus {i+1}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Сохранение визуализации
        vis_path = os.path.join(output_folder, f"{base_filename}_visualization.jpg")
        cv2.imwrite(vis_path, vis_img)
        
        # Сохранение отдельных корпусов
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            corpus_img = image[y1:y2, x1:x2]
            if corpus_img.size > 0:
                corpus_path = os.path.join(output_folder, f"{base_filename}_corpus_{i+1}.jpg")
                cv2.imwrite(corpus_path, corpus_img)
        
        print(f"Результаты детекции сохранены в: {output_folder}")
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from shapely.geometry import Polygon
from ultralytics import YOLO
from .image_processor import ImageProcessor, ImageMetadata

class LineSegmenter:
    """Класс для сегментации строк текста внутри корпусов"""
    
    def __init__(self, model_path: str, target_size: int = 1280):
        """
        Инициализация сегментатора
        
        Args:
            model_path: Путь к модели YOLO для сегментации
            target_size: Размер изображения для инференса
        """
        self.model = YOLO(model_path)
        self.target_size = target_size
    
    def segment_corpus(self, corpus_img: np.ndarray, 
                      bbox_original: List[float],
                      original_width: int,
                      original_height: int) -> List[Dict[str, Any]]:
        """
        Сегментация строк внутри одного корпуса текста
        
        Args:
            corpus_img: Изображение корпуса текста
            bbox_original: Bbox корпуса в исходных координатах
            original_width: Ширина исходного изображения
            original_height: Высота исходного изображения
            
        Returns:
            Список сегментированных строк с масками
        """
        # Ресайз корпуса
        resized_corpus, metadata = ImageProcessor.resize_image_with_padding(
            corpus_img, target_size=self.target_size
        )
        
        # Скользящее окно по высоте
        windows, window_x_padding = self._create_sliding_windows(resized_corpus)
        
        all_segment_lines = []
        
        # Обработка каждого окна
        for window_img, window_y_offset, win_x_pad in windows:
            segment_lines = self._process_window(
                window_img, window_y_offset, win_x_pad,
                bbox_original, metadata, 
                original_width, original_height
            )
            all_segment_lines.extend(segment_lines)
        
        # Фильтрация и сортировка масок
        filtered_masks = self._filter_segmentation_masks(all_segment_lines)
        
        return filtered_masks
    
    def _create_sliding_windows(self, image: np.ndarray) -> Tuple[List[Tuple], int]:
        """
        Создание скользящих окон для обработки высоких изображений
        
        Args:
            image: Входное изображение
            
        Returns:
            Кортеж (список окон, паддинг по X)
        """
        windows = []
        img_height, img_width = image.shape[:2]
        
        # Добавление паддинга по ширине
        x_offset = 0
        if img_width < self.target_size:
            padded_image = np.zeros((img_height, self.target_size, 3), dtype=image.dtype)
            x_offset = (self.target_size - img_width) // 2
            padded_image[:, x_offset:x_offset + img_width] = image
            image = padded_image
            img_width = self.target_size
        
        # Если изображение ниже target_size
        if img_height <= self.target_size:
            return [(image, 0, x_offset)], x_offset
        
        # Расчет параметров скользящего окна
        max_step = img_height // self.target_size + 1
        max_step = max(max_step, 1)
        
        if max_step == 1:
            overlap = round(img_height / self.target_size - 1, 2) + 0.1
        else:
            overlap = round((max_step - round(img_height / self.target_size, 1)) / max_step, 2)
        
        step = int(self.target_size * (1 - overlap))
        
        # Создание окон
        for y in range(0, img_height, step):
            if y + self.target_size > img_height:
                y = max(0, img_height - self.target_size)
            
            window = image[y:y + self.target_size, 0:img_width]
            windows.append((window, y, x_offset))
            
            if y + self.target_size >= img_height:
                break
        
        return windows, x_offset
    
    def _process_window(self, window_img: np.ndarray,
                       window_y_offset: int,
                       window_x_padding: int,
                       bbox_original: List[float],
                       metadata: ImageMetadata,
                       original_width: int,
                       original_height: int) -> List[Dict[str, Any]]:
        """
        Обработка одного окна сегментации
        
        Returns:
            Список сегментированных строк в окне
        """
        segment_results = self.model(window_img, imgsz=self.target_size, conf=0.25, iou=0.5)
        
        if len(segment_results) == 0 or segment_results[0].masks is None:
            return []
        
        masks_resized = segment_results[0].masks.xy
        scores = segment_results[0].boxes.conf.cpu().numpy() if segment_results[0].boxes is not None \
                else [1.0] * len(masks_resized)
        
        segment_lines = []
        
        for mask, score in zip(masks_resized, scores):
            if len(mask) < 3 or score < 0.3:
                continue
            
            # Коррекция маски для окна
            adjusted_mask = self._adjust_mask_for_window(
                mask, window_y_offset, window_x_padding
            )
            
            # Исправление самопересечений
            fixed_mask = self._fix_self_intersections(adjusted_mask)
            
            if len(fixed_mask) < 3:
                continue
            
            # Конвертация в исходные координаты
            mask_original = self._convert_mask_to_original_coords(
                fixed_mask, bbox_original, metadata,
                original_width, original_height
            )
            
            if len(mask_original) >= 3:
                segment_lines.append({
                    "mask": [[float(x), float(y)] for x, y in mask_original],
                    "score": float(score)
                })
        
        return segment_lines
    
    def _adjust_mask_for_window(self, mask: np.ndarray,
                               window_y_offset: int,
                               window_x_padding: int) -> List[List[float]]:
        """
        Коррекция координат маски относительно окна
        """
        adjusted_mask = []
        for point in mask:
            x, y = point
            x_abs = x * self.target_size + window_x_padding
            y_abs = y * self.target_size + window_y_offset
            x_rel = x_abs / self.target_size
            y_rel = y_abs / self.target_size
            adjusted_mask.append([float(x_rel), float(y_rel)])
        return adjusted_mask
    
    def _convert_mask_to_original_coords(self, mask_resized: List[List[float]],
                                        bbox_original: List[float],
                                        metadata: ImageMetadata,
                                        original_width: int,
                                        original_height: int) -> List[List[float]]:
        """
        Конвертация маски в исходную систему координат
        """
        x1_bbox, y1_bbox, x2_bbox, y2_bbox = bbox_original
        mask_original = []
        
        for point in mask_resized:
            x_abs_res, y_abs_res = point
            
            # Обратное преобразование ресайза
            x_abs_bbox = (x_abs_res - metadata.x_offset) / metadata.scale
            y_abs_bbox = (y_abs_res - metadata.y_offset) / metadata.scale
            
            # Преобразование в координаты исходного изображения
            x_original = x1_bbox + x_abs_bbox
            y_original = y1_bbox + y_abs_bbox
            
            # Обрезка по границам корпуса
            x_original = max(x1_bbox, min(x2_bbox, x_original))
            y_original = max(y1_bbox, min(y2_bbox, y_original))
            
            mask_original.append([float(x_original), float(y_original)])
        
        return mask_original
    
    def _filter_segmentation_masks(self, segment_lines: List[Dict[str, Any]], 
                                 iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Фильтрация масок с помощью Non-Maximum Suppression
        """
        if not segment_lines:
            return []
        
        # Извлечение bounding boxes из масок
        bboxes = []
        for line in segment_lines:
            mask = np.array(line["mask"], dtype=np.float32)
            x_min, y_min = mask.min(axis=0)
            x_max, y_max = mask.max(axis=0)
            bboxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
        
        scores = [line["score"] for line in segment_lines]
        
        # Применение NMS
        keep_indices = self._non_max_suppression(bboxes, scores, iou_threshold)
        
        # Фильтрация и сортировка
        filtered_lines = [segment_lines[i] for i in keep_indices]
        filtered_lines.sort(key=lambda x: np.array(x["mask"], dtype=np.float32)[:, 1].min())
        
        # Нумерация строк
        for i, line in enumerate(filtered_lines):
            line["number_line"] = i + 1
        
        return filtered_lines
    
    def _non_max_suppression(self, boxes: List[List[float]], 
                           scores: List[float], 
                           iou_threshold: float) -> List[int]:
        """
        Non-Maximum Suppression
        """
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        indices = np.argsort(scores)[::-1]
        keep_indices = []
        
        while len(indices) > 0:
            current_idx = indices[0]
            keep_indices.append(current_idx)
            
            if len(indices) == 1:
                break
            
            current_box = boxes[current_idx]
            remaining_indices = indices[1:]
            remaining_boxes = boxes[remaining_indices]
            
            # Расчет IoU
            ious = np.array([self._calculate_iou(current_box, box) 
                           for box in remaining_boxes])
            
            indices = remaining_indices[ious < iou_threshold]
        
        return keep_indices
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Расчет Intersection over Union
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Расчет области пересечения
        x_min_inter = max(x1_min, x2_min)
        y_min_inter = max(y1_min, y2_min)
        x_max_inter = min(x1_max, x2_max)
        y_max_inter = min(y1_max, y2_max)
        
        inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
        
        # Расчет площадей боксов
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        # Расчет IoU
        iou = float(inter_area) / (area1 + area2 - inter_area + 1e-6)
        return iou
    
    def _fix_self_intersections(self, mask: List[List[float]]) -> List[List[float]]:
        """
        Исправление самопересечений полигона
        """
        if len(mask) < 3:
            return mask
        
        try:
            polygon = Polygon(mask)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
                if polygon.geom_type == 'MultiPolygon':
                    polygon = max(polygon.geoms, key=lambda p: p.area)
                mask = np.array(polygon.exterior.coords, dtype=np.float32)
                return [[float(x), float(y)] for x, y in mask]
            return mask
        except Exception:
            return mask
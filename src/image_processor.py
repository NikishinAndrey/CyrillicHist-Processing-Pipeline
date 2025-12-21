import cv2
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class ImageMetadata:
    """Метаданные изображения после ресайза"""
    scale: float
    x_offset: int
    y_offset: int
    new_width: int
    new_height: int

class ImageProcessor:
    """Класс для обработки изображений рукописей"""
    
    @staticmethod
    def resize_image_with_padding(img: np.ndarray, target_size: int = 1280) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Ресайз изображения с сохранением пропорций и добавлением паддинга
        
        Args:
            img: Входное изображение (H, W, C)
            target_size: Целевой размер (квадрат)
            
        Returns:
            Tuple[обработанное изображение, метаданные]
        """
        h, w = img.shape[:2]
        
        # Масштабирование с сохранением пропорций
        if h > w:
            scale = target_size / h
            new_h = target_size
            new_w = int(w * scale)
        else:
            scale = target_size / w
            new_w = target_size
            new_h = int(h * scale)
        
        # Ресайз
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Добавление паддинга
        padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
        
        metadata = ImageMetadata(
            scale=scale,
            x_offset=x_offset,
            y_offset=y_offset,
            new_width=new_w,
            new_height=new_h
        )
        
        return padded_img, metadata
    
    @staticmethod
    def calculate_threshold(gray_scale: np.ndarray, padding_percent: float = 0.10) -> int:
        """
        Расчет порога для бинаризации на основе гистограммы
        
        Args:
            gray_scale: Изображение в градациях серого
            padding_percent: Процент отсечения
            
        Returns:
            Пороговое значение
        """
        pixel_values = gray_scale.ravel()
        
        # Фильтрация пикселей в заданном диапазоне
        filtered_pixels = pixel_values[(pixel_values >= 80) & (pixel_values <= 210)]
        
        if len(filtered_pixels) == 0:
            return 127
        
        # Нахождение наиболее частого значения
        counts = np.bincount(filtered_pixels - 120)
        max_freq_relative_index = np.argmax(counts)
        pixel_with_max_freq = max_freq_relative_index + 120
        threshold = int(pixel_with_max_freq - padding_percent * pixel_with_max_freq)
        
        return threshold
    
    @staticmethod
    def binarize_line_image(line_img: np.ndarray) -> np.ndarray:
        """
        Бинаризация изображения строки текста
        
        Args:
            line_img: Изображение строки (RGB)
            
        Returns:
            Бинаризованное изображение
        """
        # Преобразование в оттенки серого
        gray_scale = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)
        
        # Расчет порога
        threshold = ImageProcessor.calculate_threshold(gray_scale)
        
        # Применение порога
        _, thresh_img = cv2.threshold(gray_scale, threshold, 255, cv2.THRESH_BINARY)
        
        # Морфологическое закрытие для устранения разрывов
        closed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, 
                                     np.ones((3, 3), np.uint8), iterations=1)
        
        return closed_img
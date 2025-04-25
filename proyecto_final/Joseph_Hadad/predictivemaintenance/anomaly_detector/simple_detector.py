# -*- coding: utf-8 -*-
import numpy as np
import json
import logging
import os
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("anomaly_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('simple_detector')

class SimpleAnomalyDetector:
    """Detector de anomalías simplificado para prueba real"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Parámetros de configuración
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.7)
        self.min_history_points = self.config.get('min_history_points', 5)
        
        # Directorio para almacenar datos
        self.data_dir = self.config.get('data_dir', './data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Historial de métricas por servicio
        self.metrics_history = {}
        
        # Umbrales por servicio
        self.thresholds = self.load_thresholds()
        
        logger.info(f"Detector de anomalías simple inicializado (umbral: {self.anomaly_threshold})")
        logger.info(f"Umbrales cargados: {self.thresholds}")
    
    def load_thresholds(self):
        """Carga umbrales desde archivo"""
        threshold_file = os.path.join(self.data_dir, 'thresholds.json')
        
        default_thresholds = {
            "default": {
                "memory_usage": 60,
                "cpu_usage": 70,
                "response_time_ms": 300,
                "error_rate": 5,
                "active_connections": 80,
                "query_time_avg": 100,
                "gc_collection_time": 400
            }
        }
        
        try:
            if os.path.exists(threshold_file):
                with open(threshold_file, 'r', encoding='utf-8-sig') as f:
                    thresholds = json.load(f)
                logger.info(f"Umbrales cargados desde {threshold_file}")
                return thresholds
            else:
                # Si no existe el archivo, crearlo con valores por defecto
                with open(threshold_file, 'w', encoding='utf-8') as f:
                    json.dump(default_thresholds, f, indent=2)
                logger.info(f"Archivo de umbrales creado en {threshold_file}")
                return default_thresholds
        except Exception as e:
            logger.error(f"Error al cargar umbrales: {str(e)}")
            return default_thresholds
    
    def save_thresholds(self):
        """Guarda umbrales en archivo"""
        threshold_file = os.path.join(self.data_dir, 'thresholds.json')
        
        try:
            with open(threshold_file, 'w', encoding='utf-8') as f:
                json.dump(self.thresholds, f, indent=2)
            logger.info(f"Umbrales guardados en {threshold_file}")
        except Exception as e:
            logger.error(f"Error al guardar umbrales: {str(e)}")
    
    def add_metric_point(self, service_id, metrics):
        """Añade un punto de métricas al historial"""
        try:
            if service_id not in self.metrics_history:
                self.metrics_history[service_id] = []
            
            # Filtrar solo campos numéricos
            numeric_metrics = {}
            for key, value in metrics.items():
                if key not in ['service_id', 'timestamp'] and isinstance(value, (int, float)):
                    numeric_metrics[key] = value
            
            # Añadir timestamp si no existe
            if 'timestamp' not in metrics:
                metrics['timestamp'] = datetime.now().isoformat()
            
            # Guardar punto completo (con valores no numéricos)
            self.metrics_history[service_id].append(metrics)
            
            # Limitar tamaño del historial
            max_history = self.config.get('max_history_points', 100)
            if len(self.metrics_history[service_id]) > max_history:
                self.metrics_history[service_id] = self.metrics_history[service_id][-max_history:]
                
            return True
            
        except Exception as e:
            logger.error(f"Error al añadir punto de métrica: {str(e)}")
            return False
    
    def detect_anomalies_direct(self, data_point):
        """
        Versión directa y simplificada de detección de anomalías 
        basada en umbrales directos
        """
        try:
            # Extraer información básica
            service_id = data_point.get('service_id', 'unknown')
            
            # Añadir punto al historial
            self.add_metric_point(service_id, data_point)
            
            # Obtener umbrales para este servicio (o usar default)
            service_thresholds = self.thresholds.get(service_id, self.thresholds.get('default', {}))
            
            # Variables para tracking de anomalías
            anomaly_metrics = []
            anomaly_details = {}
            max_severity = 0.0
            
            # Verificar cada métrica importante
            for metric, value in data_point.items():
                if metric not in ['service_id', 'timestamp'] and isinstance(value, (int, float)):
                    threshold = service_thresholds.get(metric)
                    
                    if threshold:
                        # Para hit_rate, valores bajos son problemáticos
                        if metric == 'hit_rate':
                            if value < threshold:
                                severity = min(1.0, (threshold - value) / threshold)
                                anomaly_metrics.append(f"{metric}: {value:.2f} < {threshold:.2f}")
                                anomaly_details[metric] = {
                                    'value': value,
                                    'threshold': threshold,
                                    'severity': severity
                                }
                                max_severity = max(max_severity, severity)
                        # Para las demás métricas, valores altos son problemáticos
                        else:
                            if value > threshold:
                                severity = min(1.0, (value - threshold) / threshold)
                                anomaly_metrics.append(f"{metric}: {value:.2f} > {threshold:.2f}")
                                anomaly_details[metric] = {
                                    'value': value,
                                    'threshold': threshold,
                                    'severity': severity
                                }
                                max_severity = max(max_severity, severity)
            
            # Calcular score general (normalizado entre 0 y 1)
            anomaly_score = min(1.0, max_severity)
            
            # Determinar si hay anomalía
            is_anomaly = anomaly_score >= self.anomaly_threshold
            
            # Log detallado para debug
            log_msg = f"Análisis para {service_id}: "
            if is_anomaly:
                log_msg += f"ANOMALÍA (score: {anomaly_score:.3f}, umbral: {self.anomaly_threshold})"
                logger.info(log_msg)
                logger.info(f"Métricas anómalas: {', '.join(anomaly_metrics)}")
            else:
                log_msg += f"normal (score: {anomaly_score:.3f}, umbral: {self.anomaly_threshold})"
                logger.debug(log_msg)
            
            # Generar detalles
            details = {
                "anomaly_type": "threshold_violation" if is_anomaly else "none",
                "metrics_analyzed": len(anomaly_details),
                "metrics_exceeded": len(anomaly_metrics),
                "thresholds": service_thresholds,
                "anomaly_details": anomaly_details
            }
            
            return is_anomaly, anomaly_score, details
            
        except Exception as e:
            logger.error(f"Error al detectar anomalías: {str(e)}")
            return False, 0.0, {"error": str(e)}
    
    def detect_anomalies(self, data_point):
        """
        Alias para el método principal de detección
        """
        return self.detect_anomalies_direct(data_point)
    
    def get_history(self, service_id, limit=20):
        """Obtiene historial de métricas para un servicio"""
        if service_id in self.metrics_history:
            return self.metrics_history[service_id][-limit:]
        return []
    
    def get_threshold(self, service_id, metric):
        """Obtiene umbral específico"""
        service_thresholds = self.thresholds.get(service_id, self.thresholds.get('default', {}))
        return service_thresholds.get(metric)
    
    def set_threshold(self, service_id, metric, value):
        """Establece umbral manualmente"""
        if service_id not in self.thresholds:
            self.thresholds[service_id] = self.thresholds.get('default', {}).copy()
        
        self.thresholds[service_id][metric] = value
        self.save_thresholds()
        
        return True

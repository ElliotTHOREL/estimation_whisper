#!/usr/bin/env python3
"""
Script de monitoring de la mémoire pendant l'évaluation des modèles

Ce script :
1. Surveille l'utilisation de la RAM en temps réel
2. Alerte en cas de dépassement de seuil
3. Enregistre les statistiques de consommation
4. Peut être lancé en parallèle de l'évaluation

Usage:
    python evaluation/code/memory_monitor.py --threshold 80 --log-interval 30
"""

import psutil
import time
import json
import sys
from pathlib import Path
from datetime import datetime
import argparse
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Moniteur de mémoire pour l'évaluation des modèles"""
    
    def __init__(self, threshold_percent: float = 85.0, log_interval: int = 30):
        self.threshold_percent = threshold_percent
        self.log_interval = log_interval
        self.memory_stats = []
        self.start_time = datetime.now()
        
        # Fichier de log des statistiques
        self.stats_file = Path("evaluation/benchmarks") / f"memory_stats_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
    
    def get_memory_info(self) -> dict:
        """Récupère les informations de mémoire actuelles"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_ram_gb": round(memory.total / (1024**3), 2),
            "available_ram_gb": round(memory.available / (1024**3), 2),
            "used_ram_gb": round(memory.used / (1024**3), 2),
            "ram_percent": memory.percent,
            "swap_used_gb": round(swap.used / (1024**3), 2),
            "swap_percent": swap.percent
        }
    
    def check_gpu_memory(self) -> dict:
        """Vérifie la mémoire GPU si disponible"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_stats()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_reserved()
                
                return {
                    "gpu_available": True,
                    "gpu_total_gb": round(total_memory / (1024**3), 2),
                    "gpu_allocated_gb": round(allocated / (1024**3), 2),
                    "gpu_cached_gb": round(cached / (1024**3), 2),
                    "gpu_percent": round((allocated / total_memory) * 100, 2)
                }
            else:
                return {"gpu_available": False}
        except ImportError:
            return {"gpu_available": False, "error": "PyTorch non disponible"}
    
    def log_memory_status(self, force_log: bool = False):
        """Enregistre le statut mémoire actuel"""
        memory_info = self.get_memory_info()
        gpu_info = self.check_gpu_memory()
        
        # Combinaison des informations
        full_info = {**memory_info, **gpu_info}
        self.memory_stats.append(full_info)
        
        # Alerte si dépassement de seuil
        if memory_info["ram_percent"] > self.threshold_percent or force_log:
            status = "🚨 ALERTE" if memory_info["ram_percent"] > self.threshold_percent else "📊 STATUS"
            
            logger.info(f"{status} - RAM: {memory_info['ram_percent']:.1f}% "
                       f"({memory_info['used_ram_gb']:.1f}GB/{memory_info['total_ram_gb']:.1f}GB)")
            
            if gpu_info.get("gpu_available"):
                logger.info(f"         GPU: {gpu_info['gpu_percent']:.1f}% "
                           f"({gpu_info['gpu_allocated_gb']:.1f}GB/{gpu_info['gpu_total_gb']:.1f}GB)")
        
        # Sauvegarde périodique des stats
        if len(self.memory_stats) % 10 == 0:
            self.save_stats()
    
    def save_stats(self):
        """Sauvegarde les statistiques dans un fichier JSON"""
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "monitoring_start": self.start_time.isoformat(),
                "threshold_percent": self.threshold_percent,
                "log_interval": self.log_interval,
                "stats": self.memory_stats
            }, f, indent=2, ensure_ascii=False)
    
    def run_monitoring(self, duration_minutes: int = None):
        """Lance le monitoring continu"""
        logger.info(f"🔍 Démarrage du monitoring mémoire (seuil: {self.threshold_percent}%)")
        logger.info(f"📝 Logs sauvegardés dans: {self.stats_file}")
        
        if duration_minutes:
            logger.info(f"⏱️ Durée prévue: {duration_minutes} minutes")
            end_time = time.time() + (duration_minutes * 60)
        else:
            logger.info("⏱️ Monitoring continu (Ctrl+C pour arrêter)")
            end_time = None
        
        try:
            while True:
                self.log_memory_status()
                time.sleep(self.log_interval)
                
                if end_time and time.time() > end_time:
                    break
                    
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt du monitoring demandé")
        
        finally:
            # Sauvegarde finale
            self.save_stats()
            self.generate_summary()
    
    def generate_summary(self):
        """Génère un résumé des statistiques de monitoring"""
        if not self.memory_stats:
            return
        
        ram_usages = [stat["ram_percent"] for stat in self.memory_stats]
        
        summary = {
            "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
            "total_measurements": len(self.memory_stats),
            "ram_usage": {
                "min_percent": min(ram_usages),
                "max_percent": max(ram_usages),
                "avg_percent": sum(ram_usages) / len(ram_usages),
                "threshold_exceeded_count": sum(1 for usage in ram_usages if usage > self.threshold_percent)
            }
        }
        
        # Statistiques GPU si disponibles
        gpu_usages = [stat.get("gpu_percent", 0) for stat in self.memory_stats if stat.get("gpu_available")]
        if gpu_usages:
            summary["gpu_usage"] = {
                "min_percent": min(gpu_usages),
                "max_percent": max(gpu_usages),
                "avg_percent": sum(gpu_usages) / len(gpu_usages)
            }
        
        logger.info("\n" + "="*50)
        logger.info("📊 RÉSUMÉ DU MONITORING MÉMOIRE")
        logger.info("="*50)
        logger.info(f"⏱️ Durée: {summary['duration_minutes']:.1f} minutes")
        logger.info(f"📊 Mesures: {summary['total_measurements']}")
        logger.info(f"💾 RAM min/max/moy: {summary['ram_usage']['min_percent']:.1f}% / "
                   f"{summary['ram_usage']['max_percent']:.1f}% / "
                   f"{summary['ram_usage']['avg_percent']:.1f}%")
        logger.info(f"🚨 Dépassements seuil: {summary['ram_usage']['threshold_exceeded_count']}")
        
        if "gpu_usage" in summary:
            logger.info(f"🎮 GPU min/max/moy: {summary['gpu_usage']['min_percent']:.1f}% / "
                       f"{summary['gpu_usage']['max_percent']:.1f}% / "
                       f"{summary['gpu_usage']['avg_percent']:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Monitoring de la mémoire pendant l'évaluation")
    parser.add_argument('--threshold', type=float, default=85.0,
                       help="Seuil d'alerte RAM en pourcentage (défaut: 85%%)")
    parser.add_argument('--log-interval', type=int, default=30,
                       help="Intervalle de logging en secondes (défaut: 30s)")
    parser.add_argument('--duration', type=int,
                       help="Durée du monitoring en minutes (défaut: continu)")
    parser.add_argument('--output-file', type=str,
                       help="Fichier de sortie personnalisé pour les stats")
    
    args = parser.parse_args()
    
    monitor = MemoryMonitor(
        threshold_percent=args.threshold,
        log_interval=args.log_interval
    )
    
    if args.output_file:
        monitor.stats_file = Path(args.output_file)
    
    # Log initial
    monitor.log_memory_status(force_log=True)
    
    # Lancement du monitoring
    monitor.run_monitoring(duration_minutes=args.duration)

if __name__ == "__main__":
    main() 
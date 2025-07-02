import re
import string
from typing import List, Dict, Union, Tuple
import difflib
from collections import Counter
import numpy as np
import pandas as pd
from jiwer import wer, cer, mer, wil, measures
import sacrebleu
from rouge_score import rouge_scorer
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechRecognitionMetrics:
    """
    Classe pour calculer diverses métriques d'évaluation pour la reconnaissance vocale
    """
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalise le texte pour les calculs de métriques
        
        Args:
            text: Texte à normaliser
            
        Returns:
            Texte normalisé
        """
        # Conversion en minuscules
        text = text.lower()
        
        # Remplacement de toutes les apostrophes (simples, typographiques, etc.) par un espace
        text = re.sub(r"[''`´]", " ", text)
        
        # Suppression de la ponctuation (hors apostrophes déjà traitées)
        text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
        
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Suppression des espaces en début/fin
        text = text.strip()
        
        return text
    
    def word_error_rate(self, reference: str, hypothesis: str, normalize: bool = True) -> float:
        """
        Calcule le Word Error Rate (WER)
        
        Args:
            reference: Texte de référence (vérité terrain)
            hypothesis: Texte prédit par le modèle
            normalize: Si True, normalise les textes avant calcul
            
        Returns:
            WER en pourcentage
        """
        if normalize:
            reference = self.normalize_text(reference)
            hypothesis = self.normalize_text(hypothesis)
        
        return wer(reference, hypothesis) * 100
    
    def character_error_rate(self, reference: str, hypothesis: str, normalize: bool = True) -> float:
        """
        Calcule le Character Error Rate (CER)
        
        Args:
            reference: Texte de référence
            hypothesis: Texte prédit
            normalize: Si True, normalise les textes
            
        Returns:
            CER en pourcentage
        """
        if normalize:
            reference = self.normalize_text(reference)
            hypothesis = self.normalize_text(hypothesis)
        
        return cer(reference, hypothesis) * 100
    
    def match_error_rate(self, reference: str, hypothesis: str, normalize: bool = True) -> float:
        """
        Calcule le Match Error Rate (MER)
        
        Args:
            reference: Texte de référence
            hypothesis: Texte prédit
            normalize: Si True, normalise les textes
            
        Returns:
            MER en pourcentage
        """
        if normalize:
            reference = self.normalize_text(reference)
            hypothesis = self.normalize_text(hypothesis)
        
        return mer(reference, hypothesis) * 100
    
    def word_information_lost(self, reference: str, hypothesis: str, normalize: bool = True) -> float:
        """
        Calcule le Word Information Lost (WIL)
        
        Args:
            reference: Texte de référence
            hypothesis: Texte prédit
            normalize: Si True, normalise les textes
            
        Returns:
            WIL en pourcentage
        """
        if normalize:
            reference = self.normalize_text(reference)
            hypothesis = self.normalize_text(hypothesis)
        
        return wil(reference, hypothesis) * 100
    
    def bleu_score(self, reference: str, hypothesis: str, normalize: bool = True) -> Dict[str, float]:
        """
        Calcule le score BLEU
        
        Args:
            reference: Texte de référence
            hypothesis: Texte prédit
            normalize: Si True, normalise les textes
            
        Returns:
            Dictionnaire avec les scores BLEU
        """
        if normalize:
            reference = self.normalize_text(reference)
            hypothesis = self.normalize_text(hypothesis)
        
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
        
        return {
            "bleu": bleu.score,
            "bleu_1": bleu.precisions[0],
            "bleu_2": bleu.precisions[1] if len(bleu.precisions) > 1 else 0,
            "bleu_3": bleu.precisions[2] if len(bleu.precisions) > 2 else 0,
            "bleu_4": bleu.precisions[3] if len(bleu.precisions) > 3 else 0
        }
    
    def rouge_scores(self, reference: str, hypothesis: str, normalize: bool = True) -> Dict[str, float]:
        """
        Calcule les scores ROUGE
        
        Args:
            reference: Texte de référence
            hypothesis: Texte prédit
            normalize: Si True, normalise les textes
            
        Returns:
            Dictionnaire avec les scores ROUGE
        """
        if normalize:
            reference = self.normalize_text(reference)
            hypothesis = self.normalize_text(hypothesis)
        
        scores = self.rouge_scorer.score(reference, hypothesis)
        
        return {
            "rouge1_f": scores['rouge1'].fmeasure,
            "rouge1_p": scores['rouge1'].precision,
            "rouge1_r": scores['rouge1'].recall,
            "rouge2_f": scores['rouge2'].fmeasure,
            "rouge2_p": scores['rouge2'].precision,
            "rouge2_r": scores['rouge2'].recall,
            "rougeL_f": scores['rougeL'].fmeasure,
            "rougeL_p": scores['rougeL'].precision,
            "rougeL_r": scores['rougeL'].recall
        }
    
    def detailed_wer_analysis(self, reference: str, hypothesis: str, normalize: bool = True) -> Dict:
        """
        Analyse détaillée du WER avec comptage des types d'erreurs
        
        Args:
            reference: Texte de référence
            hypothesis: Texte prédit
            normalize: Si True, normalise les textes
            
        Returns:
            Dictionnaire avec l'analyse détaillée
        """
        if normalize:
            reference = self.normalize_text(reference)
            hypothesis = self.normalize_text(hypothesis)
        
        # Utilisation directe des fonctions jiwer
        wer_score = wer(reference, hypothesis) * 100
        cer_score = cer(reference, hypothesis) * 100
        mer_score = mer(reference, hypothesis) * 100
        wil_score = wil(reference, hypothesis) * 100
        
        return {
            "wer": wer_score,
            "mer": mer_score,
            "wil": wil_score,
            "cer": cer_score,
            "ref_len": len(reference.split()),
            "hyp_len": len(hypothesis.split())
        }
    
    def compute_all_metrics(self, reference: str, hypothesis: str, normalize: bool = True) -> Dict:
        """
        Calcule toutes les métriques disponibles
        
        Args:
            reference: Texte de référence
            hypothesis: Texte prédit
            normalize: Si True, normalise les textes
            
        Returns:
            Dictionnaire avec toutes les métriques
        """
        metrics = {}
        
        # Métriques de base
        metrics["wer"] = self.word_error_rate(reference, hypothesis, normalize)
        metrics["cer"] = self.character_error_rate(reference, hypothesis, normalize)
        metrics["mer"] = self.match_error_rate(reference, hypothesis, normalize)
        metrics["wil"] = self.word_information_lost(reference, hypothesis, normalize)
        
        # Scores BLEU
        bleu_scores = self.bleu_score(reference, hypothesis, normalize)
        metrics.update(bleu_scores)
        
        # Scores ROUGE
        rouge_scores = self.rouge_scores(reference, hypothesis, normalize)
        metrics.update(rouge_scores)
        
        # Analyse détaillée WER
        detailed_wer = self.detailed_wer_analysis(reference, hypothesis, normalize)
        metrics["detailed_wer"] = detailed_wer
        
        return metrics


class ModelEvaluator:
    """
    Classe pour évaluer les performances de modèles de reconnaissance vocale
    """
    
    def __init__(self):
        self.metrics_calculator = SpeechRecognitionMetrics()
    
    def evaluate_single_prediction(self, reference: str, hypothesis: str, 
                                 model_name: str = None, audio_file: str = None) -> Dict:
        """
        Évalue une seule prédiction
        
        Args:
            reference: Texte de référence
            hypothesis: Texte prédit
            model_name: Nom du modèle (optionnel)
            audio_file: Nom du fichier audio (optionnel)
            
        Returns:
            Dictionnaire avec les résultats d'évaluation
        """
        metrics = self.metrics_calculator.compute_all_metrics(reference, hypothesis)
        
        result = {
            "model_name": model_name,
            "audio_file": audio_file,
            "reference": reference,
            "hypothesis": hypothesis,
            "metrics": metrics
        }
        
        return result
    
    def evaluate_dataset(self, predictions: List[Dict]) -> Dict:
        """
        Évalue un dataset complet
        
        Args:
            predictions: Liste de dictionnaires avec 'reference' et 'hypothesis'
                        Peut aussi contenir 'model_name', 'audio_file', etc.
        
        Returns:
            Dictionnaire avec les statistiques d'évaluation
        """
        all_metrics = []
        
        for pred in predictions:
            reference = pred["reference"]
            hypothesis = pred["hypothesis"]
            
            metrics = self.metrics_calculator.compute_all_metrics(reference, hypothesis)
            metrics.update({
                "model_name": pred.get("model_name"),
                "audio_file": pred.get("audio_file")
            })
            
            all_metrics.append(metrics)
        
        # Calcul des statistiques
        df = pd.DataFrame(all_metrics)
        
        # Métriques moyennes
        avg_metrics = {
            "avg_wer": df["wer"].mean(),
            "avg_cer": df["cer"].mean(),
            "avg_mer": df["mer"].mean(),
            "avg_wil": df["wil"].mean(),
            "avg_bleu": df["bleu"].mean(),
            "avg_rouge1_f": df["rouge1_f"].mean(),
            "avg_rouge2_f": df["rouge2_f"].mean(),
            "avg_rougeL_f": df["rougeL_f"].mean()
        }
        
        # Statistiques descriptives
        stats = {
            "total_samples": len(predictions),
            "metrics": avg_metrics,
            "detailed_stats": {
                "wer": {
                    "mean": df["wer"].mean(),
                    "std": df["wer"].std(),
                    "min": df["wer"].min(),
                    "max": df["wer"].max(),
                    "median": df["wer"].median()
                },
                "cer": {
                    "mean": df["cer"].mean(),
                    "std": df["cer"].std(),
                    "min": df["cer"].min(),
                    "max": df["cer"].max(),
                    "median": df["cer"].median()
                }
            },
            "raw_results": all_metrics
        }
        
        return stats
    
    def compare_models(self, model_predictions: Dict[str, List[Dict]]) -> Dict:
        """
        Compare plusieurs modèles
        
        Args:
            model_predictions: Dictionnaire {model_name: [predictions]}
            
        Returns:
            Dictionnaire avec la comparaison des modèles
        """
        model_stats = {}
        
        for model_name, predictions in model_predictions.items():
            logger.info(f"Évaluation du modèle: {model_name}")
            stats = self.evaluate_dataset(predictions)
            stats["model_name"] = model_name
            model_stats[model_name] = stats
        
        # Création d'un tableau de comparaison
        comparison_df = pd.DataFrame({
            model_name: stats["metrics"] 
            for model_name, stats in model_stats.items()
        }).T
        
        # Tri par WER (meilleur modèle = WER plus faible)
        comparison_df = comparison_df.sort_values("avg_wer")
        
        return {
            "individual_results": model_stats,
            "comparison_table": comparison_df.to_dict(),
            "best_model": comparison_df.index[0],
            "ranking": list(comparison_df.index)
        }
    
    def save_results(self, results: Dict, output_file: str) -> None:
        """
        Sauvegarde les résultats dans un fichier CSV
        
        Args:
            results: Résultats d'évaluation
            output_file: Chemin du fichier de sortie
        """
        if "raw_results" in results:
            df = pd.DataFrame(results["raw_results"])
            df.to_csv(output_file, index=False)
            logger.info(f"Résultats sauvegardés dans {output_file}")
        elif "comparison_table" in results:
            df = pd.DataFrame(results["comparison_table"]).T
            df.to_csv(output_file)
            logger.info(f"Comparaison sauvegardée dans {output_file}")


if __name__ == "__main__":
    # Exemple d'utilisation
    metrics = SpeechRecognitionMetrics()
    
    # Test avec des exemples
    reference = "Bonjour, comment allez-vous aujourd'hui?"
    hypothesis = "Bonjour, comment aller vous aujourd'hui?"
    
    print("Exemple d'évaluation:")
    print(f"Référence: {reference}")
    print(f"Hypothèse: {hypothesis}")
    
    all_metrics = metrics.compute_all_metrics(reference, hypothesis)
    print(f"WER: {all_metrics['wer']:.2f}%")
    print(f"CER: {all_metrics['cer']:.2f}%")
    print(f"BLEU: {all_metrics['bleu']:.2f}")
    print(f"ROUGE-L F1: {all_metrics['rougeL_f']:.3f}")


def calculate_metrics(references: List[str], predictions: List[str], normalize: bool = True) -> Dict:
    """
    Fonction utilitaire pour calculer les métriques sur une liste de prédictions
    
    Args:
        references: Liste des textes de référence
        predictions: Liste des prédictions
        normalize: Si True, normalise les textes
        
    Returns:
        Dictionnaire avec les métriques moyennes
    """
    if len(references) != len(predictions):
        raise ValueError("Les listes de références et prédictions doivent avoir la même taille")
    
    metrics_calculator = SpeechRecognitionMetrics()
    all_metrics = []
    
    for ref, pred in zip(references, predictions):
        metrics = metrics_calculator.compute_all_metrics(ref, pred, normalize=normalize)
        all_metrics.append(metrics)
    
    # Calcul des moyennes
    avg_metrics = {}
    if all_metrics:
        keys = all_metrics[0].keys()
        for key in keys:
            if key != 'detailed_wer':  # Exclure les données complexes
                values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
                if values:
                    avg_metrics[key] = sum(values) / len(values)
    
    return avg_metrics 
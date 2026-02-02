"""Metrics tracking and evaluation utilities."""

import json
import os
from typing import Dict, List
from datetime import datetime


class MetricsLogger:
    """Tracks and logs training metrics."""
    
    def __init__(self, log_dir: str = "logs/"):
        """Initialize metrics logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = {
            "rounds": [],
            "loss": [],
            "accuracy": [],
            "client_times": [],
            "server_times": [],
            "total_times": []
        }
        
        self.start_time = datetime.now()
    
    def log_round(self, 
                  round_num: int,
                  loss: float,
                  accuracy: float,
                  client_time: float = 0.0,
                  server_time: float = 0.0,
                  total_time: float = 0.0):
        """Log metrics for a training round.
        
        Args:
            round_num: Federation round number
            loss: Training loss
            accuracy: Classification accuracy
            client_time: Client-side computation time
            server_time: Server-side computation time
            total_time: Total round time
        """
        self.metrics["rounds"].append(round_num)
        self.metrics["loss"].append(float(loss))
        self.metrics["accuracy"].append(float(accuracy))
        self.metrics["client_times"].append(float(client_time))
        self.metrics["server_times"].append(float(server_time))
        self.metrics["total_times"].append(float(total_time))
        
        print(f"Round {round_num}: Loss={loss:.4f}, Acc={accuracy:.4f}, "
              f"Time={total_time:.2f}s")
    
    def save(self, filename: str = "metrics.json"):
        """Save metrics to JSON file.
        
        Args:
            filename: Output filename
        """
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics.
        
        Returns:
            Dictionary with summary metrics
        """
        if not self.metrics["accuracy"]:
            return {}
        
        return {
            "final_accuracy": self.metrics["accuracy"][-1],
            "max_accuracy": max(self.metrics["accuracy"]),
            "final_loss": self.metrics["loss"][-1],
            "min_loss": min(self.metrics["loss"]),
            "avg_round_time": sum(self.metrics["total_times"]) / len(self.metrics["total_times"]),
            "total_training_time": (datetime.now() - self.start_time).total_seconds()
        }
    
    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()
        if not summary:
            print("No metrics recorded.")
            return
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Final Accuracy: {summary['final_accuracy']:.4f}")
        print(f"Max Accuracy: {summary['max_accuracy']:.4f}")
        print(f"Final Loss: {summary['final_loss']:.4f}")
        print(f"Min Loss: {summary['min_loss']:.4f}")
        print(f"Average Round Time: {summary['avg_round_time']:.2f}s")
        print(f"Total Training Time: {summary['total_training_time']:.2f}s")
        print("="*50 + "\n")

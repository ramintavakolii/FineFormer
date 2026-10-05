import os
import torch
import argparse
import matplotlib.pyplot as plt

class CVResultsVisualizer:
    def __init__(self, result_path, summary_name):
        """Load and visualize cross-validation results"""
        file_path = os.path.join(result_path, summary_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Summary file not found at: {file_path}")
            
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        self.histories = data['training_histories']
        self.metrics = data['metrics_summary']

    def _setup_subplot(self, ax, title, xlabel, ylabel):
        """Configure subplot appearance"""
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_metric(self, ax, histories, metric_key, fold, **kwargs):
        """Plot a single metric for one fold"""
        if metric_key in histories:
            data = histories[metric_key]
            if metric_key == 'val_loss':
                data = [x for x in data if x is not None]
            if data:
                ax.plot(data, label=f'Fold {fold}', alpha=0.8, **kwargs)

    def plot_training_grid(self, save_path=None):
        """Plot 2x2 grid of training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        for fold, hist in enumerate(self.histories, 1):
            self._plot_metric(axes[0, 0], hist, 'train_loss', fold)
            self._plot_metric(axes[0, 1], hist, 'train_acc', fold)
            self._plot_metric(axes[1, 0], hist, 'val_loss', fold)
            self._plot_metric(axes[1, 1], hist, 'val_acc', fold)

        titles = ['Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy']
        positions = [(0,0), (0,1), (1,0), (1,1)]

        for (i, j), title in zip(positions, titles):
            self._setup_subplot(axes[i, j], title, 'Epoch', title.split()[1])

        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'training_curves.png'))
            print(f"Saved training curves to {save_path}/training_curves.png")
        plt.show()


    def plot_bars(self, metrics=['accuracy', 'recall'], save_path=None):
        """Plot bar charts for specified metrics"""
        available_metrics = [m for m in metrics if m in self.metrics]

        if not available_metrics:
            return

        fig, axes = plt.subplots(1, len(available_metrics), figsize=(7*len(available_metrics), 5))
        if len(available_metrics) == 1:
            axes = [axes]

        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']

        for i, metric in enumerate(available_metrics):
            data = self.metrics[metric]
            scores, mean_val, std_val = data['scores'], data['mean'], data['std']

            bars = axes[i].bar(range(1, len(scores)+1), scores,
                             color=colors[i % len(colors)], alpha=0.7)
            axes[i].axhline(mean_val, color='red', linestyle='--',
                          label=f'Mean = {mean_val:.3f} ± {std_val:.3f}')

            # Add value labels
            for j, bar in enumerate(bars):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{scores[j]:.3f}', ha='center', va='bottom', fontsize=9)

            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_xlabel('Fold')
            axes[i].set_ylabel('Score')
            axes[i].legend()
            axes[i].grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'bar_charts.png'))
            print(f"Saved bar charts to {save_path}/bar_charts.png")
        plt.show()

    def print_summary(self):
        """Print results summary"""
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS SUMMARY")
        print("="*60)

        for metric, data in self.metrics.items():
            mean_val, std_val = data['mean'], data['std']
            print(f"{metric.upper():20s}: {mean_val:.4f} ± {std_val:.4f}")

    def visualize_all(self, save_path=None):
        """Generate all visualizations"""
        print("📊 Generating visualizations...")
        self.plot_training_grid(save_path)
        self.plot_bars(save_path=save_path)
        self.print_summary()
        print("✅ Complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize CV Results")
    parser.add_argument("--result-path", required=True, help="Path to the directory containing the .pt summary file")
    parser.add_argument("--summary-name", required=True, help="Name of the .pt summary file (e.g., cv_summary_healthy_vs_schizo.pt)")
    parser.add_argument("--save", action="store_true", help="Save the plots as PNG files in the result directory")
    args = parser.parse_args()

    visualizer = CVResultsVisualizer(args.result_path, args.summary_name)
    
    # Pass the result_path as the save_path if the user uses the --save flag
    save_path = args.result_path if args.save else None
    visualizer.visualize_all(save_path=save_path)
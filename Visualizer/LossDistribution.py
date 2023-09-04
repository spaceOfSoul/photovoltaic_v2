import numpy as np
import matplotlib.pyplot as plt
import logging
import os

class LossStatistics:
    def __init__(self, losses, val_losses):
        self.losses = losses
        self.val_losses = val_losses

    def process(self, save_dir):
        # Calculate statistics
        losses_mean = np.mean(self.losses)
        losses_var = np.var(self.losses)
        losses_max = np.max(self.losses)
        val_losses_mean = np.mean(self.val_losses)
        val_losses_var = np.var(self.val_losses)
        val_losses_max = np.max(self.val_losses)

        # Print statistics
        logging.info("\nTraining losses: Mean = {:.4f} [(kW/hour)^2], Variance = {:.4f}, Max = {:.4f}".format(losses_mean, losses_var, losses_max))
        logging.info("Validation losses: Mean = {:.4f} [(kW/hour)^2], Variance = {:.4f}, Max = {:.4f}".format(val_losses_mean, val_losses_var, val_losses_max))

        # Plot distributions
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Training loss
        axes[0].hist(self.losses, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
        axes[0].set_title('Training Loss Distribution')
        axes[0].set_xlabel('Loss')
        axes[0].set_ylabel('frequency')
        axes[0].set_xlim(left=0)
        stat_text = 'mean: {:.2f}\nVar: {:.2f}'.format(losses_mean, losses_var)
        axes[0].text(0.95, 0.95, stat_text, transform=axes[0].transAxes, ha='right', va='top', fontsize=10)
        
        # Validation loss
        axes[1].hist(self.val_losses, bins='auto', color='red', alpha=0.7, rwidth=0.85)
        axes[1].set_title('Validation Loss Distribution')
        axes[1].set_xlabel('Loss')
        axes[1].set_ylabel('frequency')
        axes[1].set_xlim(left=0)
        stat_text = 'mean: {:.2f}\nVar: {:.2f}'.format(val_losses_mean, val_losses_var)
        axes[1].text(0.95, 0.95, stat_text, transform=axes[1].transAxes, ha='right', va='top', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "loss_distribution.png"))
        plt.close()


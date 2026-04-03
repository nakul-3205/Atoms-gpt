import os
import torch
from src.logger import logger
from src.components.model_factory import AtomsGPT


class TrainPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.epochs = config['training']['epochs']
        self.lr = config['training']['learning_rate']
        self.grad_clip = config['training']['grad_clip']
        self.ckpt_dir = config['paths']['checkpoint_dir']
        self.model_dir = config['paths']['model_dir']
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.device = torch.device("cpu")  # Intel CPU — reliable & stable
        logger.info(f"Training on device: {self.device}")

    def run(self, train_loader, val_loader):
        model = AtomsGPT(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        best_val_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            # ── Train ──
            model.train()
            train_loss = 0.0
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
                train_loss += loss.item()

                if batch_idx % 100 == 0:
                    logger.info(
                        f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} "
                        f"| Loss: {loss.item():.4f}"
                    )

            avg_train = train_loss / len(train_loader)

            # ── Validate ──
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    _, loss = model(x, y)
                    val_loss += loss.item()
            avg_val = val_loss / len(val_loader)

            scheduler.step()
            logger.info(
                f"=== Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} ==="
            )

            # ── Checkpoint every epoch ──
            ckpt_path = os.path.join(self.ckpt_dir, f"atoms_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': avg_val,
                'config': self.config
            }, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

            # ── Save best model ──
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_path = os.path.join(self.model_dir, "atoms_best.pth")
                torch.save(model.state_dict(), best_path)
                logger.info(f"✅ New best model saved! Val Loss: {best_val_loss:.4f}")

        logger.info("=== Training Complete ===")
        return model
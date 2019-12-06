from config import Configuration
from train import GeneralTrainer

trainer = GeneralTrainer(Configuration())
trainer.train()

import logging

from config import ArteryVeinConfiguration
from train import VeinArteryTrainer

logging.basicConfig(format='%(levelname)s: [%(asctime)s] [%(name)s:%(lineno)d-%(funcName)20s()] %(message)s',
                    level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S')

trainer = VeinArteryTrainer(ArteryVeinConfiguration())
trainer.train()

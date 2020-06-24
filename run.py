import logging

from config import AVRConfiguration, SegmentationConfiguration
from train import VeinArteryTrainer, SegmentationTrainer

logging.basicConfig(format='%(levelname)s: [%(asctime)s] [%(name)s:%(lineno)d-%(funcName)20s()] %(message)s',
                    level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S')

trainer = SegmentationTrainer(SegmentationConfiguration())
trainer.train()

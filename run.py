import logging

from config import Configuration
from train import RegistrationTrainer

logging.basicConfig(format='%(levelname)s: [%(asctime)s] [%(name)s:%(lineno)d-%(funcName)20s()] %(message)s',
                    level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S')

trainer = RegistrationTrainer(Configuration())
trainer.train()

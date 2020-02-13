from config import Configuration, VesselConfiguration
from train import GeneralTrainer, VesselTrainer

trainer = VesselTrainer(VesselConfiguration())
trainer.train()

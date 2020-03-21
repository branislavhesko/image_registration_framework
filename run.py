from config import ArteryVeinConfiguration, Configuration, VesselConfiguration
from train import VeinArteryTrainer, GeneralTrainer, VesselTrainer

trainer = VeinArteryTrainer(ArteryVeinConfiguration())
trainer.train()

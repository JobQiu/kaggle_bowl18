from bowl_config import BowlConfig

class InferenceConfig(BowlConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

class InferenceConfig101(BowlConfig):
    GPU_COUNT = 1
    RESNET_ARCHITECTURE = 'resnet101'
    IMAGES_PER_GPU = 1
inference_config101 = InferenceConfig101()
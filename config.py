# Configuration for CUDA, model paths, and thresholds

CUDA_VISIBLE_DEVICES = "0"

MODEL_PATHS = {
    'yolo': 'models/yolo_weights.weights',
    'resnet': 'models/resnet_weights.pth'
}

THRESHOLDS = {
    'detection': 0.5,
    'embedding': 0.6
}
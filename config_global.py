from dataclasses import dataclass, field
from typing import List, Tuple
import torch

@dataclass
class GlobalConfig:
    """모든 실험이 공유하는 기본 설정"""
    
    # 1. 환경 설정
    DEVICE: torch.device = field(default_factory=lambda: 
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # 2. 학습 파라미터
    NUM_EPOCHS: int = 30
    BATCH_SIZE: int = 64
    LOG_INTERVAL: int = 100

    # 3. 데이터셋 파라미터
    SHUFFLE_SIZE: int = 3000

    # 4. 데이터 경로 (Train/Test 분리)
    # TRAIN_DIFFFACE_SHARDS: List[str] = field(default_factory=lambda: [
    #     "/local_datasets/deepfake-detection/data/diffusion_face/ADM-{000..004}.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/DDIM-{000..004}.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/DDPM-{000..004}.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/DiffSwap-{000..004}.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/Inpaint-{000..004}.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/LDM-{000..004}.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/PNDM-{000..004}.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/SDv15_DS0.3-{000..004}.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/SDv15_DS0.5-{000..004}.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/SDv15_DS0.7-{000..004}.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/SDv21_DS0.3-{000..004}.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/SDv21_DS0.5-{000..004}.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/SDv21_DS0.7-{000..004}.tar",
    # ])
    # # TRAIN_CELEBA_SHARDS: List[str] = field(default_factory=lambda: [
    # #     "/local_datasets/deepfake-detection/data/mm_celeba_hq/mm_celeba_hq-{000..004}.tar"
    # # ])



    TRAIN_AIFaceDataset3000_SHARDS: List[str] = field(default_factory=lambda: [
        "/local_datasets/deepfake-detection/data/AIFaceDataset3000/AIFaceDataset3000-{000000..000028}.tar",
    ])
    TRAIN_CELEBA2_SHARDS: List[str] = field(default_factory=lambda: [
        "/local_datasets/deepfake-detection/data/CelebA/CelebA-{000000..002024}.tar"
    ])

    TRAIN_FFHQ_SHARDS: List[str] = field(default_factory=lambda: [
        "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_1-{000000..000048}.tar",
        "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_2-{000000..000048}.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_3-{000000..000048}.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_4-{000000..000048}.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_5-{000000..000048}.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_6-{000000..000048}.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_7-{000000..000048}.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_8-{000000..000048}.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_9-{000000..000048}.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_10-{000000..000048}.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_11-{000000..000048}.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_12-{000000..000048}.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_13-{000000..000048}.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_14-{000000..000048}.tar",
    ])

    TRAIN_SFHQ_SHARDS: List[str] = field(default_factory=lambda: [
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_1A/SFHQ_1A-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_1B/SFHQ_1B-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_1C/SFHQ_1C-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_1D/SFHQ_1D-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_2A/SFHQ_2A-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_2B/SFHQ_2B-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_2C/SFHQ_2C-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_2D/SFHQ_2D-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_3A/SFHQ_3A-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_3B/SFHQ_3B-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_3C/SFHQ_3C-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_3D/SFHQ_3D-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_4A/SFHQ_4A-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_4B/SFHQ_4B-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_4C/SFHQ_4C-{000000..00022}.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_4D/SFHQ_4D-{000000..00022}.tar",
    ])




    # TEST_DIFFFACE_SHARDS: List[str] = field(default_factory=lambda: [
    #     "/local_datasets/deepfake-detection/data/diffusion_face/ADM-005.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/DDIM-005.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/DDPM-005.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/DiffSwap-005.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/Inpaint-005.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/LDM-005.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/PNDM-005.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/SDv15_DS0.3-005.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/SDv15_DS0.5-005.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/SDv15_DS0.7-005.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/SDv21_DS0.3-005.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/SDv21_DS0.5-005.tar",
    #     "/local_datasets/deepfake-detection/data/diffusion_face/SDv21_DS0.7-005.tar",
    # ])
    # TEST_CELEBA_SHARDS: List[str] = field(default_factory=lambda: [
    #     "/local_datasets/deepfake-detection/data/mm_celeba_hq/mm_celeba_hq-005.tar"
    # ])

    TEST_AIFaceDataset3000_SHARDS: List[str] = field(default_factory=lambda: [
        "/local_datasets/deepfake-detection/data/AIFaceDataset3000/AIFaceDataset3000-000029.tar",
    ])
    TEST_CELEBA2_SHARDS: List[str] = field(default_factory=lambda: [
        "/local_datasets/deepfake-detection/data/CelebA/CelebA-002025.tar"
    ])

    TEST_FFHQ_SHARDS: List[str] = field(default_factory=lambda: [
        "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_1-000049.tar",
        "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_2-000049.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_3-000049.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_4-000049.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_5-000049.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_6-000049.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_7-000049.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_8-000049.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_9-000049.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_10-000049.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_11-000049.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_12-000049.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_13-000049.tar",
        # "/local_datasets/deepfake-detection/data/FFHQ/FFHQ_14-000049.tar",
    ])

    TEST_SFHQ_SHARDS: List[str] = field(default_factory=lambda: [
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_1A/SFHQ_1A-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_1B/SFHQ_1B-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_1C/SFHQ_1C-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_1D/SFHQ_1D-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_2A/SFHQ_2A-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_2B/SFHQ_2B-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_2C/SFHQ_2C-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_2D/SFHQ_2D-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_3A/SFHQ_3A-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_3B/SFHQ_3B-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_3C/SFHQ_3C-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_3D/SFHQ_3D-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_4A/SFHQ_4A-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_4B/SFHQ_4B-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_4C/SFHQ_4C-000023.tar",
        "/local_datasets/deepfake-detection/data/SFHQ/SFHQ_4D/SFHQ_4D-000023.tar",
    ])




    # data_loader.py가 필요로 하는 값들
    IMG_SIZE: int = 256  # (기본값을 설정하거나 None으로 설정)
    MEAN: List[float] = None
    STD: List[float] = None
    
    # 모델 고유의 값들 (engine.py는 이 값들이 존재한다고 가정하지 않음)
    MODEL_NAME: str = "BaseModel"
    LR: float = 0.001
    BETAS: Tuple[float, float] = (0.9, 0.999)
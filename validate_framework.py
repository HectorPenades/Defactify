#!/usr/bin/env python
"""Quick validation that all modules can be imported."""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Validating Defactify framework imports...\n")

try:
    print("✓ Importing src.config...")
    from src.config import load_config, Config

    print("✓ Importing src.data modules...")
    from src.data.base import BaseDataset
    from src.data.dataset import DefactifyDataset
    from src.data.transforms import ImagePreprocessor

    print("✓ Importing src.models modules...")
    from src.models.base import BaseModel
    from src.models.rgb_resnet import RGBResNet50

    print("✓ Importing src.training modules...")
    from src.training.base import BaseTrainer
    from src.training.trainer import DefaultTrainer
    from src.training.losses import get_loss_function
    from src.training.metrics import MetricsComputer

    print("✓ Importing src.inference modules...")
    from src.inference.evaluation import Evaluator

    print("✓ Importing src.utils modules...")
    from src.utils.logging import ExperimentLogger
    from src.utils.reproducibility import ReproducibilityManager

    print("\n" + "="*60)
    print("✅ All imports successful!")
    print("="*60)

    # Quick config test
    print("\nValidating config loading...")
    base_config_path = project_root / "configs" / "base_config.yaml"
    if base_config_path.exists():
        config = load_config(str(base_config_path))
        print(f"✓ Config loaded: {config['experiment_name']}")
        print(f"✓ Model architecture: {config['model']['architecture']}")
        print(f"✓ Dataset mode: {config['dataset']['mode']}")
    else:
        print(f"⚠ Base config not found at {base_config_path}")

    print("\n✅ Framework validation complete!")

except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)

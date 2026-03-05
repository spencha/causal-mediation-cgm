#!/usr/bin/env python3
"""
config.py
=========
Centralized configuration for the causal mediation analysis project.

This module handles all path definitions and project settings. It supports
three configuration methods (in order of precedence):

1. Environment variables (for HPC/cluster use)
2. Local config file (config_local.yaml, git-ignored)
3. Default paths relative to this file

Usage:
    from config import CONFIG
    data_path = CONFIG.ANALYSIS_DATA_DIR / "meal_windows.csv"
"""

import os
from pathlib import Path


def _get_project_root() -> Path:
    """
    Get the project root directory.
    
    For this project structure:
    - ae_python_code/config.py -> project root is parent
    """
    # This file is in ae_python_code/, so parent is project root
    return Path(__file__).parent.parent.resolve()


def _load_local_config() -> dict:
    """Load local configuration file if it exists."""
    config_file = _get_project_root() / "config_local.yaml"
    if config_file.exists():
        try:
            import yaml
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            # yaml not installed, skip local config
            pass
    return {}


class ProjectConfig:
    """Project configuration with paths and settings."""
    
    def __init__(self):
        self._local_config = _load_local_config()
        self._setup_paths()
    
    def _get_path(self, env_var: str, config_key: str, default: Path) -> Path:
        """Get path from environment, local config, or default."""
        # 1. Check environment variable
        if env_var in os.environ:
            return Path(os.environ[env_var])
        
        # 2. Check local config file
        if config_key in self._local_config:
            return Path(self._local_config[config_key])
        
        # 3. Return default
        return default
    
    def _setup_paths(self):
        """Setup all project paths matching actual directory structure."""
        
        # Project root (causal_ae/)
        self.PROJECT_ROOT = _get_project_root()
        
        # Base directory - can be overridden for HPC
        self.BASE_DIR = self._get_path(
            env_var="CAUSAL_AE_BASE_DIR",
            config_key="base_dir",
            default=self.PROJECT_ROOT
        )
        
        # ==============================================
        # CODE DIRECTORIES (relative to project root)
        # ==============================================
        
        # Python code directory
        self.AE_CODE_DIR = self.PROJECT_ROOT / "ae_python_code"
        
        # R cluster scripts
        self.CMA_CLUSTER_DIR = self.PROJECT_ROOT / "cma_cluster"
        
        # Data processing scripts
        self.DATA_PROCESSING_DIR = self.PROJECT_ROOT / "data_processing"
        
        # Raw data (OhioT1DM)
        self.RAW_DATA_DIR = self._get_path(
            env_var="CAUSAL_AE_RAW_DATA_DIR",
            config_key="raw_data_dir",
            default=self.BASE_DIR / "OhioT1DM"
        )

        # 2018 data paths
        self.OHIO_TRAIN_DIR = self.RAW_DATA_DIR / "2018" / "train"
        self.OHIO_TEST_DIR = self.RAW_DATA_DIR / "2018" / "test"

        # 2020 data paths
        self.OHIO_2020_TRAIN_DIR = self.RAW_DATA_DIR / "2020" / "train"
        self.OHIO_2020_TEST_DIR = self.RAW_DATA_DIR / "2020" / "test"
        
        # ==============================================
        # DATA DIRECTORIES
        # ==============================================
        
        # Main analysis data (embeddings, RData files, weights)
        self.ANALYSIS_DATA_DIR = self._get_path(
            env_var="CAUSAL_AE_DATA_DIR",
            config_key="analysis_data_dir",
            default=self.BASE_DIR / "cma_cluster" / "analysis_data"
        )
        
        # Weights subdirectory
        self.WEIGHTS_DIR = self.ANALYSIS_DATA_DIR / "weights"

        # Meal windows data (for AE training)
        # 2018 cohort data
        self.MEAL_WINDOWS_2018_DIR = self._get_path(
            env_var="CAUSAL_AE_MEAL_WINDOWS_2018_DIR",
            config_key="meal_windows_2018_dir",
            default=self.AE_CODE_DIR / "meal_windows_2018"
        )

        # Combined (2018 + 2020) train/test directories
        self.MEAL_WINDOWS_COMBINED_DIR = self.AE_CODE_DIR / "meal_windows_combined"
        self.MEAL_WINDOWS_COMBINED_TRAIN_DIR = self.AE_CODE_DIR / "meal_windows_combined" / "train"
        self.MEAL_WINDOWS_COMBINED_TEST_DIR = self.AE_CODE_DIR / "meal_windows_combined" / "test"

        # Legacy alias for backward compatibility
        self.MEAL_WINDOWS_DIR = self.MEAL_WINDOWS_2018_DIR
        
        # ==============================================
        # OUTPUT DIRECTORIES
        # ==============================================
        
        # Mediation results from HPC jobs
        self.MEDIATION_RESULTS_DIR = self._get_path(
            env_var="CAUSAL_AE_MEDIATION_RESULTS_DIR",
            config_key="mediation_results_dir",
            default=self.BASE_DIR / "cma_cluster" / "mediation_results"
        )
        
        # Horizon-specific embeddings directory
        self.HORIZON_EMBEDDINGS_DIR = self.ANALYSIS_DATA_DIR / "horizon_embeddings"
        
        # ==============================================
        # FIGURE DIRECTORIES (new organization)
        # ==============================================

        # Main visualizations directory
        self.VISUALIZATIONS_DIR = self._get_path(
            env_var="CAUSAL_AE_VISUALIZATIONS_DIR",
            config_key="visualizations_dir",
            default=self.BASE_DIR / "visualizations"
        )

        # Topic-specific visualization directories
        self.DATA_DISTRIBUTION_DIR = self.VISUALIZATIONS_DIR / "data_distribution"
        self.AE_EMBEDDINGS_DIR = self.VISUALIZATIONS_DIR / "ae_embeddings"
        self.NPCBPS_BALANCE_DIR = self.VISUALIZATIONS_DIR / "npcbps_balance"
        self.MEDIATION_VIZ_DIR = self.VISUALIZATIONS_DIR / "mediation_visualizations"
        self.EXPERIMENTS_VIZ_DIR = self.VISUALIZATIONS_DIR / "incremental_data_experiment"

        # Experiment results data (CSVs from autoencoder experiments)
        self.EXPERIMENT_RESULTS_DIR = self.EXPERIMENTS_VIZ_DIR / "data"

        # Legacy aliases for backward compatibility
        self.FIGURES_DIR = self.VISUALIZATIONS_DIR / "data_distribution" / "figures"
        self.AE_FIGURES_DIR = self.VISUALIZATIONS_DIR / "ae_embeddings" / "figures"
        self.MEDIATION_FIGURES_DIR = self.VISUALIZATIONS_DIR / "mediation_visualizations" / "figures"
        self.BALANCE_FIGURES_DIR = self.VISUALIZATIONS_DIR / "npcbps_balance" / "figures"
        self.MODEL_COMPARISON_FIGURES_DIR = self.VISUALIZATIONS_DIR / "incremental_data_experiment" / "figures"
        self.DATA_SUMMARY_DIR = self.VISUALIZATIONS_DIR / "data_distribution" / "figures"
    
    def ensure_dirs(self):
        """Create all output directories if they don't exist."""
        dirs_to_create = [
            self.WEIGHTS_DIR,
            self.MEDIATION_RESULTS_DIR,
            self.EXPERIMENT_RESULTS_DIR,
            # Note: HORIZON_EMBEDDINGS_DIR is created on-demand by scripts that use it
            self.MEAL_WINDOWS_COMBINED_TRAIN_DIR,
            self.MEAL_WINDOWS_COMBINED_TEST_DIR,
            # Visualization directories (figures and tables)
            self.DATA_DISTRIBUTION_DIR / "figures",
            self.DATA_DISTRIBUTION_DIR / "tables",
            self.AE_EMBEDDINGS_DIR / "figures",
            self.AE_EMBEDDINGS_DIR / "tables",
            self.NPCBPS_BALANCE_DIR / "figures",
            self.NPCBPS_BALANCE_DIR / "tables",
            self.MEDIATION_VIZ_DIR / "figures",
            self.MEDIATION_VIZ_DIR / "tables",
            self.EXPERIMENTS_VIZ_DIR / "figures",
            self.EXPERIMENTS_VIZ_DIR / "tables",
        ]
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)
    
    def print_config(self):
        """Print current configuration for debugging."""
        print("\n" + "=" * 60)
        print("PROJECT CONFIGURATION")
        print("=" * 60)
        print(f"  PROJECT_ROOT:           {self.PROJECT_ROOT}")
        print(f"  BASE_DIR:               {self.BASE_DIR}")
        print(f"  AE_CODE_DIR:            {self.AE_CODE_DIR}")
        print(f"  ANALYSIS_DATA_DIR:      {self.ANALYSIS_DATA_DIR}")
        print(f"  MEAL_WINDOWS_2018_DIR:  {self.MEAL_WINDOWS_2018_DIR}")
        print(f"  MEDIATION_RESULTS_DIR:  {self.MEDIATION_RESULTS_DIR}")
        print(f"  FIGURES_DIR:            {self.FIGURES_DIR}")
        print("=" * 60 + "\n")


# Global config instance
CONFIG = ProjectConfig()


if __name__ == "__main__":
    # When run directly, print configuration and verify paths
    CONFIG.print_config()
    
    print("Checking directory existence:")
    for name, path in [
        ("AE_CODE_DIR", CONFIG.AE_CODE_DIR),
        ("ANALYSIS_DATA_DIR", CONFIG.ANALYSIS_DATA_DIR),
        ("MEAL_WINDOWS_2018_DIR", CONFIG.MEAL_WINDOWS_2018_DIR),
        ("MEAL_WINDOWS_COMBINED_DIR", CONFIG.MEAL_WINDOWS_COMBINED_DIR),
        ("MEDIATION_RESULTS_DIR", CONFIG.MEDIATION_RESULTS_DIR),
        ("FIGURES_DIR", CONFIG.FIGURES_DIR),
    ]:
        status = "✓ exists" if path.exists() else "✗ missing"
        print(f"  {name}: {status}")

import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.helpers import load_config, setup_logger

logger = setup_logger("dataset_loader")


class MBICDataLoader:
    """
    Data Loader for the Media Bias Identification Corpus (MBIC).
    
    Supports:
    - Raw sentence-level annotations with annotator metadata (1,700 sentences).
    - Annotator demographics and political ideology records (784 annotators).
    - Token-level dataset with engineered linguistic & contextual features (31,794 words).
    - Train / Validation / Test reproducible partitioning matching Spinde et al. (2021).
    """

    def __init__(self, config_or_path: Optional[Any] = None, base_dir: Optional[str] = None):
        if isinstance(config_or_path, dict):
            self.config = config_or_path
        elif isinstance(config_or_path, str):
            self.config = load_config(config_or_path)
        else:
            self.config = load_config("configs/config.yaml")

        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_dir = base_dir

        self.paths = self.config.get("paths", {})
        self.target_label = self.config.get("dataset", {}).get("target_label", "label4")

    def _resolve_path(self, relative_path: str) -> str:
        """Resolve relative path against project base directory."""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.base_dir, relative_path)

    def load_raw_sentences(self) -> pd.DataFrame:
        """
        Load the sentence-level MBIC dataset (1,700 sentences from 14 topics).
        Contains sentence text, outlet, topic, type, and aggregated crowd annotations.
        """
        rel_path = self.paths.get("raw_sentences", "data/raw/labeled_dataset.parquet")
        abs_path = self._resolve_path(rel_path)
        logger.info(f"Loading raw sentences dataset from: {abs_path}")

        if os.path.exists(abs_path):
            if abs_path.endswith(".parquet"):
                return pd.read_parquet(abs_path)
            elif abs_path.endswith((".xlsx", ".xls")):
                return pd.read_excel(abs_path)
            else:
                return pd.read_csv(abs_path)
        
        # Fallback to xlsx if parquet not yet generated
        alt_path = abs_path.replace(".parquet", ".xlsx")
        if os.path.exists(alt_path):
            logger.info(f"Loading fallback excel file: {alt_path}")
            return pd.read_excel(alt_path)

        raise FileNotFoundError(f"Sentence dataset not found at {abs_path} or {alt_path}")

    def load_annotations(self) -> pd.DataFrame:
        """
        Load the raw crowd annotations: one row per annotator per sentence (17,775 rows).
        Holds each annotator's sentence label, free-text word highlights (`words`)
        and background (including `political_ideology`, -10 liberal to +10 conservative).
        """
        rel_path = self.paths.get("annotations", "data/raw/annotations.xlsx")
        abs_path = self._resolve_path(rel_path)
        logger.info(f"Loading raw per-annotator annotations from: {abs_path}")

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Annotations file not found at {abs_path}")
        return pd.read_excel(abs_path)

    def load_annotators(self) -> pd.DataFrame:
        """
        Load annotator background characteristics (784 participants).
        Includes demographics, education, political orientation, and news consumption.
        """
        rel_path = self.paths.get("annotators", "data/raw/annotators.csv")
        abs_path = self._resolve_path(rel_path)
        logger.info(f"Loading annotator metadata from: {abs_path}")

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Annotator metadata file not found at {abs_path}")
        return pd.read_csv(abs_path)

    def load_token_dataset(self, drop_missing_tfidf: bool = True) -> pd.DataFrame:
        """
        Load the token-level feature dataset (dt_final) containing words,
        linguistic tags, sentiment lexicons, and crowd bias labels.
        
        Matches paper preprocessing:
        - Drops tokens where tfidf_art is missing (tokens from missing full articles).
        """
        rel_path = self.paths.get("token_features_dataset", "data/processed/dt_final.parquet")
        abs_path = self._resolve_path(rel_path)
        logger.info(f"Loading token-level feature dataset from: {abs_path}")

        if os.path.exists(abs_path):
            if abs_path.endswith(".parquet"):
                df = pd.read_parquet(abs_path)
            elif abs_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(abs_path)
            else:
                df = pd.read_csv(abs_path)
        else:
            alt_path = abs_path.replace(".parquet", ".xlsx")
            if os.path.exists(alt_path):
                logger.info(f"Loading fallback excel token dataset: {alt_path}")
                df = pd.read_excel(alt_path)
            else:
                raise FileNotFoundError(f"Token dataset not found at {abs_path} or {alt_path}")

        if drop_missing_tfidf and "tfidf_art" in df.columns:
            initial_len = len(df)
            df = df.dropna(subset=["tfidf_art"]).copy()
            logger.info(f"Dropped {initial_len - len(df)} rows with missing 'tfidf_art'. Remaining rows: {len(df)}")

        return df

    def get_train_test_split(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: float = 0.10,
        val_size: float = 0.10,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Partition dataset into Train (81%), Validation (9%), and Test (10%) sets
        strictly replicating the protocol in Section 3.6 of Spinde et al. (2021).
        
        Returns:
            Dictionary containing train, val, and test feature DataFrames and label Series.
        """
        if df is None:
            df = self.load_token_dataset(drop_missing_tfidf=True)

        target = self.target_label
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found in dataset columns.")

        X = df.copy()
        y = df[target].copy()

        # Split 90% train_val, 10% test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Split remaining 90% into 81% train and 9% val (10% of train_val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )

        logger.info(
            f"Dataset split complete: Train={len(X_train)} ({(1-test_size)*(1-val_size):.1%}), "
            f"Val={len(X_val)} ({(1-test_size)*val_size:.1%}), "
            f"Test={len(X_test)} ({test_size:.1%})"
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": [c for c in X.columns if c not in ["label3", "label4", "label5"]],
            "total_samples": len(df),
            "positive_samples": int((y == 1).sum()),
            "negative_samples": int((y == 0).sum()),
            "imbalance_ratio": float((y == 1).sum() / len(y)),
        }

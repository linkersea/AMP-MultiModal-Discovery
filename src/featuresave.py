from features.feature_extractor import FeatureExtractor

if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.batch_extract_and_save(
        csv_path=r"F:\Biobert_PEP\data\raw\120dataset.csv",
        seq_col="sequence",
        save_path=r"F:\Biobert_PEP\data\processed\preprocessed_data_phy.csv"
    )
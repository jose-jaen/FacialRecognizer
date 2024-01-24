from data_extractor import DataExtractor

if __name__ == '__main__':
    data_extractor = DataExtractor(n_hidden=25, n_test=3)

    # Download the data
    data_extractor.get_data()

    # Partition the data
    data_extractor.set_up_data(random_state=42)

    # Get validation split
    data_extractor.get_validation(n_valid=8)

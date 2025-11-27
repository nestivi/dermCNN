# main.py

from data_loader import load_dataframe, split_data, get_generators
from trainer import train_model


def main():
    df = load_dataframe()
    train_df, test_df = split_data(df)

    train_gen, test_gen = get_generators(train_df, test_df)

    train_model(train_gen, test_gen)


if __name__ == "__main__":
    main()

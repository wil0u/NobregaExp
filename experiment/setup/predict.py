from experiment import utils
from src import data_utils
import pandas as pd




def main():
    logger = utils.get_logger("predict")

    # setup
    exp_setup = utils.setup()
    df = pd.read_csv('exp_result.csv')
    df_input_file = pd.read_csv('C:\\Users\\achan\\Downloads\\exp_willeme.csv', sep=";", header=0)
    userIds = [str(uid) for uid in df_input_file['userId']]

    all_films = pd.read_csv('C:\\Users\\achan\\PycharmProjects\\LIRE\\ml-latest-small\\movies.csv')['movieId'].tolist()

    # load data and rec model
    logger.info("Load data and recommender")
    dataset = data_utils.load_data()
    rec_model = data_utils.load_dump(exp_setup.rec_name)

    # calculate predictions
    logger.info("Generate predictions")
    # predictions = rec_model.predict(dataset.test_df)
    # predictions2 = rec_model.predict(dataset.training_df)
    # dataset.training_df['prediction'] = predictions2
    # dataset.test_df['prediction'] = predictions
    # data_utils.save_predictions(dataset.test_df, exp_setup.rec_name)

    rating_pred = []

    # calculate recs
    logger.info("Generate recommendations")
    selected_users = ["1"]
    recs = rec_model.recommend(userIds)
    listpred = []
    listMae = []

    for index, row in df.iterrows():
        r = int(row[1])
        r2 = int(row[2])
        white_box_pred = row[6]
        match = recs.loc[(recs['user_id'] == str(r)) & (recs['item_id'] == str(r2))]
        a = match['user_id'].tolist()[0]
        b = match['item_id'].tolist()[0]
        c = match['prediction'].tolist()[0]

        listpred.append(c)
        listMae.append(abs(c - white_box_pred))
        print()
    df['BlackBoxPred'] = listpred
    df['MAE'] = listMae
    df = df.iloc[:, 1:]
    df.to_csv("exp_result_mae2.csv", index=False)
    print()
    data_utils.save_recs(recs, exp_setup.rec_name)


if __name__ == "__main__":
    main()

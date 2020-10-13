import json
import os

import numpy as np
import pandas as pd

from experiment import utils
from src import data_utils
from src.lime_rs import LimeRSExplainer
import src.lime_rs as lime_rs

from scipy.spatial.distance import cosine as cosine_dist


logger = utils.get_logger("limers")
OUTFILE = 'exp_edbt_res_2.csv'
INFILE = 'C:\\Users\\achan\\PycharmProjects\\LIRE\\res\\edbt\\exp_edbt_283_14_14.csv'
MOVIE_LIST = 'C:\\Users\\achan\\PycharmProjects\\LIRE\\ml-20m\\movies.csv'


def extract_features(explanation_all_ids, feature_type, feature_map):
    filtered_dict = dict()
    if feature_type == "features":
        for tup in explanation_all_ids:
            if not (feature_map[tup[0]].startswith('user_id') or
                    feature_map[tup[0]].startswith('item_id')):
                filtered_dict[feature_map[tup[0]]] = round(tup[1], 3)

    elif feature_type == "item":
        top_features = 10000
        for tup in explanation_all_ids:
            if feature_map[tup[0]].startswith('item_id') and len(filtered_dict) <= top_features:
                filtered_dict[feature_map[tup[0]]] = round(tup[1], 3)

    return filtered_dict


def generate_explanations(instances_to_explain, explainer, rec_model, feature_type='item'):
    result = list()

    for instance in instances_to_explain.itertuples(index=False):
        logger.info("explaining-> (user: {}, item: {})".format(instance.user_id, instance.item_id))

        exp = explainer.explain_instance(instance,
                                         rec_model,
                                         neighborhood_entity="item",
                                         labels=[0],
                                         num_samples=1000)

        # filter
        filtered_features = extract_features(exp.local_exp[0],
                                             feature_type=feature_type,
                                             feature_map=explainer.feature_map)
        #
        explanation_str = json.dumps(filtered_features)
        output_df = pd.DataFrame({'user_id': [instance.user_id], 'item_id': [instance.item_id],
                                  'explanations': [explanation_str],
                                  'local_prediction': [round(exp.local_pred[0], 3)]})

        result.append(output_df)

    return pd.concat(result)


def main():
    # setup
    exp_setup = utils.setup()
    # load data and rec model
    logger.info("Load data and recommender")
    rec_model = data_utils.load_dump(exp_setup.rec_name)

    # setup explainer
    feature_names = rec_model.one_hot_columns
    feature_map = {i: rec_model.one_hot_columns[i] for i in range(len(list(rec_model.one_hot_columns)))}
    explainer = LimeRSExplainer(rec_model.dataset.training_df,
                                feature_names=feature_names,
                                feature_map=feature_map,
                                mode='regression',
                                class_names=np.array(['rec']),
                                feature_selection='none')

    print("Loading eval data")
    #uid,iid,train_size,pert_ratio,robustness_5,robustness_10,robustness_15,mae,neighbors,dist_to_neigh
    df_input_file = pd.read_csv(INFILE, sep=",", header=0)
    userIds = [uid for uid in df_input_file['uid']]
    listesVoisins = [list(map(int, listeVoisins.split(';'))) for listeVoisins in df_input_file['neighbors']]
    listesDistances = [list(map(float, listeDistances.split(';'))) for listeDistances in df_input_file['dist_to_neigh']]

    all_films = pd.read_csv(MOVIE_LIST)['movieId'].tolist()
    rev_iid_map = {v: k for k, v  in lime_rs.iid_map.items()}
    movieIds = [rev_iid_map[iid] for iid in df_input_file['iid']]
    dicti_pred_black_box_white_box = {}


    toBeExplained = []
    for i, couple in enumerate(zip(userIds, movieIds)):
        userId, movieId = couple
        listeVoisins = listesVoisins[i]

        toBeExplained.append((userId, movieId))
        for voisin in listeVoisins:
            toBeExplained.append((voisin, movieId))


    print("End of block 1")


    listDict = {}
    print("Generating", len(toBeExplained), "explanations ...")
    instances_to_explain = pd.DataFrame(toBeExplained, columns=["user_id", "item_id"])
    explanations = generate_explanations(instances_to_explain, explainer, rec_model)

    for index, row in explanations.iterrows():
        key_dict = str(row[0]) + '_' + str(row[1])

        movieIdDictionnaryExp = dict.fromkeys(all_films, 0)
        json_dictionnary = json.loads(row[2])
        dicti_pred_black_box_white_box[key_dict] = [row[3]]
        pred = LimeRSExplainer.make_black_box_slice(lime_rs.U, lime_rs.sigma, lime_rs.Vt, lime_rs.user_means, [row[0]])[0][lime_rs.iid_map[row[1]]]
        dicti_pred_black_box_white_box[key_dict].append(pred)

        for key in json_dictionnary:
            id = int(''.join(key.split('item_id_')))
            movieIdDictionnaryExp[id] = json_dictionnary[key]

        listDict[key_dict] = movieIdDictionnaryExp

    # Robustness compute
    dictio_ratio_target = {}
    for i, couple in enumerate(zip(userIds, movieIds)):
        userId, movieId = couple
        print("Working on user", userId, "movie", movieId)
        listeVoisins = listesVoisins[i]
        listeDistances = listesDistances[i]
        key_dict = str(userId) + '_' + str(movieId)

        explanations_dict = listDict[key_dict]
        explanations_vector_target = [explanations_dict[key] for key in sorted(explanations_dict.keys())]
        listeRatioMax = []
        listeRatio = []
        n = 0
        for voisin in listeVoisins:
            key_dict = str(voisin) + '_' + str(movieId)
            explanations_dict_neighbor = listDict[key_dict]
            explanations_vector_neighbor = [explanations_dict_neighbor[key] for key in sorted(explanations_dict_neighbor.keys())]
            ratio = cosine_dist(explanations_vector_target, explanations_vector_neighbor)/listeDistances[n]
            listeRatio.append(ratio)
            if(n == 4):
                listeRatioMax.append(np.max(listeRatio))
            if(n == 9):
                listeRatioMax.append(np.max(listeRatio))
            n = n + 1

        listeRatioMax.append(np.max(listeRatio))
        dictio_ratio_target[str(userId) + '_' + str(movieId)] = listeRatioMax

    columns = ['userId', 'movieId', 'maxAt5', 'maxAt10', 'maxAt15', 'WhiteBoxPred', 'BlackBoxPred', 'MAE']
    list_Output = []
    for key in dictio_ratio_target.keys():
        splitoutput = key.split('_')
        dict_key = splitoutput[0]+'_'+splitoutput[1]
        wb_pred = dicti_pred_black_box_white_box[dict_key][0]
        bb_pred = dicti_pred_black_box_white_box[dict_key][1]
        list_Output.append([splitoutput[0],
                            splitoutput[1],
                            dictio_ratio_target[key][0],
                            dictio_ratio_target[key][1],
                            dictio_ratio_target[key][2],
                            wb_pred,
                            bb_pred,
                            abs(wb_pred - bb_pred)
                            ])
    df_output_ = pd.DataFrame.from_records(list_Output, columns=columns)
    df_output_.to_csv(OUTFILE)
    print("Done saved data to", OUTFILE)
    # save
    logger.info("Save LimeRS explanations")
    output_filename = "limers_explanations-{}".format(exp_setup.rec_name)
    explanations.to_csv(path_or_buf=os.path.join(data_utils.DEFAULT_OUTPUT_FOLDER, output_filename),
                        sep='\t', index=False, header=True)


if __name__ == '__main__':
    main()

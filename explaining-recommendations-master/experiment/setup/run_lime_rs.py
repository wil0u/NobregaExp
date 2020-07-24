import json
import os

import numpy as np
import pandas as pd

from experiment import utils
from src import data_utils
from src.lime_rs import LimeRSExplainer
import random
import collections
from scipy.spatial.distance import cosine as cosine_dist
import numpy as np
import math

logger = utils.get_logger("limers")


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



    '''debut de mon code :'''
    df_input_file = pd.read_csv('/media/wilou/Elements/developmentNobra/exp_willeme.csv',sep=";",header=0)
    userIds = [uid for uid in df_input_file['userId']]
    listesVoisins = [eval(listeVoisins) for listeVoisins in df_input_file['voisins']]
    listesDistances = [eval(listeDistances) for listeDistances in df_input_file['distances']]
    trainItems = rec_model.dataset.training_df['item_id'].to_list()
    testItems = rec_model.dataset.test_df['item_id'].to_list()
    all_films = pd.read_csv('/media/wilou/Elements/developmentNobra/ml-latest-small/movies.csv')['movieId'].tolist()
    movieIds = random.choices(testItems + trainItems, k=4)
    dicti_pred_black_box_white_box = {}

    t = [(str(uid),str(iid))  for uid in userIds for iid in movieIds]
    #df_pr['userId'] = df_input_file['userId']
    df_pr = pd.DataFrame.from_records(t,columns=['userId','itemId'])
    prd = rec_model.predict(df_pr)
    toBeExplained = []
    for i in range(len(userIds)):
        userId = userIds[i]
        listeVoisins = listesVoisins[i]

        for movieId in movieIds:

            toBeExplained.append((userId,movieId))
            for voisin in listeVoisins:
                df_pr = pd.DataFrame([(str(voisin), str(movieId))])
                pred = rec_model.predict(df_pr)
                toBeExplained.append((voisin,movieId))


    print()


    #
    listDict = {}
    instances_to_explain = pd.DataFrame(toBeExplained, columns=["user_id", "item_id"])
    explanations = generate_explanations(instances_to_explain, explainer, rec_model)
    for index, row in explanations.iterrows():
        movieIdDictionnaryExp = dict.fromkeys(all_films, 0)
        json_dictionnary = json.loads(row[2])
        dicti_pred_black_box_white_box[str(row[0]) + '_' + str(row[1])] = [row[3]]
        df_pr = pd.DataFrame([(str(row[0]), str(row[1]))])
        pred = rec_model.predict(df_pr)
        dicti_pred_black_box_white_box[str(row[0]) + '_' + str(row[1])].append(pred)

        for key in json_dictionnary:

            id = int(''.join(key.split('item_id_')))
            movieIdDictionnaryExp[id] = json_dictionnary[key]
        key_dict = str(row[0])+'_'+str(row[1])
        listDict[key_dict] = movieIdDictionnaryExp

    dictio_ratio_target = {}
    for i in range(len(userIds)):
        userId = userIds[i]
        listeVoisins = listesVoisins[i]
        listeDistances = listesDistances[i]

        for movieId in movieIds:
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
                ratio = cosine_dist(explanations_vector_target,explanations_vector_neighbor)/listeDistances[n]
                listeRatio.append(ratio)
                if(n == 4):
                    listeRatioMax.append(np.max(listeRatio))
                if(n == 9):
                    listeRatioMax.append(np.max(listeRatio))
                n = n + 1

            listeRatioMax.append(np.max(listeRatio))
            dictio_ratio_target[str(userId) + '_' + str(movieId)] = listeRatioMax

        print()
    columns = ['userId','movieId','maxAt5','maxAt10','maxAt15','WhiteBoxPred','BlackBoxPred','MAE']
    list_Output = []
    for key in dictio_ratio_target.keys():
        splitoutput = key.split('_')
        list_Output.append([splitoutput[0],
                            splitoutput[1],
                            dictio_ratio_target[key][0],
                            dictio_ratio_target[key][1],
                            dictio_ratio_target[key][2],
                           dicti_pred_black_box_white_box[splitoutput[0]+'_'+splitoutput[1]][0],
                            dicti_pred_black_box_white_box[splitoutput[0] + '_' + splitoutput[1]][1][0],
                            abs(dicti_pred_black_box_white_box[splitoutput[0]+'_'+splitoutput[1]][0] - dicti_pred_black_box_white_box[splitoutput[0] + '_' + splitoutput[1]][1][0])
                            ])
    df_output_ = pd.DataFrame.from_records(list_Output,columns=columns)
    df_output_.to_csv('exp_result.csv')
    print()
    # save
    logger.info("Save LimeRS explanations")
    output_filename = "limers_explanations-{}".format(exp_setup.rec_name)
    explanations.to_csv(path_or_buf=os.path.join(data_utils.DEFAULT_OUTPUT_FOLDER, output_filename),
                        sep='\t', index=False, header=True)


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import sklearn
import pickle
from lime import lime_base, explanation
from sklearn.utils import check_random_state
from scipy.sparse import csr_matrix

from src.dataset import Dataset

black_box_folder = "C:\\Users\\achan\\PycharmProjects\\LIRE\\temp\\"
U = np.loadtxt(black_box_folder + "U.gz")
sigma = np.loadtxt(black_box_folder + "sigma.gz")
Vt = np.loadtxt(black_box_folder + "Vt.gz")
user_means = np.loadtxt(black_box_folder + 'user_means.gz')
iid_map = pickle.load(open(black_box_folder + "iid_map.p", mode="rb"))


class LimeRSExplainer():

    def __init__(self,
                 training_df,
                 feature_names,
                 feature_map,
                 mode="classification",
                 kernel_width=25,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 random_state=None):

        # exponential kernel
        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel, verbose,
                                       random_state=self.random_state)

        self.feature_names = list(feature_names)
        self.feature_map = feature_map
        self.mode = mode
        self.class_names = class_names
        self.feature_selection = feature_selection

        self.categorical_features = list(range(feature_names.shape[0]))

        self.n_rows = training_df.shape[0]
        # TODO hardcoded 20M dataset
        self.training_df = pd.read_csv(black_box_folder + "ratings.csv", header=0,
                                       names=['user_id', 'item_id', 'rating', 'timestamp']).drop(labels=["rating", "timestamp"], axis='columns')

        self.user_freq = self.training_df['user_id'].value_counts(normalize=True)
        self.item_freq = self.training_df['item_id'].value_counts(normalize=True)

    @staticmethod
    def read_sparse(rfile):
        indptr = [0]
        indices = []
        data = []
        vocabulary = {}

        curr_id = 1

        with open(rfile) as f:
            skip = 1
            for l in f:
                if skip:
                    skip -= 1
                    continue

                l = l.split(",")
                if int(l[0]) != curr_id:
                    # new user
                    indptr.append(len(indices))
                    curr_id = int(l[0])
                index = vocabulary.setdefault(int(l[1]), len(vocabulary))
                indices.append(index)
                data.append(float(l[2]))
            indptr.append(len(indices))  # one last time
        return csr_matrix((data, indices, indptr), dtype=float), vocabulary

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    @staticmethod
    def make_black_box_slice(U, sigma, Vt, means, indexes):
        return (U[indexes] @ sigma @ Vt) + np.tile(means[indexes].reshape(len(indexes), 1), (1, Vt.shape[1]))

    def explain_instance(self,
                         instance,
                         rec_model,
                         neighborhood_entity,
                         labels=(1,),
                         num_features=10,
                         num_samples=50,
                         distance_metric='cosine',
                         model_regressor=None):

        # get neighborhood
        neighborhood_df = self.generate_neighborhood(instance, neighborhood_entity, num_samples)

        # compute distance based on interpretable format
        data, _ = Dataset.convert_to_pyfm_format(neighborhood_df, columns=rec_model.one_hot_columns)
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        # get predictions from original complex model
        # TODO change to use our black box model, problem item and user is need to match ....
        #yss = np.array(rec_model.predict(neighborhood_df))
        bb_iids = np.array(list(map(iid_map.__getitem__, map(int, neighborhood_df["item_id"]))))
        slice = self.make_black_box_slice(U, sigma, Vt, user_means, np.array(list(map(int, set(neighborhood_df.user_id)))))
        yss_svd = slice[:, bb_iids].transpose()

        # for classification, the model needs to provide a list of tuples - classes along with prediction probabilities
        if self.mode == "classification":
            raise NotImplementedError("LIME-RS does not currently support classifier models.")
        # for regression, the output should be a one-dimensional array of predictions
        else:
            predicted_value = slice[0, bb_iids]
            min_y = min(yss_svd)
            max_y = max(yss_svd)

        ret_exp = explanation.Explanation(domain_mapper=None,
                                          mode=self.mode,
                                          class_names=self.class_names)
        if self.mode == "classification":
            raise NotImplementedError("LIME-RS does not currently support classifier models.")
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data,
                yss_svd,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

        return ret_exp

    def generate_neighborhood(self, instance, entity, num_samples):
        samples = list()
        samples.append({"user_id": str(instance.user_id), "item_id": str(instance.item_id)})
        if entity == 'user':
            sample_users = np.random.choice(self.user_freq.index.tolist(), num_samples - 1, replace=False,
                                            p=self.user_freq.values.tolist())
            for u in sample_users:
                samples.append({"user_id": str(u), "item_id": str(instance.item_id)})

        elif entity == 'item':
            sample_items = np.random.choice(self.item_freq.index.tolist(), num_samples - 1, replace=False,
                                            p=self.item_freq.values.tolist())
            for i in sample_items:
                samples.append({"user_id": str(instance.user_id), "item_id": str(i)})
        else:
            sample_rows = np.random.choice(range(self.n_rows), num_samples - 1, replace=False)
            for s in self.training_df.iloc[sample_rows].itertuples():
                samples.append({"user_id": str(s.user_id), "item_id": str(s.item_id)})

        samples_df = pd.DataFrame(samples)
        samples_df = samples_df[['user_id', 'item_id']]

        return samples_df
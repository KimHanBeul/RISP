import os
import numpy as np
import torch
import random
import pickle
import networkx as nx
import json
import pandas as pd
from tqdm import tqdm
from collections import Counter

from recbole.trainer import Trainer
from recbole.utils import EvaluatorType, set_color
from recbole.data.interaction import Interaction

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist



class SelectedUserTrainer(Trainer):
    def __init__(self, config, model, dataset):
        super().__init__(config, model)
        self.selected_user_suffix = config['selected_user_suffix']  # candidate generation model, by default, random
        self.recall_budget = config['recall_budget']                # size of candidate Sets, by default, 20
        self.fix_pos = config['fix_pos']                            # whether fix the position of ground-truth items in the candidate set, by default, -1
        self.selected_uids, self.sampled_items = self.load_selected_users(config, dataset)

        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.ITEM_SEQ = self.ITEM_ID + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID

    def load_selected_users(self, config, dataset):
        selected_users = []
        sampled_items = []
        selected_user_file = os.path.join(config['data_path'], f'{config["dataset"]}.{self.selected_user_suffix}')
        user_token2id = dataset.field2token_id['user_id']
        item_token2id = dataset.field2token_id['item_id']
        count = 0
        with open(selected_user_file, 'r', encoding='utf-8') as file:
            for line in file:
                uid, iid_list = line.strip().split('\t')
                selected_users.append(uid)
                sampled_items.append([item_token2id[_] if (_ in item_token2id) else 0 for _ in iid_list.split(' ')])
                count+=1
                if count >=self.config['num_data']:
                    break
        selected_uids = list([user_token2id[_] for _ in selected_users])
        return selected_uids, sampled_items

    @torch.no_grad()
    def evaluate(
        self, eval_data, valid_data, load_best_model=True, model_file=None, show_progress=False
    ):
        self.model.eval()
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )
        unsorted_selected_interactions = []
        unsorted_selected_pos_i = []
        unsorted_selected_user_seqs = []
        unsorted_selected_uids = []
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, history_index, positive_u, positive_i = batched_data
            for i in range(len(interaction)):
                if interaction['user_id'][i].item() in self.selected_uids:
                    pr = self.selected_uids.index(interaction['user_id'][i].item())
                    # print (f"pr: {pr}")
                    unsorted_selected_interactions.append((interaction[i], pr))
                    unsorted_selected_pos_i.append((positive_i[i], pr))
                    unsorted_selected_uids.append((interaction['user_id'][i].item(), pr))

                    test_item_seq = interaction[self.ITEM_SEQ][i]#.detach().cpu().tolist()
                    test_item_seq_len = interaction[self.ITEM_SEQ_LEN][i]
                    test_real_len = min(self.config['max_his_len'], test_item_seq_len.item())
                    unsorted_selected_user_seqs.append((test_item_seq.detach().cpu().tolist()[:test_real_len], pr))
        unsorted_selected_interactions.sort(key=lambda t: t[1])
        unsorted_selected_pos_i.sort(key=lambda t: t[1])
        unsorted_selected_user_seqs.sort(key=lambda t: t[1])
        unsorted_selected_uids.sort(key=lambda t: t[1])
        selected_interactions = [_[0] for _ in unsorted_selected_interactions]
        selected_pos_i = [_[0] for _ in unsorted_selected_pos_i]
        selected_user_seqs = [_[0] for _ in unsorted_selected_user_seqs]
        selected_uidsx = [_[0] for _ in unsorted_selected_uids]

        new_inter = {
            col: torch.stack([inter[col] for inter in selected_interactions]) for col in
            selected_interactions[0].columns
        }
        selected_interactions = Interaction(new_inter)
        selected_pos_i = torch.stack(selected_pos_i)
        selected_pos_u = torch.arange(selected_pos_i.shape[0])

        if self.config['has_gt']: # should be true here.
            self.logger.info('Has ground truth.')
            idxs = torch.LongTensor(self.sampled_items)
            for i in range(idxs.shape[0]):
                if selected_pos_i[i] in idxs[i]:
                    pr = idxs[i].numpy().tolist().index(selected_pos_i[i].item())
                    idxs[i][pr:-1] = torch.clone(idxs[i][pr+1:])

            idxs = idxs[:,:self.recall_budget - 1]
            if self.fix_pos == -1 or self.fix_pos == self.recall_budget - 1:
                idxs = torch.cat([idxs, selected_pos_i.unsqueeze(-1)], dim=-1).numpy()
            elif self.fix_pos == 0:
                idxs = torch.cat([selected_pos_i.unsqueeze(-1), idxs], dim=-1).numpy()
            else:
                idxs_a, idxs_b = torch.split(idxs, (self.fix_pos, self.recall_budget - 1 - self.fix_pos), dim=-1)
                idxs = torch.cat([idxs_a, selected_pos_i.unsqueeze(-1), idxs_b], dim=-1).numpy()
        else:
            self.logger.info('Does not have ground truth.')
            idxs = torch.LongTensor(self.sampled_items)
            idxs = idxs[:,:self.recall_budget]
            idxs = idxs.numpy()

        if self.fix_pos == -1: # should be -1
            self.logger.info('Shuffle ground truth')
            for i in range(idxs.shape[0]):
                np.random.shuffle(idxs[i])       


        # Get train and test_data_seq embedding matrix
        train_emb_file = os.path.join(self.config['data_path'], f"item_seq_emb.pt")
        self.train_seq_emb = torch.load(train_emb_file)

        test_emb_file = os.path.join(self.config['data_path'], f"test_seq_emb.pt")
        self.test_seq_emb = torch.load(test_emb_file)

        # Save test dataset for test dataset embedding
        test_seq_path = os.path.join(self.config['data_path'], f"test_data_seq.pickle")
        test_seq_df = pd.DataFrame({
            'item_id_list': selected_interactions['item_id_list'].tolist(),
            'item_length' : selected_interactions['item_length'].tolist(),
        })
        test_seq_df.to_pickle(test_seq_path)

        scores = self.model.predict_on_subsets(selected_interactions.to(self.device), idxs, self.global_graph, self.train_data_seq, self.train_seq_emb, self.test_seq_emb, valid_data, self.selected_uids)
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        self.eval_collector.eval_batch_collect(
            scores, selected_interactions, selected_pos_u, selected_pos_i
        )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result
    
    def _make_global_graph(self, train_data, valid_data):
        global_relations_path = os.path.join(self.config['data_path'], f"global_relation.pickle")
        pop_dict_path = os.path.join(self.config['data_path'], f"pop_dict.pickle")
        train_seq_path = os.path.join(self.config['data_path'], f"train_data_seq.pickle")

        try:
            global_relations = pickle.load(open(global_relations_path, 'rb'))
            pop_dict = pickle.load(open(pop_dict_path, 'rb'))
            self.train_data_seq = pickle.load(open(train_seq_path, 'rb'))
        except:
            print("Make Global Graph")
            # remove augmentation from train data
            uids = np.unique(train_data['user_id'])
            users_full_sequence = []    # for making graph
            users_sliced_sequence = []  # for making seq embedding 

            temp_dict = {'user_id': train_data['user_id'].tolist(), 'item_id_list': train_data['item_id_list'].tolist(), 'item_id': train_data['item_id'].tolist()}
            train_data2 = pd.DataFrame.from_dict(temp_dict)

            for i in tqdm(range(len(uids))):
                user_seqs = train_data2[train_data2['user_id'] == uids[i]]['item_id_list'].tolist()
                full_sequence = []

                for seq in user_seqs:
                    indices = np.nonzero(seq)
                    full_sequence.append(seq[indices[0][-1]])
                full_sequence.append(train_data2[train_data2['user_id'] == uids[i]]['item_id'].iloc[-1].item())
                full_sequence.append(valid_data['item_id'][valid_data['user_id'] == uids[i]].item())

                # slice full sequence into max length 100
                sliced_lists = []
                if len(full_sequence) > 100:
                    sliced_lists = [full_sequence[i:i + 100] for i in range(0, len(full_sequence), 100)]
                    if len(sliced_lists) > 1 and len(sliced_lists[-1]) < 100:
                        sliced_lists[-2].extend(sliced_lists.pop())

                    for sliced_list in sliced_lists:
                        users_sliced_sequence.append(sliced_list)
                else:
                    users_sliced_sequence.append(full_sequence)

                users_full_sequence.append(full_sequence)

            full_seq_df = pd.DataFrame({
                'item_id_list': users_sliced_sequence,
            })
            full_seq_df.to_pickle(train_seq_path)
            self.train_data_seq = full_seq_df

            # make global graph
            global_relations = []
            for i in tqdm(range(len(uids))):
                cur_user_full_data = users_full_sequence[i]
                for j in range(len(cur_user_full_data) - 1):
                    global_relations.append([int(cur_user_full_data[j]), int(cur_user_full_data[j-1])])

            global_relations = np.array(global_relations)
            pickle.dump(global_relations, open(global_relations_path, 'wb'))

            pop_dict = Counter(np.concatenate(global_relations))
            pop_dict = {key: value/sum(pop_dict.values()) for key, value in pop_dict.items()}
            pickle.dump(pop_dict, open(pop_dict_path, 'wb'))

        self.global_graph = nx.from_edgelist(global_relations, create_using=nx.DiGraph)
        for node in self.global_graph.nodes():
            self.global_graph.nodes[node]['weight'] = pop_dict[node]

        print("Global Graph Constrcution End")
        # return global_graph, users_full_sequence

    # def get_train_seq_emb(self):
    #     seq_text = []

    #     for user_seq in self.train_data_seq['item_id_list'].tolist():
    #         seq_text.append(' [SEP] '.join([self.model.item_text[seq_id] for seq_id in user_seq]))

    #     self.train_seq_emb = self.model.sentence_model.encode(seq_text, show_progress_bar = False)
    #     train_emb_file = os.path.join(self.config['data_path'], "output_files", f"train_seq_emb.pt")
    #     torch.save(self.train_seq_emb, train_emb_file)

    def get_emb_multivector(self, train_data, valid_data, eval_data, load_best_model=True, model_file=None, show_progress=False):
        self.model.eval() 
        item_text = self.model.item_text

        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        train_iter_data = (
            tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Train   ", "pink"),
            )
            if show_progress
            else valid_data
        )
        training_user_seqs = []
        training_user_pos_items = []
        training_user_ids = []
        for batch_idx, batched_data in enumerate(train_iter_data):
            interaction, history_index, positive_u, positive_i = batched_data
            pos_items = interaction[self.POS_ITEM_ID]
            item_seq_lens = interaction[self.ITEM_SEQ_LEN]
            item_seqs = interaction[self.ITEM_SEQ]

            user_ids = interaction['user_id']
            training_user_ids.extend(user_ids.tolist())

            training_user_pos_items.extend(pos_items)
            for train_i in range(len(item_seqs)):
                train_real_len = min(self.config['max_his_len'], item_seq_lens[train_i].item())
                # if train_real_len<10:
                #     continue
                training_user_seqs.append(item_seqs[train_i].detach().cpu().tolist()[:train_real_len])
            # break

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )
        unsorted_selected_interactions = []
        unsorted_selected_pos_i = []
        unsorted_selected_user_seqs = []
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, history_index, positive_u, positive_i = batched_data
            for i in range(len(interaction)):
                if interaction['user_id'][i].item() in self.selected_uids:
                    pr = self.selected_uids.index(interaction['user_id'][i].item())
                    unsorted_selected_interactions.append((interaction[i], pr))
                    unsorted_selected_pos_i.append((positive_i[i], pr))

                    test_item_seq = interaction[self.ITEM_SEQ][i]  # .detach().cpu().tolist()
                    test_item_seq_len = interaction[self.ITEM_SEQ_LEN][i]
                    test_real_len = min(self.config['max_his_len'], test_item_seq_len.item())
                    unsorted_selected_user_seqs.append((test_item_seq.detach().cpu().tolist()[:test_real_len], pr))
        unsorted_selected_interactions.sort(key=lambda t: t[1])
        unsorted_selected_pos_i.sort(key=lambda t: t[1])
        unsorted_selected_user_seqs.sort(key=lambda t: t[1])
        selected_interactions = [_[0] for _ in unsorted_selected_interactions]
        selected_pos_i = [_[0] for _ in unsorted_selected_pos_i]
        selected_user_seqs = [_[0] for _ in unsorted_selected_user_seqs]

        # user_matrix_aug_sim = self.multi_hot_similarity(selected_user_seqs, training_user_seqs)

        test_list_x = selected_user_seqs[:]
        train_list_x = training_user_seqs[:]


        test_user_list = []
        for i, test_seq in enumerate(test_list_x):
            item_hot_list = [0. for ii in range(len(item_text))]
            for item_pos in test_seq:
                item_hot_list[item_pos] = 1.
            test_user_list.append(item_hot_list)
        test_user_matrix = np.array(test_user_list)
        normalized_test_user_matrix = test_user_matrix

        train_user_list = []
        for i, train_seq in enumerate(train_list_x):
            item_hot_list = [0. for ii in range(len(item_text))]
            for item_pos in train_seq:
                item_hot_list[item_pos] = 1.
            train_user_list.append(item_hot_list)
        train_user_matrix = np.array(train_user_list)
        normalized_train_user_matrix = train_user_matrix  # / np.sum(train_user_matrix, axis=1)[:, np.newaxis]

        if not os.path.exists(os.path.join(self.config['data_path'], "output_files")):
            os.makedirs(os.path.join(self.config['data_path'], "output_files"))

        train_emb_file = os.path.join(self.config['data_path'], "output_files", f"train_multivector_emb.npy")
        test_emb_file = os.path.join(self.config['data_path'], "output_files", f"test_multivector_emb.npy")
        train_emb = normalized_train_user_matrix
        test_emb = normalized_test_user_matrix

        np.save(train_emb_file, train_emb)
        np.save(test_emb_file, test_emb)
        print("saving@@@saseec", train_emb_file)
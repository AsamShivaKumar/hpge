# for loading the data into files which is later used by input.py to generate training dataset

import numpy as np
from tqdm import tqdm

NEG_SAMPLING_POWER = 0.7

class DataLoader:

    def __init__(self, file_path, nbr_size, neg_size, num_edge_types, delim, sample_type):
        self.nbr_size = nbr_size
        self.neg_size = neg_size
        self.num_edge_types = num_edge_types
        self.sample_type = sample_type

        self.graph = {} # map<sid, map<edge_type,[[tid1,timestamp1],[tid2,timestamp2], ...]
        self.nodes = {} # map<node_type, map<node_id,count>>

        self.sids = []
        self.tids = []
        self.s_types = []
        self.t_types = []
        self.e_types = []
        self.time_stamps = []
        
        self.max_time = 0
        self.nodes_ids = {}
        
        with open(file_path) as file:
            while True:
                line = file.readline()
                if not line: break
                sid, s_type, tid, t_type, e_type, timestamp = [int(x)  for x in line.strip().split(delim)]        

                self.nodes.setdefault(s_type,{})
                self.nodes[s_type].setdefault(sid,0)
                self.nodes[s_type][sid] += 1
                self.nodes_ids.setdefault(sid,1)
                self.nodes_ids.setdefault(tid,1)

                self.max_time = max(timestamp,self.max_time)

                self.sids.append(sid)
                self.tids.append(tid)
                self.s_types.append(s_type)
                self.t_types.append(t_type)
                self.e_types.append(e_type)
                self.time_stamps.append(timestamp)

                # adding edge to the graph
                self.graph.setdefault(sid,{})
                self.graph[sid].setdefault(e_type,[])
                self.graph[sid][e_type].append([tid,timestamp])

        self.min_time = min(self.time_stamps)
        self.timespan = self.max_time - self.min_time
        self.node_size = len(self.nodes_ids.keys())
        self.no_of_edges = len(self.e_types)

        print(self.min_time, self.timespan, self.node_size, self.no_of_edges)

        for node_id, edges in self.graph.items():
            for edge_type, list_of_edges in edges.items():
                edges[edge_type] = sorted(list_of_edges, key = lambda x: x[1])
            self.graph[node_id] = edges
        
        self.neg_table = {}
        for node_type in self.nodes.keys():
            self.neg_table[node_type] = self.gen_neg_table(node_type)
        
        self.node_size_type = {}
        for n_type, node_list in self.nodes.items():
            self.node_size_type[n_type] = len(node_list)
    
    def gen_neg_table(self,n_type):
        nodes = self.nodes[n_type]
        node_ids = list(nodes.keys())
        node_degrees = list(nodes.values())
        tot_sum, cur_sum, por = 0., 0., 0.
        n_id = 0
        tot_sum = np.power(node_degrees, NEG_SAMPLING_POWER).sum()
        node_size = len(nodes)
        neg_table = np.zeros(node_size, )
        for k in range(node_size):
            if (k + 1.) / node_size > por:
                cur_sum += np.power(node_degrees[n_id], NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            neg_table[k] = node_ids[n_id - 1]
        return neg_table

    def negative_sampling(self, n_type):
        rand_idx = np.random.randint(0, self.node_size_type[n_type], (self.neg_size,))
        sampled_nodes = self.neg_table[n_type][rand_idx]
        sampled_nodes = ",".join(np.array(sampled_nodes, dtype=np.int).astype(np.str))
        return sampled_nodes

    def generate_whole_node_nbrs(self):

        # node_brs - map<node_id><e_type><timestamp> -> [ids,weights]
        node_nbrs = {}
        process_node = tqdm(total=len(self.graph))
        count = 0
        for nid, hete_nbrs in self.graph.items():
            count += 1
            if count % 100 == 0:
                process_node.update(100)
            temp = {}
            for e_type, e_list in hete_nbrs.items():
                temp[e_type] = {}
                sampled_nbrs = self.node_neighbor_sampling(np.array(e_list), e_list[-1][1])
                for _, [ids, weights, timestamp] in enumerate(sampled_nbrs):
                    temp[e_type][timestamp] = [ids, weights]
            node_nbrs[nid] = temp
        process_node.close()
        return node_nbrs

    def generate_training_dataset(self, filename, num_process=10):
        node_nbrs = self.generate_whole_node_nbrs()
        with open(filename, "w") as wf:

            # wf.write('str(e_type);str(sid);str(s_type);neg_s_nodes;' + ' ' + 'str(tid);str(t_type);neg_t_nodes') 

            process = tqdm(total=len(self.e_types))
            for i in range(len(self.e_types)):
                if (i + 1) % 10000 == 0:
                    process.update(10000)
                sid = self.sids[i]
                s_type = self.s_types[i]
                tid = self.tids[i]
                t_type = self.t_types[i]
                e_type = self.e_types[i]
                timestamp = self.time_stamps[i]

                neg_s_nodes = self.negative_sampling(s_type)
                neg_t_nodes = self.negative_sampling(t_type)
                s_hist_ids = ['' for _ in range(self.num_edge_types)]
                s_hist_weights = ['' for _ in range(self.num_edge_types)]
                s_hist_flags = ['-1' for _ in range(self.num_edge_types)]
                for et, sampled_nbrs in node_nbrs[sid].items():
                    if timestamp in sampled_nbrs:
                        temp_ids, temp_weights = sampled_nbrs[timestamp]
                        s_hist_ids[et] = temp_ids
                        s_hist_weights[et] = temp_weights
                        s_hist_flags[et] = '1'
                t_hist_ids = ['' for _ in range(self.num_edge_types)]
                t_hist_weights = ['' for _ in range(self.num_edge_types)]
                t_hist_flags = ['-1' for _ in range(self.num_edge_types)]
                for et, sampled_nbrs in node_nbrs[tid].items():
                    if timestamp in sampled_nbrs:
                        temp_ids, temp_weights = sampled_nbrs[timestamp]
                        t_hist_ids[et] = temp_ids
                        t_hist_weights[et] = temp_weights
                        t_hist_flags[et] = '1'
                outs = [str(e_type), str(sid), str(s_type), neg_s_nodes] + s_hist_ids + s_hist_weights + s_hist_flags + \
                       [str(tid), str(t_type), neg_t_nodes] + t_hist_ids + t_hist_weights + t_hist_flags
                train_info = ";".join(outs) + "\n"
                # 4 + 3 + 3*2*2
                wf.write(train_info)
            process.close()
        return train_info
    
    def node_neighbor_sampling(self, node_nbrs, t):
        # node_brs - list of [node_id, time_stamp] of all neighbours of a node with an edge type
        # t - timestamp of the most recent edge of the node
        if len(node_nbrs) == 0:
            return []
        else:
            times = node_nbrs[:, 1]
            ids = node_nbrs[:, 0]
            delta_t = (times - t) * 1.0 / self.timespan
            p = np.exp(delta_t)
            outs = self.importance_sampler(ids, p) if self.sample_type == 'important' else self.cutoff_sampler(ids, p)
            outs.append(t)
            new_t = node_nbrs[-1][1]
            new_node_nbr_idx = node_nbrs[(np.where(node_nbrs[:, 1] < new_t))]
            return [outs] + self.node_neighbor_sampling(new_node_nbr_idx, new_t)

    def importance_sampler(self, ids, p):
        uniq_ids, ids_index, ids_inverse = np.unique(ids, return_index=True, return_inverse=True)
        id_matrix = np.eye(len(uniq_ids), dtype=np.int)[ids_inverse]
        sum_uniq_p = np.dot(p, id_matrix).reshape(-1)  # 1 * d
        sum_uniq_q = sum_uniq_p ** 2
        norm_q = sum_uniq_q / np.sum(sum_uniq_q)
        sampled_ids = np.random.choice(np.arange(len(uniq_ids)), size=self.nbr_size, p=norm_q, replace=True)
        sp_ids, sp_counts = np.unique(sampled_ids, return_counts=True)
        weight = np.multiply((sum_uniq_p / norm_q)[sp_ids], sp_counts * 1.0 / self.nbr_size)
        norm_weight = weight / weight.sum()
        sp_node_ids = uniq_ids[sp_ids]
        return [','.join(sp_node_ids.astype(np.str)), ','.join(norm_weight.astype(np.str))]
    
    # selects nbr_size no. of nbrs from given ids based on time decay values - p
    # p values are in increasing order
    def cutoff_sampler(self, ids, p):
        # returns [sampled_node_ids,sampled_nodes_time_decay_vals]
        if self.nbr_size == 0:
            return ['', '']
        elif len(ids) < self.nbr_size:
            return [','.join(ids.astype(np.str)), ','.join(np.array(p).astype(np.str))]
        else:
            return [','.join(ids.astype(np.str)[len(ids) - self.nbr_size:]),
                    ','.join(np.array(p).astype(np.str)[len(ids) - self.nbr_size:])]
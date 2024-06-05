import pickle
import torch
import collections
from torch_geometric.data import InMemoryDataset, Data
import re
from torch_geometric.data import DataLoader
import numpy as np
from scipy.linalg import fractional_matrix_power
import pandas as pd

# calculate the distance between two places
def calc_dist_vec(longitudes1, latitudes1, longitudes2, latitudes2):
    """Calculate the distance (unit: km) between two places on earth, vectorised"""
    # convert degrees to radians
    lng1 = np.radians(longitudes1)
    lat1 = np.radians(latitudes1)
    lng2 = np.radians(longitudes2)
    lat2 = np.radians(latitudes2)
    radius = 6371.0088  # mean earth radius, en.wikipedia.org/wiki/Earth_radius#Mean_radius

    # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
    dlng = np.fabs(lng1 - lng2)
    dlat = np.fabs(lat1 - lat2)
    dist = 2 * radius * np.arcsin(np.sqrt(
        (np.sin(0.5 * dlat)) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(0.5 * dlng)) ** 2))
    return dist

class BuildGraph(InMemoryDataset):
    def __init__(self, root, phrase, transform=None, pre_transform=None):
        assert phrase in ['train', 'test']
        self.phrase = phrase
        super(BuildGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.phrase + '.txt']

    @property
    def processed_file_names(self):
        return [self.phrase + '.pt']

    def download(self):
        pass

    def process(self):
        x_all = []
        y = []
        nodes = {}  # {274:5037} poi_idx
        pair = {}
        senders_all = []
        receivers_all = []
        i = 0
        train_file = 'data/NYC/NYC_traj_train.txt'
        test_file = 'data/NYC/NYC_traj_test.txt'
        with open(train_file,'r') as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip()
                    uid = int(re.split('\t', line)[0])
                    poi_list = re.split('\t', line)[1:]
                    poi_list_id = [int(i) for i in poi_list]   # poi_lisi_id=[579,927]

                    # sequence = [1, 3, 2, 2, 1, 3, 4]
                    senders = []
                    x=[]
                    for node in poi_list_id:
                        if node not in nodes:
                            nodes[node] = i
                            x.append([node])
                            x_all.append(nodes[node]) # 改成索引
                            i += 1

                        senders.append(nodes[node])
                    receivers = senders[:]

                    if len(senders) != 1:
                        del senders[-1]  # the last item is a receiver
                        del receivers[0]  # the first item is a sender
                    for sender in senders:
                        senders_all.append(sender)
                    for receiver in receivers:
                        receivers_all.append(receiver)


            with open(test_file, 'r') as f:
                for line in f.readlines():
                    if len(line) > 0:
                        line = line.strip()
                        uid = int(re.split('\t', line)[0])
                        poi_list = re.split('\t', line)[1:]
                        poi_list_id = [int(i) for i in poi_list]  # poi_lisi_id=[579,927]

                        # sequence = [1, 3, 2, 2, 1, 3, 4]
                        senders = []
                        x = []
                        for node in poi_list_id[:-1]:
                            if node not in nodes:
                                nodes[node] = i
                                x.append([node])
                                x_all.append(nodes[node])
                                i += 1

                            senders.append(nodes[node])
                        receivers = senders[:]

                        if len(senders) != 1:
                            del senders[-1]  # the last item is a receiver
                            del receivers[0]  # the first item is a sender
                        for sender in senders:
                            senders_all.append(sender)
                        for receiver in receivers:
                            receivers_all.append(receiver)

            # print(nodes)

            # add geographical information
            poi_geo=collections.defaultdict(list)
            df = pd.read_csv('data/NYC/foursquare_nyc.geo', sep=',', header=None,
                             names=['geo_id', 'type', 'coordinates', 'venue_category_id', 'venue_category_name'],
                             encoding='ISO-8859-1')

            for poiid in nodes.keys():
                geo_str=df[df['geo_id'] == str(poiid)]['coordinates'].values[0]
                geo_list=geo_str.lstrip('[').rstrip(']').split(',')
                longitude=geo_list[0]
                latitude =geo_list[1]
                poi_geo[poiid].append(longitude)
                poi_geo[poiid].append(latitude)

            poi_list=list(poi_geo.keys())
            for i in range(len(poi_list) - 1):
                poiid=poi_list[i]
                next_poiid=poi_list[i+1]
                lon1 = float(poi_geo[poiid][0])
                lat1 = float(poi_geo[poiid][1])
                lon2 = float(poi_geo[next_poiid][0])
                lat2 = float(poi_geo[next_poiid][1])
                if calc_dist_vec(lon1, lat1, lon2, lat2) <= 0.3:
                    # print(poiid,next_poiid)
                    senders_all.append(nodes[poiid])
                    receivers_all.append(nodes[next_poiid])
                    senders_all.append(nodes[next_poiid])
                    receivers_all.append(nodes[poiid])

            sur_senders = senders_all[:]
            sur_receivers = receivers_all[:]
            j = 0
            for sender, receiver in zip(sur_senders, sur_receivers):
                if str(sender) + '-' + str(receiver) in pair:
                    pair[str(sender) + '-' + str(receiver)] += 1
                    del senders_all[i]
                    del receivers_all[i]
                else:
                    pair[str(sender) + '-' + str(receiver)] = 1
                    j += 1
            # print(pair)
            # print(nodes)

            edge_attr = torch.tensor(
                [pair[str(senders_all[i]) + '-' + str(receivers_all[i])] for i in range(len(senders_all))],
                dtype=torch.float)
            # print(edge_attr) 每条边权重

            edge_index = torch.tensor([senders_all, receivers_all], dtype=torch.long)
            # print(edge_index)

            x_all = torch.tensor(x_all, dtype=torch.long)
            graph = Data(x=x_all, edge_index=edge_index, edge_attr=edge_attr)
            data_list=[]
            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data

if __name__ == '__main__':

    G=BuildGraph("../", "train").process()
    print(G)
    #

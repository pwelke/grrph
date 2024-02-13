import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset, extract_tar
from urllib.request import urlretrieve
import os
from tqdm import tqdm

from ..converter import load_dimacs

# torch_geometric.seed_everything(2022)


class RIGIDDataset(InMemoryDataset):
    '''The dataset described in 
    https://www.lics.rwth-aachen.de/cms/LICS/Forschung/Publikationen/~rtok/Benchmark-Graphs/
    
    and the publication

    Pascal Schweizer, Daniel Neuen (2017):
    Benchmark Graphs for Practical Graph Isomorphism
    https://arxiv.org/abs/1705.03686

    '''

    datasets = {
        'cfi-rigid-z2' : 'https://www.lics.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcgajq',
        'cfi-rigid-r2' : 'https://www.lics.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcgalg',
        'cfi-rigid-s2' : 'https://www.lics.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcgaln',
        'cfi-rigid-t2' : 'https://www.lics.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcgalz',
        'cfi-rigid-z3' : 'https://www.lics.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcgama',
        'cfi-rigid-d3' : 'https://www.lics.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcgamx',
    }

    part_dict = {
        'cfi-rigid-d3': (0, 92),
        'cfi-rigid-r2': (92, 292),
        'cfi-rigid-s2': (292, 544), 
        'cfi-rigid-t2': (544, 806), 
        'cfi-rigid-z2': (806, 994), 
        'cfi-rigid-z3': (994, 1086)
    }

    def __init__(
        self,
        name="AACHEN",
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, self.name, name)
    
    @property
    def raw_dir(self):
        name = "raw"
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return [f'{d}.tar.gz' for d in self.datasets]

    @property
    def processed_file_names(self):
        return ['aachen.pt']

    def download(self) -> None:
        for subset in self.datasets:
            print(f'Downloading {subset} from {self.datasets[subset]}')
            urlretrieve(f'{self.datasets[subset]}', os.path.join(self.raw_dir, f'{subset}.tar.gz'))


    def process(self):

        part_dict = dict()
        data_list = list()
        offset = 0

        for f in sorted(self.datasets):
            extract_tar(os.path.join(self.raw_dir, f'{f}.tar.gz'), os.path.join(self.raw_dir, 'unzipped'))
            counter = 0
            print(f'converting {f}')
            for data in sorted(os.listdir(os.path.join(self.raw_dir, 'unzipped', f))):
                data_list.append(load_dimacs(os.path.join(self.raw_dir, 'unzipped', f, data)))
                num_nodes = data_list[-1].num_nodes
                perm = torch.randperm(num_nodes)
                data_list.append(Data(edge_index=perm[data_list[-1].edge_index], num_nodes=num_nodes))
                perm = torch.randperm(num_nodes)
                data_list.append(Data(edge_index=perm[data_list[-1].edge_index], num_nodes=num_nodes))
                perm = torch.randperm(num_nodes)
                data_list.append(Data(edge_index=perm[data_list[-1].edge_index], num_nodes=num_nodes))
                perm = torch.randperm(num_nodes)
                data_list.append(Data(edge_index=perm[data_list[-1].edge_index], num_nodes=num_nodes))
                perm = torch.randperm(num_nodes)
                data_list.append(Data(edge_index=perm[data_list[-1].edge_index], num_nodes=num_nodes))
                os.remove(os.path.join(self.raw_dir, 'unzipped', f, data))
                counter += 1
            part_dict[f] = (offset // 2, (offset + counter) // 2)
            offset += counter
            os.rmdir(os.path.join(self.raw_dir, 'unzipped', f))
            
        print(part_dict)
        os.rmdir(os.path.join(self.raw_dir, 'unzipped'))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def main():
    dataset = RIGIDDataset()
    print('number of graphs', len(dataset))
    ns = np.array([g.num_nodes for g in dataset])
    ms = np.array([g.num_edges for g in dataset])

    print(f'n nodes \in [{np.min(ns)}, {np.max(ns)}]\nn edges \in [{np.min(ms)}, {np.max(ms)}]')
    print(f'mean nodes: {np.mean(ns)} +- {np.std(ns)}\nmean edges: {np.mean(ms)} +- {np.std(ms)}')

    for part, range in dataset.part_dict.items():
        print(f'{part} contains {range[1]-range[0]} pairs')


def test_dimacs(f):
    # for line in open(f):
    #     print(line)
    d = dimacs2list(open(f))
    print('n = ', d['n'])
    print('m = ', d['m'])
    print(d['edges'])
    e = load_dimacs(f)


if __name__ == "__main__":
    main()
    # test_dimacs('Data/AACHEN/raw/cfi-rigid-d3/cfi-rigid-d3-0180-01-1')


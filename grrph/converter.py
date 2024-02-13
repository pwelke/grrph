import networkx as nx


def graph6_to_pyg(x):
    '''convert graph6 format to pytorch_geometric object
    source: https://github.com/GraphPKU/BREC/blob/Release/base/BRECDataset_v3.py
    '''
    return from_networkx(nx.from_graph6_bytes(x))


def dimacs2list(file, offset=0):
    '''read dimacs 'edge' format into a dict {'n': ..., 'm': ..., 'edges': ...}.
    edges is a list of lists of length 2 and node indexing starts at offset'''
    edges = list()
    for line in file:
        if line[0] == 'e':
            tokens = line.split()
            edges.append([int(tokens[1])-1+offset, int(tokens[2])-1+offset])
            edges.append([int(tokens[2])-1+offset, int(tokens[1])-1+offset])
        elif line[0] == 'c':
            # skip comments
            continue
        elif line[0] == 'p':
            tokens = line.split()
            if tokens[1].lower() != 'edge':
                raise IOError(f'unknown problem instance: {line} in {file}')
            n = int(tokens[2])
            m = int(tokens[3])
        else:
            raise IOError(f'unknown line format: {line} in {file}')

    if (n == 0) or (m == 0):
        print(f'{file.name} has {n} nodes and {m} edges. Check, please')
    return {'n': n, 'm': m, 'edges': edges}


def load_dimacs(file):
    '''return a pytorch_geometric object representing the graph in the dimacs file'''
    with open(file, 'r') as f:
        g = dimacs2list(f, 0)

        data = dict()
        data['edge_index'] = torch.tensor(g['edges'], dtype=torch.int64).T

        G = Data.from_dict(data)
        G.num_nodes = g['n']
        
        return G
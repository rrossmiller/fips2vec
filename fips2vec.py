def fips2vec(dimensions=64, force_train=False, vizualize=False, workers=6, file_name='fips2vec'):
    import os

    if not os.path.exists('fips2vec.kv') or force_train:
        import pandas as pd
        import networkx as nx
        from node2vec import Node2Vec
        from tqdm import trange

        print('loading raw fips file')
        # source: https://www2.census.gov/geo/docs/reference/county_adjacency.txt
        with open('fips_adjacency_raw.txt') as f:
            raw_zips = f.readlines()

        # format the raw txt
        edges = []
        curr = ''
        names = []
        for i in trange(len(raw_zips)):
            row = raw_zips[i].replace('\n', '')
            if row[0] == '"':
                tmp = row.split('\t')
                curr = tmp[1]
                edges.append([curr, tmp[3]])
                names.append((tmp[1], tmp[0].replace('"','')))
                names.append((tmp[3], tmp[2].replace('"','')))
            else:
                row = row.split('\t')
                edges.append([curr, row[-1]])
                names.append((row[-1], row[-2].replace('"','')))
        
        names_df = pd.DataFrame(names, columns=['num', 'name'])
        names_df = names_df.drop_duplicates()
        names_df.to_csv('names.csv', index=False) 
        edges_df = pd.DataFrame(edges, columns=['src', 'tgt'])
        edges_df.to_csv('fips_adjacency.csv', index=False) #save the graph as a csv
        return
        g = nx.Graph()
        g.add_edges_from(edges_df.to_numpy())

        node2vec = Node2Vec(g, dimensions=dimensions, walk_length=30, num_walks=200, workers=workers)

        # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
        model = node2vec.fit(window=10, min_count=1,
                             batch_words=4)

        # Save embeddings for later use
        if file_name:
            model.wv.save_word2vec_format(file_name+'.kv')

        if vizualize:
            print(model.wv['53045'])
            print(model.wv.most_similar('53045'))
            viz = nx.nx_agraph.to_agraph(g)
            viz.layout()  # layout with default (neato)
            viz.draw("graph.png")  # draw png
    else:
        import gensim
        embedding = gensim.models.KeyedVectors.load_word2vec_format('fips2vec.kv')
        print(embedding['53045'])

def write_tsv(file_path='.'):
    import csv
    import pandas as pd

    with open('fips2vec.kv') as fin:
        file = fin.readlines()

    file.pop(0)
    vecs = []
    labels = ['FIPS\tName\n']
    names_df = pd.read_csv('names.csv') 
    for i in range(len(file)):
        splits = file[i].split()
        if len(names_df.loc[names_df.num == int(splits[0])].name) == 0:
            print(splits)
            print(names_df.loc[names_df.num == int(splits[0])].name)
        labels.append(f'{splits[0]}\t{names_df.loc[names_df.num == int(splits[0])].name.iloc[0]}\n')
        vecs.append(splits[1:])

    with open(f'{file_path}/fips_vecs.tsv', 'w') as fout:
        csv.writer(fout, delimiter='\t').writerows(vecs)

    labels[-1] = labels[-1].replace('\n','')
    print(labels[-1])
    with open(f'{file_path}/fips_labels.tsv', 'w') as fout:
        fout.writelines(labels)
		
if __name__ == '__main__':
  fips2vec(dimensions=200, force_train=True)
  write_tsv()
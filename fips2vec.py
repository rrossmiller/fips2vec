def fips2vec(dimensions=64, force_train=False, vizualize=False, file_name='fips2vec'):
    import os
    if not os.path.exists('fips2vec.kv') or force_train:
        import pandas as pd
        import networkx as nx
        from node2vec import Node2Vec
        # source: https://www2.census.gov/geo/docs/reference/county_adjacency.txt
        with open('fips_adjacency_raw.txt') as f:
            raw_zips = f.readlines()

        # format the raw txt
        edges = []
        curr = ''
        for i in range(len(raw_zips)):
            row = raw_zips[i].replace('\n', '')
            if row[0] == '"':
                tmp = row.split('\t')
                curr = tmp[1]
                edges.append([curr, tmp[3]])

            else:
                row = row.split('\t')
                edges.append([curr, row[-1]])

        edges_df = pd.DataFrame(edges, columns=['src', 'tgt'])
        edges_df.to_csv('fips_adjacency.csv', index=False) #save the graph as a csv
        g = nx.Graph()
        g.add_edges_from(edges_df.to_numpy())

        node2vec = Node2Vec(g, dimensions=dimensions, walk_length=30, num_walks=200, workers=6)

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


if __name__ == '__main__':
    fips2vec()
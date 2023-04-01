def load_flicker():
    #from dgl.data import RedditDataset

    # load reddit data
    #data = RedditDataset(self_loop=True)
    #g = data[0]
    #g.ndata['features'] = g.ndata['feat']
    #g.ndata['labels'] = g.ndata['label']
    #return g, data.num_classes
    from dgl.data import FlickrDataset
    dataset = FlickrDataset()
    g = dataset[0]
    # get node feature
    g.ndata['features'] = g.ndata['feat']
    # get node labels
    g.ndata['labels'] = g.ndata['label']
    return g, dataset.num_classes

#def load_240m():
    dataset = dgl.data.rdf.AMDataset()
    graph = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data['train_mask']
    test_mask = g.nodes[category].data['test_mask']
    label = g.nodes[category].data['label']
    
    
def load_yelp():
    from dgl.data import YelpDataset
    dataset = YelpDataset()
    g = dataset[0]
    # get node feature
    g.ndata['features'] = g.ndata['feat']
    # get node labels
    g.ndata['labels'] = g.ndata['label']
    return g, dataset.num_classes
   
   
def load_reddit():
    from dgl.data import RedditDataset
    dataset = RedditDataset()
    g = dataset[0]
    # get node feature
    g.ndata['features'] = g.ndata['feat']
    # get node labels
    g.ndata['labels'] = g.ndata['label']
    return g, dataset.num_classes
#http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

with open('comp_change_proba_sorted.pkl','rb') as handle:
    proba_sorted=pickle.load(handle)

dict_plot={}
states=[]
for key, sub_dict in proba_sorted.items():
    states.append(key)
    for sub_key, value in sub_dict.items():
        states.append(sub_key)
        keys=(key, sub_key)
        dict_plot[keys]=value
states=list(set(states))


G=nx.MultiDiGraph()
G.add_nodes_from(states)
for k, v in dict_plot.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
print(f'Edges:')
#pprint(G.edges(data=True))    

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos)
print('draw')
# create edge labels for jupyter plot but is not necessary
edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
print('start to dot file...')
nx.drawing.nx_pydot.write_dot(G, 'third_run.dot')


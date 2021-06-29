import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
from networkx import *
import matplotlib.pyplot as plt


sns.set()


def distance(s_lat, s_lng, e_lat, e_lng):
    # approximate radius of earth in km
    r = 6373.0
    s_lat = s_lat * np.pi / 180.0
    s_lng = np.deg2rad(s_lng)
    e_lat = np.deg2rad(e_lat)
    e_lng = np.deg2rad(e_lng)
    d = np.sin((e_lat - s_lat) / 2) ** 2 + np.cos(s_lat) * np.cos(e_lat) * np.sin((e_lng - s_lng) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(d))


def create_graph(df, auth, auth_lat, auth_long):
    g = nx.Graph()
    df_copy = df[(df.Centername == auth)].copy().reset_index()
    source = df_copy['identity'].tolist()
    authlist = [auth, ]
    source.extend(authlist)

    g.add_nodes_from(source)
    for index, c in df_copy.iterrows():
        g.add_edge(auth, c['identity'], weight=distance(auth_lat, auth_long, c['Lat'], c['Long']))

    for cindex, c in df_copy.iterrows():
        for cindex1, c1 in df_copy.iterrows():
            if c['identity'] == c1['identity']:
                continue
            g.add_edge(c['identity'], c1['identity'], weight=distance(c['Lat'], c['Long'], c1['Lat'], c1['Long']))
    nx.draw(g)
    plt.savefig("{}.png".format(auth))
    return g


def graph_list(centers, df_final):
    dict_of_graphs = {}
    for index, a in centers.iterrows():
        g = create_graph(df_final, a['Centername'], a['Center Lat'], a['Center Long'])
        dict_of_graphs[a['Centername']] = g
    return dict_of_graphs


def find_tsp(centers, df_tot, df_final):
    dict_g = graph_list(centers, df_final)
    df_path = pd.DataFrame(columns=['lat', 'long', 'auth_name'])
    for source, g in dict_g.items():
        path = find_best_path(g)
        print(path)
        for index in range(len(path)):
            df = {'lat': df_tot[(df_tot.identity == path[index])].iloc[0].Lat,
                  'long': df_tot[(df_tot.identity == path[index])].iloc[0].Long,
                  'auth_name': source}
            temp_df = pd.DataFrame([df])
            df_path = pd.concat([df_path, temp_df], ignore_index=True)

    return df_path


def find_best_path(g):
    global smallestdis, best_tsp_path
    all_tsp_paths = {}
    for source in g.nodes:
        path_calc = list(g.nodes)
        path_calc.remove(source)
        path = [source, ]
        dis, path = find_path(g, source, source, path, path_calc)
        all_tsp_paths[dis] = path
        smallestdis = list(all_tsp_paths.keys())[0]
        best_tsp_path = all_tsp_paths[smallestdis]
    for dis in all_tsp_paths.keys():
        if dis < smallestdis:
            best_tsp_path = all_tsp_paths[dis]
    return best_tsp_path


def find_path(g, gsource, source, path, path_calc, totdis=0):
    if len(path_calc) == 1:
        path.append(path_calc[0])
        path.append(gsource)
        totdis = totdis + nx.single_source_dijkstra(g, gsource, path_calc[0])[0]
        return totdis, path
    closest_node = path_calc[0]
    dis = nx.single_source_dijkstra(g, source, closest_node)[0]
    for node in path_calc:
        tempdis = nx.single_source_dijkstra(g, source, node)[0]
        if tempdis < dis:
            closest_node = node
            dis = tempdis
    path.append(closest_node)
    path_calc.remove(closest_node)
    totdis = totdis + dis
    totdis, path = find_path(g, gsource, closest_node, path, path_calc, totdis)
    return totdis, path


def cluster_data(df_cit, df_auth):
    km = KMeans(n_clusters=count_auth, random_state=101)
    km.fit(X=df_cit[["Lat", "Long"]])
    centers = pd.DataFrame(km.cluster_centers_, columns=["Center Lat", "Center Long"])
    centers["Cluster"] = centers.index
    df_cit["Cluster"] = km.labels_

    for index, c in centers.iterrows():
        clong = c['Center Long']
        clat = c['Center Lat']  # when you have space between the name
        ds = []
        for ind, auth in df_auth.iterrows():
            authlong = auth.Long
            authlat = auth.Lat
            distance_center = distance(clong, clat, authlong, authlat)
            ds.append(distance_center)
        idx = np.argmin(np.array(ds))

        centers.at[index, "Center Lat"] = df_auth.at[idx, "Lat"]
        centers.at[index, "Center Long"] = df_auth.at[idx, "Long"]
        centers.at[index, "Centername"] = df_auth.at[idx, "identity"]

    df = pd.merge(df_cit, centers)
    return df, centers


def get_dataframes(file_name):
    global count_auth
    df = pd.read_excel(file_name)
    for index, c in df.iterrows():
        if 'citizen' in c['identity']:
            df.at[index, "level"] = '1'
        elif 'shop' in c['identity']:
            df.at[index, "level"] = '2'
            count_auth = count_auth + 1
    df_return = df.copy()[['latitude', 'longitude', 'identity', 'level']]
    df_return = df_return.rename(columns={"longitude": "Long", 'latitude': "Lat", 'identity': "identity"})
    return df_return


file_name_cit = "D:\\Documents\\project\\citizen.xlsx"
file_name_auth = "D:\\Documents\\project\\shop.xlsx"
df_cit = get_dataframes(file_name_cit)
count_auth = 0
df_auth = get_dataframes(file_name_auth)
print(count_auth)
df_tot = pd.concat([df_cit, df_auth], ignore_index=True)
df_final, centers = cluster_data(df_cit, df_auth)
centers.drop_duplicates(subset="Centername", keep='first', inplace=True)
df_final.groupby('Centername')
df_final.to_excel("D:\\Documents\\project\\clustured dataset.xlsx")
G = nx.Graph()
df_path = find_tsp(centers, df_tot, df_final)
df_path.to_excel("D:\\Documents\\project\\output_dftest.xlsx")
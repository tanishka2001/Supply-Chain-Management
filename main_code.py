import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import seaborn as sns
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


def get_nxgraph(warehouse, vendors, customers, branch_colname, existing):
    vendors['vendor_key'] = 'vendor_' + vendors['City']
    source = vendors['vendor_key'].tolist()
    source.extend(warehouse['City'].tolist())

    g = nx.Graph()
    g.add_nodes_from(source)

    for vindex, v in vendors.iterrows():
        for windex, w in warehouse[(warehouse.Level == 1)].iterrows():
            g.add_edge(v['vendor_key'], w['City'], weight=distance(v['Lat'], v['Lon'], w['Lat'], w['Lon']))

    for windex, m in warehouse[(warehouse.Level < 3)].iterrows():
        for rindex, r in warehouse.iterrows():
            g.add_edge(m['City'], r['City'], weight=distance(m['Lat'], m['Lon'], r['Lat'], r['Lon']))

    for mindex, m in warehouse[(warehouse.Level == 1)].iterrows():
        for rindex, r in warehouse[(warehouse.Level == 2)].iterrows():
            g.add_edge(m['City'], r['City'], weight=distance(m['Lat'], m['Lon'], r['Lat'], r['Lon']))

    if existing:
        for cindex, cu in customers.iterrows():
            g.add_edge(cu[branch_colname], 'customer_' + cu['City'],
                       weight=distance(cu['Lat'], cu['Lon'], cu[branch_colname + ' Lat'], cu[branch_colname + ' Lon']))
    else:
        for cindex, cu in customers.iterrows():
            g.add_edge(cu['Cluster City'], 'customer_' + cu['City'],
                       weight=distance(cu['Lat'], c['Lon'], cu[branch_colname + ' Lat'], cu[branch_colname + ' Lon']))
    return g


def get_mother_warehouse(warehouse, regionname):
    rdc = warehouse[(warehouse.Region == regionname) & (warehouse.Level < 3)]
    if len(rdc) == 1:
        if rdc.iloc[0].Level == 1:
            return rdc.iloc[0].City
        else:
            ds = []
            temp = warehouse[(warehouse.Level == 1)].copy().reset_index()
            for mindex, m in temp.iterrows():
                plat = rdc.iloc[0].Lat
                plong = rdc.iloc[0].Lon
                d = distance(m.Lat, m.Lon, plat, plong)
                ds.append(d)

            idx = np.argmin(np.array(ds))
            return temp.at[idx, "City"]


def get_inward_distance(warehouse, vendors):
    vendors = vendors.copy()
    for vindex, v in vendors.iterrows():
        ds = []
        tmp = warehouse[(warehouse.Level == 1)].copy().reset_index()
        for mindex, m in tmp.iterrows():
            plat = v.Lat
            plong = v.Lon
            d = distance(m.Lat, m.Lon, plat, plong)
            ds.append(d)

        idx = np.argmin(np.array(ds))

        vendors.at[vindex, 'Distance'] = ds[idx]

    return vendors  # returning vendors with distance column having closest distance with level 1 warehouses


def get_existing_outward_distance(warehouse, vendors, customers, branchcolname):
    vendors = get_inward_distance(warehouse, vendors)  # vendor with distance from closest level 1 warehouses
    average_inward_distance = sum(vendors['Demand'] * vendors['Distance']) / sum(vendors['Demand'])
    customer = customers.copy()
    network_graph = get_nxgraph(warehouse, vendors, customers, branchcolname, True)
    for index, c in customer.iterrows():
        mw = get_mother_warehouse(warehouse, c.Region)
        target1 = "customer_" + c.City
        m = nx.shortest_path(network_graph, mw, target1, weight='distance')
        totaldis = 0
        for ran in range(len(m) - 1):
            totaldis = nx.dijkstra_path_length(network_graph, m[0], 'customer_' + c.City, weight='weight')
        customer.at[index, 'end_to_end_existing_distance'] = totaldis + average_inward_distance

    return customer


def get_current_outward_distance(warehouse, vendors, customers, branchcolname):
    vendors = get_inward_distance(warehouse, vendors)
    average_inward_distance = sum(vendors['Demand'] * vendors['Distance']) / sum(vendors['Demand'])
    customer = customers.copy()
    network_graph = get_nxgraph(warehouse, vendors, customers, branchcolname, False)
    for index, c in customer.iterrows():
        mw = get_mother_warehouse(warehouse, c.Region)
        m = nx.shortest_path(network_graph, mw, 'customer_' + c.City, weight='distance')
        totaldis = 0
        #  for ran in range(len(m) - 1):
        #    totaldis = totaldis + nx.single_source_dijkstra(network_graph, m[ran], m[ran + 1])[0]
        totaldis = nx.dijkstra_path_length(network_graph, m[0], 'customer_' + c.City, weight='weight')
        customer.at[index, 'end_to_end_current_distance'] = totaldis + average_inward_distance
    return customer


def get_mdc(df_vendors, df_customers, start_point):
    centers_old = None
    for rand in range(101, 103):
        df_cus = df_customers.copy()[['Lat', 'Lon', 'Demand']]  # new dataframe with 3 columns from vendor dataframe
        df_cus['Type'] = 0
        df_ven = df_vendors.copy()[['Lat', 'Lon', 'Demand']]
        df_ven['Type'] = 1
        df_cus['Demand'] = df_cus['Demand'] / 100
        df = pd.concat([df_ven, df_cus])
        km = KMeans(n_clusters=start_point, random_state=rand)  # kmeans object number of centroid, randomstate to make clusturing reusable
        km.fit(X=df[["Lat", "Lon"]], sample_weight=df["Demand"])  # df passes is dataset to work on, weight of every observation
        df['Cluster'] = km.labels_  # label of each pt
        centers = pd.DataFrame(km.cluster_centers_, columns=["Lat", "Lon"])  # lat and long of clusters
        df_demand = df[df.Type == 1].copy()
        centers["Cluster"] = centers.index  # cluster column with index in centers dataframe
        for index, c in centers.iterrows():  # iterating through the 2 centers
            clat = c["Lat"]
            clong = c['Lon']
            ds = []
            for ind, p in df_customers.iterrows():  # iterating through the customer dataset
                plat = p['Lat']
                plong = p['Lon']
                d = distance(clat, clong, plat, plong)
                ds.append(d)  # distance from every customer to the center is appended

            idx = np.argmin(np.array(ds))

            centers.at[index, "Lat"] = df_customers.at[idx, "Lat"]  # putting centers as the nearest customer
            centers.at[index, "Lon"] = df_customers.at[idx, "Lon"]
            centers.at[index, "Region"] = df_customers.at[idx, "Region"]
            centers.at[index, "City"] = df_customers.at[idx, "City"]
            centers.at[index, "Demand"] = sum(df_demand[df_demand.Cluster == c.Cluster].Demand)  # summing demand of every vendor with a particular center
            centers.at[index, "Level"] = 1
        if centers_old is None:
            centers_old = centers.copy()
        else:
            centers_new = centers.copy()
            newdistance = 0
            olddistance = 0
            for index2, c2 in centers_old.iterrows():  # iterating through the 2 centers
                clat = c2["Lat"]
                clong = c2['Lon']
                for ind, p in df_vendors.iterrows():  # iterating through the customer dataset
                    plat = p['Lat']
                    plong = p['Lon']
                    d = distance(clat, clong, plat, plong)
                    olddistance = olddistance + d

            for index1, c1 in centers_new.iterrows():  # iterating through the 2 centers
                clat = c1["Lat"]
                clong = c1['Lon']
                for ind, p in df_vendors.iterrows():  # iterating through the customer dataset
                    plat = p['Lat']
                    plong = p['Lon']
                    d = distance(clat, clong, plat, plong)
                    newdistance = newdistance + d

            if newdistance < olddistance:
                centers_old = centers_new
    return centers_old  # contains coordinates of customer nearest to the vendor centers


def get_rdc(df_centers, df_customers, start_point):
    df = df_customers[~df_customers.Region.isin(df_centers.Region)].copy()[['Lat', 'Lon', 'Demand']]  # removing customers already linked to level 1 centers
    km = KMeans(n_clusters=start_point, random_state=101)
    km.fit(X=df[["Lat", "Lon"]], sample_weight=df["Demand"])  # clusturing according to customers in west and south region
    df['Cluster'] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=["Lat", "Lon"])
    centers["Cluster"] = centers.index

    for index, c in centers.iterrows():
        clat = c["Lat"]
        clong = c['Lon']
        ds = []
        for ind, p in df_customers.iterrows():
            plat = p['Lat']
            plong = p['Lon']
            d = distance(clat, clong, plat, plong)
            ds.append(d)

        idx = np.argmin(np.array(ds))
        # everything is same as rdc only clusturing on basis of customers and level = 2
        centers.at[index, "Lat"] = df_customers.at[idx, "Lat"]
        centers.at[index, "Lon"] = df_customers.at[idx, "Lon"]
        centers.at[index, "Region"] = df_customers.at[idx, "Region"]
        centers.at[index, "City"] = df_customers.at[idx, "City"]
        centers.at[index, "Demand"] = sum(df[df.Cluster == c.Cluster].Demand)
        centers.at[index, "Level"] = 2

    return centers


def get_cluster(df_customers, df_dcs, start_point, rand):
    df_old = None
    for rand in range(rand, rand+1):  # random state variable
        df = df_customers.copy()

        km = KMeans(n_clusters=start_point, random_state=rand)  # no of centers is k (10 to 30)
        km.fit(X=df[["Lat", "Lon"]], sample_weight=df["TotalDemand"])  # dataset is all customers

        df["Cluster"] = km.labels_
        centers = pd.DataFrame(km.cluster_centers_, columns=["Cluster Lat", "Cluster Lon"])  # k clustured warehouse locations
        centers["Cluster"] = centers.index
        for index, c in centers.iterrows():
            clat = c["Cluster Lat"]
            clong = c['Cluster Lon']
            ds = []
            for ind, p in df_customers.iterrows():
                plat = p['Lat']
                plong = p['Lon']
                d = distance(clat, clong, plat, plong)
                ds.append(d)
            idx = np.argmin(np.array(ds))

            centers.at[index, "Cluster Lat"] = df_customers.at[idx, "Lat"]
            centers.at[index, "Cluster Lon"] = df_customers.at[idx, "Lon"]
            centers.at[index, "Cluster City"] = df_customers.at[idx, "City"]  # marking k warehouses as the closest customer location

        for index, c in df_dcs.iterrows():  # the nearest warehouses to 4 (level 1 and 2) distribution centers are shifted to those 4 DCs
            clat = c["Lat"]
            clong = c['Lon']
            ds = []
            for ind, p in centers.iterrows():
                plat = p['Cluster Lat']
                plong = p['Cluster Lon']
                d = distance(clat, clong, plat, plong)
                ds.append(d)
            idx = np.argmin(np.array(ds))

            centers.at[idx, "Cluster Lat"] = c.Lat
            centers.at[idx, "Cluster Lon"] = c.Lon
            centers.at[idx, "Cluster City"] = c.City

        df = pd.merge(df, centers)  # merging df and centres on cluster column
        df["Distance"] = distance(df["Lat"], df["Lon"], df["Cluster Lat"], df["Cluster Lon"])
        df["Allowed Distance"] = df["Distance"] <= df["Max Distance"]  # allowance

        if df_old is None:
            df_old = df.copy()
        else:
            df_new = df.copy()  # total 3 times clusturing is run and using this df_old at the end holds the dataframe with maximum allowed distances
            new_selection = (df_new['Allowed Distance'] == True)
            old_selection = (df_old['Allowed Distance'] == True)
            if len(df_new[new_selection]) > len(df_old[old_selection]):
                df_old = df_new
    return df_old


def execute_model(df_customers, df_dcs, k, rand):
    df_final = None
    df_visualization = get_cluster(df_customers, df_dcs, k, rand)  # returns k optimized warehouse locations k -> (10,30)
    df_visualization['solution'] = k+4
    df_visualization["Cluster"] = df_visualization["Cluster"] + 1  # why?
    if df_final is None:  # use of this if else?
        df_final = df_visualization.copy()
    else:
        df_final = pd.concat([df_final, df_visualization], sort=False)

    df_final = df_final.reset_index()  # why

    return df_final


def get_dataframes(filepath):
    xl_file = pd.ExcelFile(filepath)  # reading excel
    dfs = {sheet_name: xl_file.parse(sheet_name)  # reading different sheets
           for sheet_name in xl_file.sheet_names}

    df_warehouse = dfs["Warehouse"]  # dataframes for all three sheets
    df_vendors = dfs["Vendor"]
    df_customers = dfs["Customer"]

    df_customers["Max Distance"] = (df_customers["SLA"] - 12) * 100 / 24
    df_customers["Current DC Distance"] = distance(df_customers["Branch Lat"], df_customers["Branch Lon"],
                                                   df_customers["Lat"], df_customers["Lon"])
    df_customers["Current SLA"] = df_customers["Current DC Distance"] <= df_customers["Max Distance"]

    return df_warehouse, df_vendors, df_customers


FILE_NAME = "D:\\Documents\\project\\Initial_Data.xlsx"

df_warehouse, df_vendors, df_customers = get_dataframes(FILE_NAME)

df_mdc = get_mdc(df_vendors, df_customers, 2)
df_rdc = get_rdc(df_mdc, df_customers, 2)

df_dcs = pd.concat([df_mdc, df_rdc])  # concatinating mdc and rdc total 4 areas

for index, c in df_customers.iterrows():  # use of this function?
    df_customers.at[index, "TotalDemand"] = c.Demand / c.SLA

df_visulation = None
for k in range(10, 31):
    df_old = None
    for rand in range(101, 110):
        df_final = execute_model(df_customers, df_dcs, k, rand)  # executing models for k total warehouses
        df_final.drop(columns=['TotalDemand'])

        for index, c in df_final.iterrows():
            if len(df_warehouse[df_warehouse.City == c.Branch]) > 0:
                tmp = df_warehouse[df_warehouse.City == c.Branch].reset_index()
                df_final.at[index, "existing_level"] = tmp.at[0, 'Level']

        new_warehouse = pd.DataFrame({'count': df_final.groupby(['Cluster Lat', 'Cluster Lon']).size()}).reset_index()
        new_warehouse.columns = ['Lat', 'Lon', 'Count']
        new_warehouse = new_warehouse.drop(columns=['Count'])

        for index, c in new_warehouse.iterrows():
            if len(df_dcs[(df_dcs.Lat == c.Lat) & (df_dcs.Lon == c.Lon)]) > 0:
                tmp = df_dcs[(df_dcs.Lat == c.Lat) & (df_dcs.Lon == c.Lon)].reset_index()
                new_warehouse.at[index, "Level"] = tmp.at[0, 'Level']
            else:
                new_warehouse.at[index, "Level"] = 3

            if len(df_customers[(df_customers.Lat == c.Lat) & (df_customers.Lon == c.Lon)]) > 0:
                tmp = df_customers[(df_customers.Lat == c.Lat) & (df_customers.Lon == c.Lon)].reset_index()
                new_warehouse.at[index, "City"] = tmp.at[0, 'City']
                new_warehouse.at[index, "Region"] = tmp.at[0, 'Region']

        for index, c in df_final.iterrows():
            if len(new_warehouse[(new_warehouse.Lat == c['Cluster Lat']) & (new_warehouse.Lon == c['Cluster Lon'])]) > 0:
                tmp = new_warehouse[
                    (new_warehouse.Lat == c['Cluster Lat']) & (new_warehouse.Lon == c['Cluster Lon'])].reset_index()
                df_final.at[index, "current_level"] = tmp.at[0, 'Level']

        df_final['existing_level'] = df_final['existing_level'].fillna(3).astype(int)
        df_final['current_level'] = df_final['current_level'].fillna(3).astype(int)
        new_warehouse['Level'] = new_warehouse['Level'].fillna(3).astype(int)
        df_final = get_existing_outward_distance(df_warehouse, df_vendors, df_final, 'Branch')
        df_final = get_current_outward_distance(new_warehouse, df_vendors, df_final, 'Branch')

        if df_old is None:
            df_old = df_final.copy()
            df_new = df_old.copy()
        else:
            df_new = df_final.copy()
        old_distance = sum(df_old.end_to_end_current_distance)
        new_distance = sum(df_new.end_to_end_current_distance)
        if new_distance < old_distance:
            df_old = df_new

    if df_visulation is None:
        df_visulation = df_old.copy()
    else:
        df_visulation = pd.concat([df_visulation, df_old], sort=False)
    print('Completed:', k)

df_visulation.to_excel("D:\\Documents\\project\\Output.xlsx")

print('Complete!')
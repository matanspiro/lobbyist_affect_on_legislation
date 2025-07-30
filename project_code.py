import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import re

# in pycharm: pip install networkx==3.1, pip install scipy==1.8.0

def check_parts_in_string(split_string, names_iterable):
    for name in names_iterable:
        count = 0
        count += sum(1 for part in split_string if re.search(r'\b' + re.escape(part) + r'\b', name))
        if count >= 2:
            return name

    return ''.join(split_string)

def align_com_name(com):
    if 'וועדה המיוחדת' in com: # cases of specifically dedicated committees
        return 'מיוחדת'
    elif 'הכספים' in com:
        return 'כספים'
    elif 'החינוך' in com or 'התרבות' in com or 'הספורט' in com:
        return 'חינוך, תרבות וספורט'
    elif 'העבודה' in com or 'הרווחה' in com or 'הבריאות' in com:
        return 'עבודה, רווחה ובריאות'
    elif 'העלייה' in com or 'הקליטה' in com or 'התפוצות' in com:
        return 'עלייה, קליטה ותפוצות'
    elif 'הכלכלה' in com:
        return 'כלכלה'
    elif 'המדע' in com or 'הטכנולוגיה' in com:
        return 'מדע וטכנולוגיה'
    elif 'החוקה' in com or 'חוק' in com or 'משפט' in com:
        return 'חוקה, חוק ומשפט'
    elif 'החוץ' in com or 'הביטחון' in com or 'חוץ' in com or 'ביטחון' in com:
        return 'חוץ וביטחון'
    elif 'תלמידי ישיבות' in com:
        return 'תלמידי ישיבות'
    elif 'הכנסת' in com:
        return 'כנסת'
    elif 'מוכנות העורף' in com:
        return 'מוכנות העורף'
    elif 'החקירה הפרלמנטרית' in com:
        return 'חקירה פרלמנטרית'
    elif 'כוח אדם' in com:
        return 'כ"א בצה"ל'
    elif 'הפנים' in com or 'הגנת הסביבה' in com:
        return 'הפנים והגנת הסביבה'
    elif 'המסדרת' in com:
        return 'מסדרת'
    elif 'הילד' in com:
        return 'זכויות הילד'
    else:
        return com

def set_name_order(full_name):
    split_name = full_name.split()
    ordered_name = split_name[len(split_name) - 1] # grab first name
    for i in range(0, len(split_name) - 1): # rest of the name is already ordered
        ordered_name += (' ' + split_name[i])

    return ordered_name

# preprocessing

# committees and lobbyists
committees_df = pd.read_csv('committees_data.csv')
committees_df = committees_df.dropna()
new_rows = []
for idx, row in committees_df.iterrows(): # splitting joint committees
    if 'הוועדה המשותפת' in row['committee']:
        if 'ולוועדת' in row['committee']:
            values = row['committee'].split('ולוועדת')
        elif 'ושל' in row['committee']:
            values = row['committee'].split('ושל')
        else:
            continue

        for value in values:
            new_row = row.copy()
            new_row['committee'] = value
            new_rows.append(new_row)

committees_df = pd.DataFrame(new_rows)
committees_df['committee'] = committees_df['committee'].apply(lambda com: align_com_name(com)) # simplifying committees' names

# aligning lobbyists names from the committees_df and the lobbyists_df, to match the names specified in lobbyists_df
lobbyists_df = pd.read_csv('all_lobbyists_data.csv')
lobbyists_df = lobbyists_df.dropna()
lob_unique_lobbyists_names = lobbyists_df['lobbyist'].unique()
com_to_lob_names = set()
lobbyists_in_committees_df = committees_df[committees_df['title'].str.contains('שדלן')]
for idx, row in lobbyists_in_committees_df.iterrows():
    com_lobbyist_full_name = row['attendee'].split()
    lob_name = check_parts_in_string(com_lobbyist_full_name, lob_unique_lobbyists_names)
    com_to_lob_names.add(lob_name)

real_lobbyists_in_committees_df = pd.DataFrame(list(com_to_lob_names), dtype = str)
real_lobbyists_in_committees_df.columns = ['lobbyist'] + list(real_lobbyists_in_committees_df.columns[1:])

# keeping only lobbyists there were present in the sessions
for idx, row in lobbyists_df.iterrows():
    if row['lobbyist'] in real_lobbyists_in_committees_df['lobbyist'].values:
        continue
    else:
        lobbyists_df.drop(idx, inplace = True)

# knesset members and parties
knesset_members_df = pd.read_csv('knesset_members_and_parties.csv')
knesset_members_df['member'] = knesset_members_df.iloc[:, 0].apply(lambda val: val.split(',')[3])
knesset_members_df['party'] = knesset_members_df.iloc[:, 0].apply(lambda val: val.split(',')[4])
knesset_members_df = knesset_members_df.iloc[:, 1:3]
knesset_members_df = knesset_members_df.drop_duplicates(subset = 'member')

# keeping only knesset members there were present in the sessions
for idx, row in knesset_members_df.iterrows():
    if row['member'] in committees_df['attendee'].values:
        continue
    else:
        knesset_members_df.drop(idx, inplace = True)

# vouchers' companies
vouchers_df = pd.read_csv('vouchers_data.csv', encoding = 'windows-1255')
vouchers_df = vouchers_df.dropna()
vouchers_df = vouchers_df.set_axis(['party', 'voucher', 'country', 'settlement', 'date', 'amount', 'company'], axis = 1)
vouchers_df['voucher'] = vouchers_df['voucher'].apply(lambda name: set_name_order(name))
vouchers_df = vouchers_df.dropna(subset = 'company').reset_index(drop = True)
vouchers_df['amount'] = vouchers_df['amount'].str.replace(',', '').astype(float)
new_rows = []
for index, row in vouchers_df.iterrows():
    companies = row['company'].split(',')
    for company in companies:
        new_rows.append({'voucher': row['voucher'], 'company': company, 'party':row['party']})
vouchers_df = pd.DataFrame(new_rows)


# creating graph. nodes: committees, lobbyists, knesset members, parties, companies and vouchers
g = nx.DiGraph() # directed graph

# combining all nodes
all_nodes = pd.concat([committees_df['committee'], lobbyists_df['lobbyist'], knesset_members_df['member'], knesset_members_df['party'], lobbyists_df['client'], vouchers_df['voucher']])
unique_nodes = all_nodes.unique()
for node in unique_nodes:
    g.add_node(node)

# adding edges between committees and knesset members and between committees and lobbyists
for idx, row in committees_df.iterrows():
    if row['attendee'] in unique_nodes and (row['attendee'] in knesset_members_df['member'].values or row['attendee'] in lobbyists_df['lobbyist'].values): # catching knesset members and lobbyists
        if row['attendee'] in knesset_members_df['member'].values: # knesset member
            edge = (row['committee'], row['attendee'])
        else: # lobbyist
            edge = (row['attendee'], row['committee'])

        # if g.has_edge(*edge): # edge = member's attendance in a committee's session
        #     g.edges[edge]['weight'] += 1
        #
        # elif g.has_edge(edge[1], edge[0]): # different order of nodes within the edge
        #     g.edges[edge[1], edge[0]]['weight'] += 1

        if not g.has_edge(*edge):
            g.add_edge(*edge, weight = 1)

# adding edges between knesset members and parties
for idx, row in knesset_members_df.iterrows():
    edge = (row['member'], row['party'])
    if not g.has_edge(*edge):
        g.add_edge(*edge, weight = 1)

# adding edges between lobbyists and companies
for idx, row in lobbyists_df.iterrows():
    edge = (row['client'], row['lobbyist'])
    if not g.has_edge(*edge):
        g.add_edge(*edge, weight = 1)

# adding edges between vouchers and companies
for idx, row in vouchers_df.iterrows():
    if row['company'] in lobbyists_df['client'].values: # only companies that are clients of lobbyists there were present during the sessions we have info on
        edge = (row['voucher'], row['company'])
        if not g.has_edge(*edge):
            g.add_edge(*edge, weight = 1)

# adding edges between voucher and parties
for idx, row in vouchers_df.iterrows():
    if row['party'] in knesset_members_df['party'].values and row['company'] in lobbyists_df['client'].values: # only parties there were represented in the data, and only companies that are represented by attending lobbyists
        edge = (row['party'], row['voucher'])
        if not g.has_edge(*edge):
            g.add_edge(*edge, weight = 1)


# visualizing graph
nodes_to_remove = [node for node, degree in dict(g.degree()).items() if degree < 3]
g.remove_nodes_from(nodes_to_remove)

committees_labels = committees_df['committee'].unique()
lobbyists_labels = lobbyists_df['lobbyist'].unique()
companies_labels = lobbyists_df['client'].unique()
knesset_members_labels = knesset_members_df['member'].unique()
vouchers_labels = vouchers_df['voucher'].unique()
parties_labels = knesset_members_df['party'].unique()

def set_colors(committees_labels, lobbyists_labels, companies_labels, knesset_members_labels, vouchers_labels, parties_labels):
    types_of_labels = {
        'type1': committees_labels,
        'type2': lobbyists_labels,
        'type3': companies_labels,
        'type4': knesset_members_labels,
        'type5': vouchers_labels,
        'type6': parties_labels
    }

    color_mapping = {
        'type1': '#F51806',  # red
        'type2': '#009933',  # green
        'type3': '#FFA500',  # orange
        'type4': '#0000FF',  # blue
        'type5': '#800080',  # purple
        'type6': '#FFFF00'   # yellow
    }
    return types_of_labels, color_mapping

attributes = {}
types_of_labels, color_mapping = set_colors(committees_labels, lobbyists_labels, companies_labels, knesset_members_labels, vouchers_labels, parties_labels)
for node_type, labels in types_of_labels.items():
    color = color_mapping.get(node_type)
    for node in labels:
        attributes[node] = {'color':color} # set the color for the current type of node

nx.set_node_attributes(g, attributes)
pos = nx.spring_layout(g, scale = 12)

plt.figure(figsize = (20, 10))
nodes_colors = [attributes[node].get('color') for node in g.nodes()]
edges_weights = [g.edges[edge]['weight'] for edge in g.edges]
nx.draw_networkx(g, pos = pos, with_labels = False, node_color = nodes_colors, width = edges_weights)

# drawing the labels. reversing nodes' hebrew strings to show hebrew properly
# all_labels = {node:node[::-1] for node in committees_labels if node in g.nodes}
# all_labels.update({node:node[::-1] for node in lobbyists_labels if node in g.nodes})
#
# companies_labels_filtered = [node for node in companies_labels if node in g.nodes]
# all_labels.update({node:node[::-1] if not re.search(r'[^a-zA-Z]', node) else node for node in companies_labels_filtered})
#
# all_labels.update({node:node[::-1] for node in knesset_members_labels if node in g.nodes})
# all_labels.update({node:node[::-1] for node in vouchers_labels if node in g.nodes})
# all_labels.update({node: node[::-1] for node in parties_labels if node in g.nodes})
#
# nx.draw_networkx_labels(g, pos = pos, labels = all_labels, font_size = 12, font_color = 'black', font_weight = 'bold')
plt.show()

# investigating vouchers - searching using recursive function: depth-first search (DFS) algorithm
def find_cyclic_paths_dfs(graph, node, path, length):
    if length == 0 and node == path[0]:
        yield path
    elif length > 0:
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                yield from find_cyclic_paths_dfs(graph, neighbor, path + [neighbor], length - 1)

def cyclic_info(graph, c_length, c_nodes_to_keep):
    found_cyc = False
    for source in graph.nodes():
        paths = list(find_cyclic_paths_dfs(graph, source, [source], c_length - 1))
        c_nodes_to_keep.extend(paths)
        for c_path in paths:
            if c_path[-1] == source:
                found_cyc = True
                print(f"Cyclic path of length {c_length} from {source} to {source}:", c_path)
    if not found_cyc:
        print(f"Didn't find cyclic path of length {c_length}")
    return c_nodes_to_keep

def find_nonC_longest_paths_dfs(graph, max_path_length):
    longest_nonC_paths = []
    def dfs(dfs_node, chained_path):
        if len(chained_path) > 1 and len(chained_path) <= max_path_length + 1:
            longest_nonC_paths.append(chained_path)
        if len(chained_path) <= max_path_length:
            for neighbor in graph.neighbors(dfs_node):
                if neighbor not in chained_path:
                    dfs(neighbor, chained_path + [neighbor])

    for nonC_node in graph.nodes():
        dfs(nonC_node, [nonC_node])

    return longest_nonC_paths



nodes_to_keep = []
# find cyclic paths of sizes 5 or 6
for c_path_len in range(5, 7):
    nodes_to_keep.extend(cyclic_info(g, c_path_len, nodes_to_keep))

# find non-cyclic paths of lengths 5 or 6
for length in range(5, 7):
    longest_paths = find_nonC_longest_paths_dfs(g, length)
    nodes_to_keep.extend(longest_paths)
    print(f"Longest paths of length {length}:")
    for path in longest_paths:
        if len(path) > 4:
            print(" -> ".join(map(str, path)))
    print("\n")

# nodes_to_remove_2 = [node for node in g.nodes() if node not in nodes_to_keep]
# g.remove_nodes_from(nodes_to_remove_2)
#
# # visualizing new graph
# committees_labels = [node for node in g.nodes() if node in committees_df['committee'].unique()]
# lobbyists_labels = [node for node in g.nodes() if node in lobbyists_df['lobbyist'].unique()]
# companies_labels = [node for node in g.nodes() if node in lobbyists_df['client'].unique()]
# knesset_members_labels = [node for node in g.nodes() if node in knesset_members_df['member'].unique()]
# vouchers_labels = [node for node in g.nodes() if node in vouchers_df['voucher'].unique()]
# parties_labels = [node for node in g.nodes() if node in knesset_members_df['party'].unique()]
#
# attributes_2 = {}
# types_of_labels, color_mapping = set_colors(committees_labels, lobbyists_labels, companies_labels, knesset_members_labels, vouchers_labels, parties_labels)
# for node_type_2, labels_2 in types_of_labels.items():
#     color = color_mapping.get(node_type_2)
#     for node in labels_2:
#         attributes_2[node] = {'color':color} # set the color for the current type of node
#
# nx.set_node_attributes(g, attributes_2)
# pos_2 = nx.spring_layout(g, scale = 12)
#
# plt.figure(figsize = (20, 10))
# nodes_colors_2 = [attributes_2[node].get('color') for node in g.nodes()]
# edges_weights_2 = [g.edges[edge]['weight'] for edge in g.edges]
# nx.draw_networkx(g, pos = pos_2, with_labels = False, node_color = nodes_colors_2, width = edges_weights_2)
# plt.show()


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpmax, fpgrowth
import csv

def preprocess_file(filepath, text_column_name="text"):
    """
    Removes column with text, as the functions for frequent itemsets require just 0/1 or True/False.
    @arg filepath - name of file with data for frequent itemset mining (str)
    @arg text_column_name - name of column which contains text (str)
    """
    df = pd.read_csv(filepath)
    del df[text_column_name]
    with open(filepath, 'w') as f:
        df.to_csv(f, header=True, index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')


def filter_itemsets(frequent_itemsets, length=0, support=0.01, fname=None, sorted=True):
    """
    Filters the frequent itemsets as a query in a database.
    @arg frequent_itemsets - frequent itemsets returned by func from mlxtend.frequent_patterns (frozetset)
    @arg length - minimal number of items in itemset (int)
    @arg support - minimal support of itemset (float)
    @arg fname - array of names of items in itemsets to be included in the filtered set (array[str])
    @arg sorted - sorts the filtered itemsets by support descending (boolean)

    @return filtered and optionally sorted set of frequent itemset
    """
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    if fname is None:
        filtered = frequent_itemsets[(frequent_itemsets['length'] > length) & (frequent_itemsets['support'] >= support)]
    else:
        filter_string = "frequent_itemsets["
        for name in fname:
            filter_string += "(frequent_itemsets['itemsets'].astype(str).str.contains(\'{}\')) & ".format(name)

        filter_string += "(frequent_itemsets['length'] > {}) & (frequent_itemsets['support'] >= {})]".format(length, support)

        filtered = eval(filter_string)
    if sorted:
        filtered = filtered.sort_values(by=['support'], ascending=False)

    return filtered

def normalize_support(filepath ,itemset, item_name):
    """
    Returns support of itemset w.r.t. the apriori of the item in the whole dataset.
    @arg filepath - dataset as returned from preprocess_file (str)
    @arg itemset - chosen itemset, row from pandas df obtained as frequent_itemsets.iloc[0] (pandas series)
    @arg item_name - name of the item from the itemset (str)
    """
    df = pd.read_csv(filepath)
    frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
    filtered = filter_itemsets(frequent_itemsets, fname=[item_name])
    itemset_item_support = itemset['support']
    dataset_item_support = filtered[filtered['length'] == 1].iloc[0]['support']

    normalized = itemset_item_support/dataset_item_support

    return normalized


if __name__ == "__main__":
    filepath="data/MLdata-PeopleData-notext.csv"
    file2 = "data/predicted_data_notext.csv"
    df = pd.read_csv(filepath)
    df2 = pd.read_csv(file2)
    df = df.loc[df['company4'] == 1]
    frequent_itemsets = fpgrowth(df, min_support=0.01, use_colnames=True)
    filtered = filter_itemsets(frequent_itemsets, 1, 0.2)
    print("Len = {}, itemsets = {}".format(len(filtered), filtered))
    print(df.shape)
    #print(association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5))

    #print(normalize_support(filepath, filtered.iloc[7], "female"))
    # print(type(filtered.iloc[0]))



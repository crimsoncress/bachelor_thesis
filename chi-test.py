import sys

import pandas as pd


def count_item_support(df, topics_all=["temperature", "air", "acoustics",
                                                                                 "light", "facilities", "ergonomics",
                                                                                 "culture", "coffee_snacks", "focus",
                                                                                 "cleanliness", "design",
                                                                                 "relax", "meetings", "man", "woman",
                                                                                 "18-25", "26-35", "36-45", "46-60",
                                                                                 "60+", "company1", "company2",
                                                                                 "company3", "company4", "company5",
                                                                                 "company6", "company7", "company8",
                                                                                 "company9", "company10",
                                                                                 "company11", "company12", "company13",
                                                                                 "company14", "company15", "company16",
                                                                                 "company17"]):
    """
    Computes frequencies of single items in a file.
    @arg df - pandas dataframe for frequent itemset mining (str)
    @arg topics_all - names of columns to compute the frequency of (array[str])

    @return dictionary where dict[item] = item frequency
    @return array with topics that have non null frequency
    """
    apriori_topics = {}
    for t in topics_all:
        apriori_topics[t] = 0
        for index, row in df.iterrows():
            if row[t] == 1:
                apriori_topics[t] += 1
    topics = []
    for t in topics_all:
        if apriori_topics[t] != 0:
            topics.append(t)

    return apriori_topics, topics


def create_topic_pairs(topics, stopping_point=None):
    """
    Creates a set of Cartesian product of topics.
    @arg topics - array of topics to create the set of pairs from (array[str])
    @arg stopping_point - number of domain topics (if topics contain demographic values that are meaningless to
    find dependency for, stopping point will ensure that they will not be included in pairs) (int)

    @return array with topic pairs array[(topic1, topic2), ...]
    """

    if stopping_point is None:
        stopping_point = len(topics)

    topic_pairs = []
    j = 1
    for t in topics[:stopping_point]:
        for t2 in topics[j:]:
            topic_pairs.append((t, t2))
        j += 1

    return topic_pairs

def create_empty_2D_dict(topic_pairs, contigency_values=["11", "10", "01", "00"]):
    """
    Creates an empty dictionary for each topic with all binary combinations from a contingency table.

    @arg topic_pairs - as obtained from create_topic_pairs() (array[(topic1, topic2), ...] )
    @arg contingency_values - all binary combinations in a k-dimensional contingency table (for k=3
    len(contingency_values) == 8)

    @return empty structure to fill by other functions
    """
    empty_dict = {}
    for pair in topic_pairs:
        empty_dict[pair] = {}
        for val in contigency_values:
            empty_dict[pair][val] = 0

    return empty_dict

def fill_2D_table(topic_pairs):
    """
    Computes number of baskets falling into each cell in contingency table.

    @arg topic_pairs - as returned by create_topic_pairs()
    @return struct filled with support of events, e.g. matrix_values[("air", "design")]["10"] == number of rows where
    row["air"] == 1 and row["design"] == 0
    """
    matrix_values = create_empty_2D_dict(topic_pairs=topic_pairs, contigency_values=["11", "10", "01", "00"])
    for index, row in df.iterrows():
        for pair in topic_pairs:
            if row[pair[0]] == 1 and row[pair[1]] == 1:
                matrix_values[pair]["11"] += 1
            elif row[pair[0]] == 1 and row[pair[1]] == 0:
                matrix_values[pair]["10"] += 1
            elif row[pair[0]] == 0 and row[pair[1]] == 1:
                matrix_values[pair]["01"] += 1
            elif row[pair[0]] == 0 and row[pair[1]] == 0:
                matrix_values[pair]["00"] += 1

    return matrix_values


def compute_chi_squared_values(n, topic_pairs, item_frequencies, matrix_values, dimension=2):
    """
    For each pair of items in topic_pairs computes chi squared statistics.

    @arg n - number of samples in dataset (int)
    @arg topic_pairs - as returned by create_topic_pairs()
    @arg item_frequencies - as returned by count_item_support()
    @arg matrix_values - as returned by fill_2D_table()
    @arg dimension - dimension of the contingency table

    @return dictionary, where dict[(topic1, topic2)] = chi squared statistics value
    """
    chivalues = {}

    for pair in topic_pairs:
        chivalues[pair] = 0

        e_r1 = item_frequencies[pair[0]] * (item_frequencies[pair[1]] / n)
        e_r1 = e_r1 if e_r1 != 0 else 0.00000000001
        chivalues[pair] += ((matrix_values[pair]["11"] - e_r1) ** 2) / e_r1

        e_r2 = item_frequencies[pair[0]] * ((n - item_frequencies[pair[1]]) / n)
        e_r2 = e_r2 if e_r2 != 0 else 0.00000000001
        chivalues[pair] += ((matrix_values[pair]["10"] - e_r2) ** 2) / e_r2

        e_r3 = (n - item_frequencies[pair[0]]) * (item_frequencies[pair[1]] / n)
        e_r3 = e_r3 if e_r3 != 0 else 0.00000000001
        chivalues[pair] += ((matrix_values[pair]["01"] - e_r3) ** 2) / e_r3

        e_r4 = (n - item_frequencies[pair[0]]) * ((n - item_frequencies[pair[1]]) / n)
        e_r4 = e_r4 if e_r4 != 0 else 0.00000000001
        chivalues[pair] += ((matrix_values[pair]["00"] - e_r4) ** 2) / e_r4

    return chivalues


def compute_interest(n, topic_pairs, item_frequencies, matrix_values, dimension=2):
    """
        For each pair of items in topic_pairs computes interest for characterization of the dependence.
        The bigger the interest[pair] than 1 the bigger positive dependence. The closer to zero the bigger negative
        dependence.

        @arg n - number of samples in dataset (int)
        @arg topic_pairs - as returned by create_topic_pairs()
        @arg item_frequencies - as returned by count_item_support()
        @arg matrix_values - as returned by fill_2D_table()
        @arg dimension - dimension of the contingency table

        @return dictionary, where dict[(topic1, topic2)]["11"] = interest value of baskets containing topic1 and topic2
        """
    interest = create_empty_2D_dict(topic_pairs)

    for pair in topic_pairs:
        e_r1 = item_frequencies[pair[0]] * (item_frequencies[pair[1]] / n)
        e_r1 = e_r1 if e_r1 != 0 else 0.00000000001
        interest[pair]["11"] = matrix_values[pair]["11"] / e_r1

        e_r2 = item_frequencies[pair[0]] * ((n - item_frequencies[pair[1]]) / n)
        e_r2 = e_r2 if e_r2 != 0 else 0.00000000001
        interest[pair]["10"] = matrix_values[pair]["10"] / e_r2

        e_r3 = (n - item_frequencies[pair[0]]) * (item_frequencies[pair[1]] / n)
        e_r3 = e_r3 if e_r3 != 0 else 0.00000000001
        interest[pair]["01"] = matrix_values[pair]["01"] / e_r3

        e_r4 = (n - item_frequencies[pair[0]]) * ((n - item_frequencies[pair[1]]) / n)
        e_r4 = e_r4 if e_r4 != 0 else 0.00000000001
        interest[pair]["00"] = matrix_values[pair]["00"] / e_r4

    return interest


if __name__ == "__main__":
    filepath = "data/MLdata-PeopleData-notext.csv"
    df = pd.read_csv(filepath)
    df = df.loc[df['company17'] == 1]

    topics_all = ["temperature", "air", "acoustics", "light", "facilities", "ergonomics", "culture", "coffee_snacks",
                  "focus", "cleanliness", "design", "relax", "meetings", "male", "female", "18-25", "26-35", "36-45",
                  "46-60", "60+", "company1", "company2", "company3", "company4", "company5", "company6", "company7",
                  "company8", "company9", "company10", "company11", "company12", "company13", "company14", "company15",
                  "company16", "company17"]

    item_frequencies, topics = count_item_support(df, topics_all=topics_all)
    # print(item_frequencies)

    topic_pairs = create_topic_pairs(topics, stopping_point=13)
    # print(topic_pairs)
    matrix_values = fill_2D_table(topic_pairs)
    chi_values = compute_chi_squared_values(df.shape[0], topic_pairs, item_frequencies, matrix_values, dimension=2)
    # print(dict(sorted(chi_values.items(), key=lambda item: item[1], reverse=True)))
    interest = compute_interest(df.shape[0], topic_pairs, item_frequencies, matrix_values, dimension=2)
    # print(item_frequencies)

    for pair in topic_pairs:
        if chi_values[pair] > 3.9:
            print(pair, chi_values[pair])
        # if interest[pair]["11"] < 0.5:
        #     print(pair, interest[pair])

    # supported = [("temperature", "woman"), ("air", "woman"), ("facilities", "man"), ("temperature", "air"),
    #              ("facilities", "company17"), ("acoustics", "woman")]
    # for s in supported:
    #     print(s, chi_values[s], interest[s], matrix_values[s])

    # p = ('coffee_snacks', 'company17')
    # print(matrix_values[p], chi_values[p], interest[p])






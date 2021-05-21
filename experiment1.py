import pandas as pd
from sklearn.model_selection import train_test_split

from baka_classik import NLPTopicClassifier
import csv

if __name__ == "__main__":
    """
    Splits the file into 5 equal parts, trains classifier on 4/5 and saves the 1/5. Then creates file with 100% 
    classifier annotated data. This file is then used for itemset mining to determine if the differs significantly 
    from frequent itemsets found on manually annotated data.
    """
    file = "data/MLdata-PeopleData.csv"
    all_data = pd.read_csv(file)

    df1, df2 = train_test_split(all_data, train_size=0.8, random_state=0, shuffle=True)
    df1, df3 = train_test_split(df1, train_size=0.75, random_state=0, shuffle=True)
    df1, df4 = train_test_split(df1, train_size=4/6, random_state=0, shuffle=True)
    df1, df5 = train_test_split(df1, train_size=0.5, random_state=0, shuffle=True)

    print(df1.shape, df2.shape, df3.shape, df4.shape, df5.shape)

    sets = [[df2, df3, df4, df5], [df1, df3, df4, df5], [df1, df2, df4, df5], [df1, df2, df3, df5], [df1, df2, df3, df4]]
    train_sets = []
    for s in sets:
        train_sets.append(pd.concat(s))

    other_labels = ["man", "woman", "18-25", "26-35", "36-45", "46-60", "60+", "company1", "company2", "company3",
                    "company4", "company5","company6","company7","company8","company9","company10","company11",
                    "company12", "company13","company14","company15","company16","company17"]

    test_sets = [df1, df2, df3, df4, df5]
    data = []

    for trn, tst in zip(train_sets, test_sets):
        klasifikator = NLPTopicClassifier(train_pct=0.95, train_df=trn)
        klasifikator.train()

        print("Trained")

        for index, row in tst.iterrows():
            predictions = {}
            text = row["text"]
            #print("Now estimating {}".format(text))
            labels = klasifikator.label(text)
            predictions["text"] = text
            for t, i in zip(klasifikator.classifiers, range(13)):
                predictions[t] = labels[i]
            for other in other_labels:
                predictions[other] = row[other]
            data.append(predictions)

    print("Saving part")

    df = pd.DataFrame(data)
    with open('data/predicted_data.csv', 'w') as f:
        df.to_csv(f, header=True, index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')
import csv

import pandas as pd
from random import randint


class MatchMaker(object):
    def __init__(self):
        # read user's replies to questions from bot:
        self.df = pd.read_csv("data/button_answers.csv")
        self.names = self.df.iloc[:, 1].values
        self.questions = self.df.iloc[:, 2].values
        self.answers = self.df.iloc[:, 3].values
        self.timestamps = self.df.iloc[:, 4].values
        self.dates = self.df.iloc[:, 5].values
        self.answers2 = self.df.iloc[:, 6].values

        # load our questions for matching purposes
        df2 = pd.read_csv("preset_msgs/3_options_questions.csv")
        self.form_questions = df2.iloc[:, 1].values

        # Load our defined phrases: 
        df4 = pd.read_csv("data/sent_sentences.csv")
        self.hate = df4.iloc[:, 0].values
        self.hatesize = len(self.hate)


    def find_match(self, index):
        name = self.names[index]
        answer = self.answers[index]
        new_index = int(index)
        answer = str(answer)
        while new_index > 0:
            new_index -= 1
            if self.questions[new_index] in self.form_questions and name == self.names[new_index]:
                answer = str(answer)
                self.df.iat[new_index, 6] = answer
                break


    def sentences_maker(self, index, company):
        q = self.questions[index]
        sent = self.answers[index]
        idea = self.answers2[index]
        final_str = ""
        la = {'temp': 0, 'air': 0, 'acou': 0, 'lig': 0, 'facil': 0, 'ergo': 0, 'cult': 0, 'coffee': 0, 'focus': 0,
              'clean': 0, 'design': 0, 'relax': 0, 'meet': 0}


        if sent in ["2", "1"]:
            final_str += self.hate[randint(0, self.hatesize-1)]
        else:
            assert 'unknown button press {sent}'

        # manually match question q to defined questions -> type of Q -> expected topic
        # also set given class label
        if q == self.form_questions[0]:
            final_str += "the job overall, "
            la['cult'] = 1
        elif q == self.form_questions[1]:
            final_str += "the conditions for focused work, "
            la['focus'] = 1
        elif q == self.form_questions[2]:
            final_str += "acoustics, "
            la['acou'] = 1
        elif q == self.form_questions[3]:
            final_str += "temperature, "
            la['temp'] = 1
        elif q == self.form_questions[4]:
            final_str += "lightning conditions, "
            la['lig'] = 1
        elif q == self.form_questions[5]:
            final_str += "air quality, "
            la['air'] = 1
        elif q == self.form_questions[6]:
            final_str += "design of our offices, "
            la['design'] = 1
        elif q == self.form_questions[7]:
            final_str += "our chairs and tables, "
            la['ergo'] = 1
        elif q == self.form_questions[8]:
            final_str += "the tidiness of our office, "
            la['clean'] = 1
        elif q == self.form_questions[9]:
            final_str += "the relax zones, "
            la['relax'] = 1
        elif q == self.form_questions[10]:
            final_str += "the coffee and snacks, "
            la['coffee'] = 1
        elif q == self.form_questions[11]:
            final_str += "the decision making of our company, "
            la['cult'] = 1
        elif q == self.form_questions[12]:
            final_str += "the meetings value, "
            la['meet'] = 1


        final_str += str(idea) + ".  "
        data = [{'company': company, 'cs_text': "", 'en_text': final_str, 'temperature': la['temp'], 'air': la['air'],
                 'acoustics': la['acou'], 'light': la['lig'], 'facilities': la['facil'], 'ergonomics': la['ergo'],
                 'culture': la['cult'], 'coffee_snacks': la['coffee'], 'focus': la['focus'], 'cleanliness': la['clean'],
                 'design': la['design'], 'relax': la['relax'], 'meetings': la['meet'], 'reviewed': 0, 'todo': 1}]

        df = pd.DataFrame(data)
        with open('data/MLdata-ClassData.csv', 'a') as f:
            df.to_csv(f, header=False, index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')




if __name__ == "__main__":
    mmake = MatchMaker()
    index = 0
    # for q in mmake.questions:
    #     if q == 'How could we make it better?':
    #         mmake.find_match(index)
    #         with open('data/answers.csv', 'w') as f:
    #             mmake.df.to_csv(f, index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')
    #     index += 1


    for q in mmake.answers2:
        if q is not None and q != "" and type(q) != float:
            mmake.sentences_maker(index, 0)
        index += 1





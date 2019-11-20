import re
from nltk.corpus import stopwords


## Stop word
class stopword:

    def __init__(self):
        self.input = []
        self.output = []

        stop_words = set(stopwords.words('english'))
        filepath1= open("dataset.txt", errors='ignore')

        text_file1 = filepath1.read()
        stmt = text_file1.split("\n")

        for text in stmt:
            #removes from each line unwanted text
            catg = text[31:text.find(",")-1]
            if ("capacity and performance" == catg):
                self.output.append(0)
            if ("usability" == catg):
                self.output.append(1)
            if ("security" == catg):
                self.output.append(2)
            if ("operational" == catg):
                self.output.append(3)



            final=text[text.find("sentence")+len("sentence")+3:-3]
            final_text = [re.sub(r"[^a-zA-Z]+", ' ', k) for k in final.split("\n")]


            #removed stop words from strings

            querywords = final_text[0].split()

            resultwords  = [word for word in querywords if word not in stop_words]
            result = ' '.join(resultwords)

            self.input.append(result)


    def getInput(self):
        return self.input

    def getOutput(self):
        return self.output

s = stopword()





import re
a = [1,2,3,4,5]

class Bah:
    target_column = -1
    regex = re.compile(",")
    def __init__(self,line):
        self.data = Bah.regex.split(line)
        self.target = self.data[self.target_column]
        del self.data[self.target_column]

    def pout(self):
        print(self.data)
        print(self.target)
        print("--")

@target_col(2)
class Gah(Bah):
    pass

b = Bah("1,2,3,4,5")

c = Gah("1,2,3,4,52d")
b.pout()
c.pout()

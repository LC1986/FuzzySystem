"""
Requires networkx (>= 1.9.0,<2.0.0), scipy (>= 0.9.0), numpy (>= 1.6.0), scikit-fuzzy 0.3
"""

# import numpy
import pandas as pd
from DecisionMaker import DecisionMaker
# import matplotlib.pyplot as plt

###############################################################################

def main():

    # Read indicators data from csv if available
    customers = pd.read_csv("indicators.csv")

    # instantiate DecisionMaker
    myDecision = DecisionMaker(customers)

    # evaluate each record
    print('|-Stock decision Percentage-|')
    print("No Decision")
    for row,customer in zip(range(len(customers)),customers.iterrows()):
        consequents = myDecision.defuzzify(dict(customer[1]))
        print("%2i " % (row+1), end='')
        print("%.6f " % (consequents[0][0]))

    return 0

if __name__ == "__main__":
    main()

###############################################################################

"""
Result:
"""
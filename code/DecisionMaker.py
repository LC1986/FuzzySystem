"""
Requires networkx (>= 1.9.0,<2.0.0), scipy (>= 0.9.0), numpy (>= 1.6.0), scikit-fuzzy 0.3
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

###############################################################################
# constants

NUM_VARIABLES = 3  # 4 variables
DEBUGLEVEL = 0

###############################################################################

class DecisionMaker:

    # constructor
    def __init__(self, data):

        if self.checkData(data) == False:
            return

        self.fuzzify(data)
        self.inference()

    # ------------------------------------------------------------------------------
    # This method verifies the correctness of the data:
    # (1) checks correct number of variables
    # (2) checks at least one row of data exists
    #
    def checkData(self, data):

        if len(data.columns) < NUM_VARIABLES:
            print("Insufficient fuzzy variables")
            return False

        if len(data) == 0:
            print("No data found")
            return False

        self.dataheaders = data.columns
        return True

    # ------------------------------------------------------------------------------
    # This method performs the fuzzification:
    # (1) sets the fuzzy partitions of each linguistic variable and
    # (2) sets the membership function of each linguistic term of the variable
    #
    def fuzzify(self, data):

        # set up RSI
        x = np.arange(0, 101, 1)
        self.RSI = ctrl.Antecedent(x, data.columns[1])

        self.RSI['lo'] = fuzz.trimf(x, [0, 0, 30])
        self.RSI['me'] = fuzz.trimf(x, [20, 50, 90])
        self.RSI['hi'] = fuzz.trimf(x, [80, 100, 100])


        # set up MACD
        x = np.arange(0, 201, 1)
        self.MACD = ctrl.Antecedent(x, data.columns[2])
        self.MACD['lo'] = fuzz.trapmf(self.MACD.universe, [0, 0, 20, 30])
        self.MACD['me'] = fuzz.trapmf(self.MACD.universe, [30, 50, 80, 100])
        self.MACD['hi'] = fuzz.trapmf(self.MACD.universe, [80, 100, 200, 200])

        #self.MACD['lo'] = fuzz.trimf(x, [0, 0, 30])
        #self.MACD['me'] = fuzz.trimf(x, [20, 50, 90])
        #self.MACD['hi'] = fuzz.trimf(x, [80, 100, 100])

        # set up ADX
        x = np.arange(20, 101, 1)
        self.ADX = ctrl.Antecedent(x, data.columns[3])
        self.ADX['lo'] = fuzz.trapmf(self.ADX.universe, [20, 20, 30, 40])
        self.ADX['me'] = fuzz.trapmf(self.ADX.universe, [30, 40, 50, 60])
        self.ADX['hi'] = fuzz.trapmf(self.ADX.universe, [50, 60, 100, 100])

        # set up decision
        x = np.arange(0, 11, 1)
        self.decision = ctrl.Consequent(x, 'decision')
        self.decision['buy'] = fuzz.trapmf(self.decision.universe, [0, 0, 3, 4])
        self.decision['hold'] = fuzz.trapmf(self.decision.universe, [3, 4, 6, 7])
        self.decision['sell'] = fuzz.trapmf(self.decision.universe, [6, 7, 10, 10])


        if DEBUGLEVEL == 1:
            self.RSI.view()
            self.MACD.view()
            self.ADX.view()
            self.decision.view()
        return

    # ------------------------------------------------------------------------------
    # This method:
    # (1) sets the rule base
    # (2) sets the inference engine to use the rule base
    #
    def inference(self):

        # rules 01 - 27
        # RSI & MACD & ADX -> decision (buy, hold, sell)
        rule01 = ctrl.Rule(self.RSI['hi'] & self.MACD['hi'] & self.ADX['me'], self.decision['sell'])
        rule02 = ctrl.Rule(self.RSI['hi'] & self.MACD['me'] & self.ADX['me'], self.decision['sell'])
        rule03 = ctrl.Rule(self.RSI['hi'] & self.MACD['lo'] & self.ADX['me'], self.decision['sell'])
        rule04 = ctrl.Rule(self.RSI['me'] & self.MACD['hi'] & self.ADX['me'], self.decision['hold'])
        rule05 = ctrl.Rule(self.RSI['me'] & self.MACD['me'] & self.ADX['me'], self.decision['hold'])
        rule06 = ctrl.Rule(self.RSI['me'] & self.MACD['lo'] & self.ADX['me'], self.decision['hold'])
        rule07 = ctrl.Rule(self.RSI['lo'] & self.MACD['hi'] & self.ADX['me'], self.decision['buy'])
        rule08 = ctrl.Rule(self.RSI['lo'] & self.MACD['me'] & self.ADX['me'], self.decision['buy'])
        rule09 = ctrl.Rule(self.RSI['lo'] & self.MACD['lo'] & self.ADX['me'], self.decision['buy'])

        rule10 = ctrl.Rule(self.RSI['hi'] & self.MACD['hi'] & self.ADX['lo'], self.decision['sell'])
        rule11 = ctrl.Rule(self.RSI['hi'] & self.MACD['me'] & self.ADX['lo'], self.decision['sell'])
        rule12 = ctrl.Rule(self.RSI['hi'] & self.MACD['lo'] & self.ADX['lo'], self.decision['sell'])
        rule13 = ctrl.Rule(self.RSI['me'] & self.MACD['hi'] & self.ADX['lo'], self.decision['hold'])
        rule14 = ctrl.Rule(self.RSI['me'] & self.MACD['me'] & self.ADX['lo'], self.decision['hold'])
        rule15 = ctrl.Rule(self.RSI['me'] & self.MACD['lo'] & self.ADX['lo'], self.decision['hold'])
        rule16 = ctrl.Rule(self.RSI['lo'] & self.MACD['hi'] & self.ADX['lo'], self.decision['buy'])
        rule17 = ctrl.Rule(self.RSI['lo'] & self.MACD['me'] & self.ADX['lo'], self.decision['buy'])
        rule18 = ctrl.Rule(self.RSI['lo'] & self.MACD['lo'] & self.ADX['lo'], self.decision['buy'])

        rule19 = ctrl.Rule(self.RSI['hi'] & self.MACD['hi'] & self.ADX['hi'], self.decision['sell'])
        rule20 = ctrl.Rule(self.RSI['hi'] & self.MACD['me'] & self.ADX['hi'], self.decision['sell'])
        rule21 = ctrl.Rule(self.RSI['hi'] & self.MACD['lo'] & self.ADX['hi'], self.decision['sell'])
        rule22 = ctrl.Rule(self.RSI['me'] & self.MACD['hi'] & self.ADX['hi'], self.decision['hold'])
        rule23 = ctrl.Rule(self.RSI['me'] & self.MACD['me'] & self.ADX['hi'], self.decision['hold'])
        rule24 = ctrl.Rule(self.RSI['me'] & self.MACD['lo'] & self.ADX['hi'], self.decision['hold'])
        rule25 = ctrl.Rule(self.RSI['lo'] & self.MACD['hi'] & self.ADX['hi'], self.decision['buy'])
        rule26 = ctrl.Rule(self.RSI['lo'] & self.MACD['me'] & self.ADX['hi'], self.decision['buy'])
        rule27 = ctrl.Rule(self.RSI['lo'] & self.MACD['lo'] & self.ADX['hi'], self.decision['buy'])

        # set up control system for decision
        self.decisionCS = ctrl.ControlSystem(
            [rule01, rule02, rule03, rule04, rule05, rule06, rule07, rule08, rule09, rule10,
             rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20,
             rule21, rule22, rule23, rule24, rule25, rule26, rule27
             ])
        self.decisionEval = ctrl.ControlSystemSimulation(self.decisionCS)

        return

    # ------------------------------------------------------------------------------
    # This function evaluates the given data with inference engine
    # and defuzzifies the output
    # Returns decision
    #
    def defuzzify(self, data):

        self.decisionEval.input[self.dataheaders[1]] = data[self.dataheaders[1]]
        self.decisionEval.input[self.dataheaders[2]] = data[self.dataheaders[2]]
        self.decisionEval.input[self.dataheaders[3]] = data[self.dataheaders[3]]
        self.decisionEval.compute()
        decision = (self.decisionEval.output['decision']-5)/10

        consequents = [[]]
        consequents[0] = [decision]

        # return decision
        return consequents


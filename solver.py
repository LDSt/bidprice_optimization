import numpy as np
import scipy as sp
import pandas as pd
from tabulate import tabulate
from scipy.optimize import linprog
import datetime as dt
import math
import sys

sys.path.insert(0, './capacityCalculator')
sys.path.insert(0, './demandForecast')

import capacityCalculator
import demandForecast


class BidPriceCalculator():

    def __init__(self, inputFile, lookup, doi, dbd):
        print("Start of the program to find bid prices.")
        print()
        self.inputFile = inputFile
        overbookings_factor = 1
        self.dbd = dbd
        self.doi = doi
        inputPath = "input/" + self.inputFile
        print(inputPath)
        print("Days before departure variable is: ")
        print(self.dbd)
        print()
        print("The input file variable is: ")
        print(self.inputFile)
        print()
        c = capacityCalculator.CapacityCalculator()
        self.d = demandForecast.DemandForecast()
        if self.dbd < 10:
            if self.dbd == 9:
                capBCN_AMS = pd.read_csv("capacityCalculator/cap-bcn-ams.csv", index_col=0)
                capAMS_JFK = pd.read_csv("capacityCalculator/cap-ams-jfk.csv", index_col=0)
                self.bookings = pd.read_csv('demandForecast/bookings.csv', index_col=0)
                self.bookings['CancellationDateTime'] = pd.to_datetime(self.bookings['CancellationDateTime'],yearfirst=True)
            else:
                capacityFileIterator = int(self.doi[-2:]) - self.dbd - 1
                capBCN_AMS = pd.read_csv("capacityCalculator/cap-bcn-ams-" + str(capacityFileIterator) + ".csv")
                capAMS_JFK = pd.read_csv("capacityCalculator/cap-ams-jfk-" + str(capacityFileIterator) + ".csv")
                self.bookings = pd.read_csv("demandForecast/bookings-" + str(capacityFileIterator) + ".csv")
                self.bookings['CancellationDateTime'] = pd.to_datetime(self.bookings['CancellationDateTime'],yearfirst=True)
            dailyInput = pd.read_excel(inputPath)

            capBCN_AMS = c.update_cap_Bcn_Ams(capBCN_AMS, dailyInput)
            capAMS_JFK = c.update_cap_Ams_Jfk(capAMS_JFK, dailyInput)
            self.bookings = d.update_demand(self.bookings, dailyInput)

            capBCN_AMS.to_csv("capacityCalculator/cap-bcn-ams-" + inputFile[:2] + ".csv", index=False)
            capAMS_JFK.to_csv("capacityCalculator/cap-ams-jfk-" + inputFile[:2] + ".csv", index=False)
            self.bookings.to_csv("demandForecast/bookings-" + inputFile[:2] + ".csv", index=False)
        else:
            capBCN_AMS = pd.read_csv("capacityCalculator/cap-bcn-ams.csv")
            del capBCN_AMS['Unnamed: 0']
            del capBCN_AMS['Unnamed: 0.1']
            capAMS_JFK = pd.read_csv("capacityCalculator/cap-ams-jfk.csv")
            del capAMS_JFK['Unnamed: 0']
            del capAMS_JFK['Unnamed: 0.1']
            self.bookings = pd.read_csv('demandForecast/bookings.csv',index_col=0)
            self.bookings['CancellationDateTime'] = pd.to_datetime(self.bookings['CancellationDateTime'],yearfirst=True)



        self.flights_capacities = c.get_remaining_capacities(capBCN_AMS, "BCN-AMS", "2016-11-28", "2016-11-20")
        self.flights_capacities = self.flights_capacities + c.get_remaining_capacities(capAMS_JFK, "AMS-JFK", "2016-11-28", "2016-11-20")
        #hard code capacities to show working of the model
        self.fare_classes = []
        print("The current flight capacity displayed in order bcn_ams_20, bcn_ams_40, ams_jfk_20, ams_jfk_40: ")
        print(self.flights_capacities)
        print()
        bid_prices = self.get_bid_prices(self.flights_capacities, self.fare_classes)
        self.save_bidprices(bid_prices)

    def get_bid_prices(self, flights_capacities, fare_classes):
        bid_prices_list = self.exec_LP_model(flights_capacities, fare_classes)
        labels = [
            'BCN - AMS - 20',
            'BCN - AMS - 40',
            'AMS - JFK - 20',
            'AMS - JFK - 40'
        ]
        df_bid_prices = pd.DataFrame.from_records(
            [bid_prices_list], columns=labels)
        return df_bid_prices

    def exec_LP_model(self, capacities, fare_classes):
        demand_bcn_ams_business = self.d.super_func(self.bookings, 'BCN-AMS','Business',self.dbd, n_clusters=2)
        demand_bcn_ams_economy = self.d.super_func(self.bookings, 'BCN-AMS','Economy',self.dbd)
        demand_ams_jfk_business = self.d.super_func(self.bookings, 'AMS-JFK','Business',self.dbd)
        demand_ams_jfk_economy = self.d.super_func(self.bookings, 'AMS-JFK','Economy',self.dbd)

        demand_total = demand_bcn_ams_business + demand_bcn_ams_economy + \
        demand_ams_jfk_business + demand_ams_jfk_economy

        fares = [item[0] for item in demand_total]
        demand = [item[1] for item in demand_total]

        number_of_columns = len(fare_classes)

        b = capacities + demand
        c = list(map(int, fares))

        A_bcn_ams_b = np.concatenate(
            [np.ones(2), np.zeros(3), np.zeros(3), np.zeros(3)])
        A_bcn_ams_e = np.concatenate(
            [np.zeros(2), np.ones(3), np.zeros(3), np.zeros(3)])
        A_ams_jfk_b = np.concatenate(
            [np.zeros(2), np.zeros(3), np.ones(3), np.zeros(3)])
        A_ams_jfk_e = np.concatenate(
            [np.zeros(2), np.zeros(3), np.zeros(3), np.ones(3)])

        A = np.vstack([A_bcn_ams_b, A_bcn_ams_e, A_ams_jfk_b, A_ams_jfk_e])
        A_identity = np.identity(11)
        A = np.vstack([A, A_identity]) * -1

        print()
        print("A matrix before transposing looks like:")
        print(A)
        print()
        print("b column vector printed as row vector looks like:")
        print()
        print(b)
        print()
        print("c vector printed as row vector looks like:")
        print(c)
        print()

        c_dual_model = b
        b_dual_model = [x * -1 for x in c]
        A_dual_model = np.transpose(A)

        bnd = [(0, None)]
        bnd = bnd * len(c_dual_model)
        res = linprog(c_dual_model, A_ub=A_dual_model, b_ub=b_dual_model,
                      options={"disp": False}, bounds=bnd)
        print("The output vector of the linear optimization is:")
        print(res.x)
        print()

        bid_prices = res.x[0:4]

        print("The four bidprices in order bcn_ams_20, bcn_ams_40, ams_jfk_20, ams_jfk_40: ")
        print(bid_prices)
        print()


        return bid_prices

    def save_bidprices(self, df_bid_prices):
        outputPath = "output/" + \
            str(int(self.doi[-2:]) - self.dbd + 1) + " nov - bidprice - group 02.xlsx"
        df_bid_prices.to_excel(outputPath, index=False)
        print("Saving the bidprices to: ")
        print(outputPath)

    def save_total_end_result(self):
        print("Updating the total results")


def main():
    print()
    date_of_interest = '2016-11-28'
    days_before_departure = 10
    inputFileNumber = str(int(date_of_interest[-2:]) - days_before_departure)
    inputFile = inputFileNumber + " nov 2016 - input.xlsx"
    calculator = BidPriceCalculator(
        inputFile, True, date_of_interest, days_before_departure)


if __name__ == "__main__":
    main()

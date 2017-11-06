import pandas as pd
import numpy as np
import datetime as dt
from tabulate import tabulate


class CapacityCalculator():

    def __init__(self):
        print("CapacityCalculator instance created")


    # This function returns a list of two integers (econ_seats, business_seats) where e is the number of booked seats for economy class for the
    # specified flight and b is the number of booked seats for the business class for the specified flight
    # This function requires pandas, datetime as dt and numpy as np
    # Inputs:
    # flightdf: Pandas dataframe; this can be either BCN-AMS or AMS-JFK
    # dpd: String; The departure date in format 'yyyy-mm-dd'
    # pred_d: String; The date you want to make predictions on (aka tomorrow) in format 'yyyy-mm-dd'
    def get_booked_seats(self, flightdf, dpd, pred_d):
        dep_date = dt.datetime.strptime(dpd, '%Y-%m-%d')
        pred_date = dt.datetime.strptime(pred_d, '%Y-%m-%d')
        flightdf["BookingCreationDateTime"] = pd.to_datetime(flightdf["BookingCreationDateTime"])
        flightdf["DepartureDate"] = pd.to_datetime(flightdf["DepartureDate"])
        flightdf["CancellationDateTime"] = pd.to_datetime(flightdf["CancellationDateTime"])
        df_econ = flightdf[(flightdf['Class'] == 'Economy') & (flightdf['DepartureDate'] == dep_date)]
        df_econ_cancelled = df_econ[df_econ['CancellationDateTime'] != dt.datetime(2020, 11, 12)]

        econ_seats = df_econ.NumberOfPax.sum()
        econ_seats_cancelled = df_econ_cancelled.NumberOfPax.sum()
        econ_seats = econ_seats - econ_seats_cancelled

        df_bus = flightdf[(flightdf['Class'] == 'Business') & (flightdf['DepartureDate'] == dep_date) & (flightdf['BookingCreationDateTime'] < pred_date)]
        df_bus_cancelled = df_bus[df_bus['CancellationDateTime'] != dt.datetime(2020, 11, 12)]

        bus_seats = df_bus.NumberOfPax.sum()
        bus_seats_cancelled = df_bus_cancelled.NumberOfPax.sum()
        bus_seats = bus_seats - bus_seats_cancelled

        return [econ_seats, bus_seats]

    # This function returns the remaining capacities of certain flight on a certain day as a list of integers [rem_cap_econ, rem_cap_bus].
    # Inputs:
    # flightdf: Pandas dataframe; this can be either BCN-AMS or AMS-JFK
    # flight: String; An indication of the flight, if flightdf==BCN-AMS this should be 'BCN-AMS' else if flightdf == AMS-JFK this should be 'AMS-JFK'
    # dpd: String; The departure date in format 'yyyy-mm-dd'
    # pred_d: String; The date you want to make predictions on (aka tomorrow) in format 'yyyy-mm-dd'
    def get_remaining_capacities(self, flightdf, flight, dpd, pred_d):
        booked_seats = self.get_booked_seats(flightdf, dpd, pred_d)
        if flight == 'BCN-AMS':
            rem_cap_econ = 180 - booked_seats[0]
            rem_cap_bus = 8 - booked_seats[1]
        elif flight == 'AMS-JFK':
            rem_cap_econ = 233 - booked_seats[0]
            rem_cap_bus = 35 - booked_seats[1]
        else:
            print("wrong input for variable flight!")
            return 0
        rem_cap_econ = max(0, rem_cap_econ)
        rem_cap_bus = max(0, rem_cap_bus)
        return [rem_cap_bus, rem_cap_econ]

    def update_cap_Bcn_Ams(self, df1, df2):
        to_append = df2[df2['SetOfFlightLegs'].str.contains("BCN-AMS")]
        if not to_append.empty:
            #to_append['CancellationDateTime'] = to_append['CancellationDateTime'].to_string()
            to_append['CancellationDateTime'].fillna('12-NOV-20', inplace=True)
            to_append['CancellationDateTime'] = to_append['CancellationDateTime'].str[0:9]
            to_append['CancellationDateTime'] =  pd.to_datetime(to_append['CancellationDateTime'], format='%d-%b-%y')
            to_append['DBD'] = 0
            to_append['DepartureDate'] = dt.datetime(2016,11,28)
            to_append = to_append.replace({'OKT': 'OCT'}, regex=True)
            to_append['BookingCreationDateTime'] = pd.to_datetime(to_append['BookingCreationDateTime'], format=('%d-%b-%y %I.%M.%S.%f %p'))
            to_append['DBD'] = to_append['DepartureDate'].dt.day.astype(int) - to_append['BookingCreationDateTime'].dt.day.astype(int)
            bflights = to_append['SetOfFlightLegs'].str.split(']', expand=True)
            temp1 = [5]*len(to_append)
            for i in (range(len(bflights))):
                for j in range(len(bflights.iloc[i])):
                    if not isinstance(bflights.iloc[i][j], str):
                        break
                    elif "BCN-AMS" in bflights.iloc[i][j]:
                        temp1[i]=j
            bflights['BCN-AMS LegNr']=temp1

            Class = []
            for i in (to_append.index.values):
                j = bflights['BCN-AMS LegNr'][i]
                if to_append['CabinLevelCombi'][i][3*j+1:3*j+3]=='20':
                    Class.append('Business')
                elif to_append['CabinLevelCombi'][i][3*j+1:3*j+3]!='40':
                    print("error")
                else:
                    Class.append('Economy')
            to_append["Class"] = Class
            df1 = pd.concat([df1, to_append])

        return df1


    def update_cap_Ams_Jfk(self, df1, df2):
        to_append = df2[df2['SetOfFlightLegs'].str.contains("AMS-JFK")]
        if not to_append.empty:
            to_append['CancellationDateTime'].fillna('12-NOV-20', inplace=True)
            to_append['CancellationDateTime'] = to_append['CancellationDateTime'].str[0:9]
            to_append['CancellationDateTime'] =  pd.to_datetime(to_append['CancellationDateTime'], format='%d-%b-%y')
            to_append['DBD'] = 0
            to_append['DepartureDate'] = dt.datetime(2016,11,28)
            to_append = to_append.replace({'OKT': 'OCT'}, regex=True)
            to_append['BookingCreationDateTime'] = pd.to_datetime(to_append['BookingCreationDateTime'], format=('%d-%b-%y %I.%M.%S.%f %p'))
            to_append['DBD'] = to_append['DepartureDate'].dt.day.astype(int) - to_append['BookingCreationDateTime'].dt.day.astype(int)
            bflights = to_append['SetOfFlightLegs'].str.split(']', expand=True)
            temp0 = [5]*len(to_append)
            for i in (range(len(bflights))):
                for j in range(len(bflights.iloc[i])):
                    if not isinstance(bflights.iloc[i][j], str):
                        break
                    elif "AMS-JFK" in bflights.iloc[i][j]:
                        temp0[i]=j
            bflights['AMS-JFK LegNr']=temp0

            Class = []
            for i in (to_append.index.values):
                j = bflights['AMS-JFK LegNr'][i]
                if to_append['CabinLevelCombi'][i][3*j+1:3*j+3]=='20':
                    Class.append('Business')
                elif to_append['CabinLevelCombi'][i][3*j+1:3*j+3]!='40':
                    print("error")
                else:
                    Class.append('Economy')
            to_append["Class"] = Class
            df1 = pd.concat([df1, to_append])
        return df1


def main():
    calculator = CapacityCalculator()

if __name__ == '__main__':
    main()

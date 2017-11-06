from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import datetime as dt
from math import radians, cos, sin, asin, sqrt


class DemandForecast():


    def __init__(self):
        print("Demand forecast instance created")
    # function to return relevant flighttable from aggregated bookingsdataset
    # flight: string; is in format 'AMS-JFK'
    # maintable: dataframe; is bookings result of above function
    # dep_date: string; is the departure date you want to have all data of, in format dd-mm-yyyy.
    def return_flighttable(self, flight, comfortclass, main_table, dep_date='28-11-2016'):
        boekingen = main_table
        if flight == 'AMS-JFK':
            res =  boekingen[(boekingen['BCNAMS_fare']==0) & (boekingen['classJFK']==comfortclass)&
                                (boekingen['DepartureDate']==dep_date)]
            res =  res[['PricingClass','dbd','CancellationDateTime','NumberOfPax','AMSJFK_fare']]
        elif flight == 'BCN-AMS':
            res =  boekingen[(boekingen['AMSJFK_fare']==0) & (boekingen['classBCN']==comfortclass)&
                                (boekingen['DepartureDate']==dep_date)]
            res = res[['PricingClass','dbd','CancellationDateTime','NumberOfPax','BCNAMS_fare']]
        elif flight == 'BCN-JFK':
            res =  boekingen[(boekingen['BCNAMS_fare']!=0) & (boekingen['AMSJFK_fare']!=0) &
                             (boekingen['classJFK']==comfortclass) & (boekingen['DepartureDate']==dep_date)]
            res['BCNJFK_fare'] = res['BCNAMS_fare'] + res['AMSJFK_fare']
            res = res[['PricingClass','dbd','CancellationDateTime','NumberOfPax','BCNJFK_fare']]
        else:
            print('invalid flight variable')
            return 0
        res = res
        return res

    # function to append average fares per pricing class in a bookingstable
    # flighttable is a table result of return_flighttable()
    def update_avg(self, flight_table):
        table = flight_table.copy()
        table.columns = ['pc','dbd','cdt','nop','fare']
        table['total_paid'] = table['nop'] * table['fare']
        grouped = table.groupby('pc').mean()
        dict_met_avg_fares = grouped.total_paid.to_dict()
        table['fare'] = table.pc.map(dict_met_avg_fares)
        return table[['pc','dbd','cdt','nop','fare']]

    def avg_per_label(self, flight_table,label):
        table = flight_table.copy()
        table['total_paid'] = table['nop'] * table['fare']
        grouped = table.groupby('label').mean()
        dict_met_avg_fares = grouped.total_paid.to_dict()
        table['fare'] = table.label.map(dict_met_avg_fares)
        fare = table[table['label']==label].fare
        return fare.iloc[0]

    #appends k-means labels
    def kmeans_label(self, flight_table, n):
        try:
            model = KMeans(n_clusters=n)
            model.fit(flight_table[['fare']])
            flight_table['label'] = model.labels_
            return flight_table.sort_values('dbd')
        except: return kmeans_label(flight_table,n=1)

        # dep_date: datetime format; blabla
    def cancellations_add(self, data, dep_date=dt.date(2016, 11, 28)):
        boekingen = data.copy()
        cancellations = boekingen[boekingen['cdt'] != boekingen['cdt'].min()]
        cancellations['dbd'] = (dep_date - cancellations['cdt']).dt.days
        cancellations['nop'] = -cancellations['nop']
        boekingen = pd.concat([boekingen, cancellations])
        return boekingen

    # function that returns a list with bookings for each dbd
    def pax_per_dbd(self, flight_table, label):
        demand = np.zeros(335)
        demand_table = flight_table[flight_table['label'] == label].groupby('dbd').sum()
        for i in demand_table.index:
            if i >= 0:
                demand[i] = demand_table['nop'][i]
        return demand


    def calculate_average(self, x_0, x_1, alpha):
        return alpha*x_0 + (1-alpha)*x_1

    def ewma(self, demand, alpha, current_dbd):
        ewma = np.zeros(335)
        for i in range(len(demand)-current_dbd):
            a = self.calculate_average(demand[333-i], ewma[334-i],alpha)
            ewma[333-i] = a
    #     for i in range(current_dbd):
    #         a = calculate_average(ewma(current_dbd-))
        return ewma


    def demand_in_final_days(self, tot_exp_demand,current_dbd):
        y_exp = tot_exp_demand[current_dbd:current_dbd+20]
        x = [[i] for i in range(len(y_exp))]
        to_predict = [[i] for i in range(0,current_dbd)]
        model = LinearRegression()
        model.fit(x,y_exp)
        return model.predict(to_predict)

    def super_func(self, bookings, flight,comfortclass, dbd,dep_date='28-11-2016',n_clusters=3,alpha=0.1):
        cancel_dep = dt.datetime.strptime(dep_date,'%d-%m-%Y')
        flighttable = self.return_flighttable(flight,comfortclass, bookings,dep_date)
        flighttable = self.update_avg(flighttable)
        flighttable_labeled = self.kmeans_label(flighttable,n_clusters)
        flighttable_labeled_canc = self.cancellations_add(flighttable_labeled,cancel_dep)
        ans = []
        for label in range(n_clusters):
            real_demand_dbd = self.pax_per_dbd(flighttable_labeled_canc,label)
            exp_demand_dbd = self.ewma(real_demand_dbd,alpha,dbd)
            demand_final_days = self.demand_in_final_days(exp_demand_dbd,dbd)
            total_demand = demand_final_days.sum()
            fare = self.avg_per_label(flighttable,label)
            ans.append([fare, total_demand])
        return ans

    def update_demand(self, old, new_day):
        # voer nieuwe dag in die je wilt concatten met oude boekingen
        new_day = self.update_demand_helper(new_day)
        return pd.concat([old, new_day])


    def datetime_booking(self, data):
        bookings = data.copy()
        bookings = bookings[bookings['Fare'] != 0.0]
        #bookings = bookings[bookings['CancellationDateTime'].isnull()]
        bookings.CancellationDateTime = pd.to_datetime(bookings.CancellationDateTime, format=('%d-%b-%y %I.%M.%S.%f %p'))
        bookings.DepartureDate = pd.to_datetime(bookings.DepartureDate, format='%Y-%m-%d')
        bookings.BookingCreationDateTime = bookings.BookingCreationDateTime.str.replace('MEI','MAY')
        bookings.BookingCreationDateTime = bookings.BookingCreationDateTime.str.replace('OKT','OCT')
        bookings.BookingCreationDateTime = bookings.BookingCreationDateTime.str.replace('MAA','MAR')
        bookings.BookingCreationDateTime = pd.to_datetime(bookings.BookingCreationDateTime, format=('%d-%b-%y %I.%M.%S.%f %p'))
        bookings['dbd'] = bookings['DepartureDate'].map(pd.Timestamp.date) - bookings['BookingCreationDateTime'].map(pd.Timestamp.date)
        bookings['dbd'] = bookings['dbd'].dt.days
        #bookings['dbd'] = (bookings['DepartureDate'] - bookings['BookingCreationDateTime']).dt.days.astype(int)
        bookings['DepartureDate'] = bookings['DepartureDate'].dt.strftime('%d-%m-%Y')
        return bookings

    def calcDist(self, dist_matrix, airports):
        departure = airports[0:3]
        arrival = airports[4:7]
        try:
            distance = dist_matrix[departure][arrival]
            return distance
        except:
            return 0

    def split_legs(self, data):
        bookings = data.copy()
        bookings['first'] = bookings['SetOfFlightLegs'].str[17:24]
        bookings['second'] = bookings['SetOfFlightLegs'].str[42:49]
        bookings['third'] = bookings['SetOfFlightLegs'].str[67:74]
        bookings['fourth'] = bookings['SetOfFlightLegs'].str[92:99]
        return bookings

    def split_dist(self, dist_matrix, data):
        testdf = data.copy()
        testdf['first_dist'] = testdf['first'].apply(lambda x: self.calcDist(dist_matrix, x))
        testdf['second_dist'] = testdf['second'].apply(lambda x: self.calcDist(dist_matrix, x))
        testdf['third_dist'] = testdf['third'].apply(lambda x: self.calcDist(dist_matrix, x))
        testdf['fourth_dist'] = testdf['fourth'].apply(lambda x: self.calcDist(dist_matrix, x))
        testdf['total_dist'] = testdf['first_dist'] + testdf['second_dist'] + testdf['third_dist']
        return testdf

    def split_fare(self, data):
        testdf = data.copy()
        testdf['first_fare'] = testdf['first_dist']/testdf['total_dist']*testdf['Fare']
        testdf['second_fare'] = testdf['second_dist']/testdf['second_dist']*testdf['Fare']
        testdf['third_fare'] = testdf['third_dist']/testdf['third_dist']*testdf['Fare']
        testdf['fourth_fare'] = testdf['fourth_dist']/testdf['fourth_dist']*testdf['Fare']
        return testdf

    def no_of_leg(self, data):
        newbookings = data.copy()
        newbookings.fillna(0, inplace=True)
        newbookings['AMSJFK_fare'] = ((newbookings['first'] == 'AMS-JFK') * newbookings['first_fare']) + \
        ((newbookings['second'] == 'AMS-JFK') * newbookings['second_fare']) + \
        ((newbookings['third'] == 'AMS-JFK') * newbookings['third_fare']) + \
        ((newbookings['fourth'] == 'AMS-JFK') * newbookings['fourth_fare'])
        newbookings['BCNAMS_fare'] = ((newbookings['first'] == 'BCN-AMS') * newbookings['first_fare']) + \
        ((newbookings['second'] == 'BCN-AMS') * newbookings['second_fare']) + \
        ((newbookings['third'] == 'BCN-AMS') * newbookings['third_fare']) + \
        ((newbookings['fourth'] == 'BCN-AMS') * newbookings['fourth_fare'])
        return newbookings

    def replace_nan(self, data):
        bookings = data.copy()
        bookings[['first_fare','second_fare','third_fare','fourth_fare']] = bookings[['first_fare','second_fare','third_fare','fourth_fare']].replace(0,'-')
    #     bookings = bookings[(bookings['dbd']<=10) & (bookings['dbd']>=0)]
        return bookings

    def leg_pos(self, data):
        newbookings = data.copy()
        newbookings['leg_BCN'] = 1 * (newbookings['BCNAMS_fare'] == newbookings['first_fare'])
        newbookings['leg_BCN'] = np.where((newbookings['leg_BCN'] == 0),2*(newbookings['BCNAMS_fare'] == newbookings['second_fare']),
                                        newbookings['leg_BCN'])
        newbookings['leg_BCN'] = np.where((newbookings['leg_BCN'] == 0),3*(newbookings['BCNAMS_fare'] == newbookings['third_fare']),
                                        newbookings['leg_BCN'])
        newbookings['leg_BCN'] = np.where((newbookings['leg_BCN'] == 0),4*(newbookings['BCNAMS_fare'] == newbookings['fourth_fare']),
                                        newbookings['leg_BCN'])
        newbookings['leg_JFK'] = 1 * (newbookings['AMSJFK_fare'] == newbookings['first_fare'])
        newbookings['leg_JFK'] = np.where((newbookings['leg_JFK'] == 0),2*(newbookings['AMSJFK_fare'] == newbookings['second_fare']),
                                        newbookings['leg_JFK'])
        newbookings['leg_JFK'] = np.where((newbookings['leg_JFK'] == 0),3*(newbookings['AMSJFK_fare'] == newbookings['third_fare']),
                                        newbookings['leg_JFK'])
        newbookings['leg_JFK'] = np.where((newbookings['leg_JFK'] == 0),4*(newbookings['AMSJFK_fare'] == newbookings['fourth_fare']),
                                        newbookings['leg_JFK'])
        return newbookings

    def class_type(self, data):
        newbookings = data.copy()
        newbookings['classBCN'] = np.where((newbookings['leg_BCN']==1),newbookings.CabinLevelCombi.str[1:3],
                                     newbookings['leg_BCN'])
        newbookings['classBCN'] = np.where((newbookings['leg_BCN']==2),newbookings.CabinLevelCombi.str[4:6],
                                     newbookings['classBCN'])
        newbookings['classBCN'] = np.where((newbookings['leg_BCN']==3),newbookings.CabinLevelCombi.str[7:9],
                                     newbookings['classBCN'])
        newbookings['classBCN'] = np.where((newbookings['leg_BCN']==4),newbookings.CabinLevelCombi.str[10:12],
                                     newbookings['classBCN'])
        newbookings['classJFK'] = np.where((newbookings['leg_JFK']==1),newbookings.CabinLevelCombi.str[1:3],
                                     newbookings['leg_JFK'])
        newbookings['classJFK'] = np.where((newbookings['leg_JFK']==2),newbookings.CabinLevelCombi.str[4:6],
                                     newbookings['classJFK'])
        newbookings['classJFK'] = np.where((newbookings['leg_JFK']==3),newbookings.CabinLevelCombi.str[7:9],
                                     newbookings['classJFK'])
        newbookings['classJFK'] = np.where((newbookings['leg_JFK']==4),newbookings.CabinLevelCombi.str[10:12],
                                     newbookings['classJFK'])

        classdict = {'20':'Business', '40':'Economy'}
        newbookings.replace(to_replace={'classBCN':classdict},inplace=True)
        newbookings.replace(to_replace={'classJFK':classdict},inplace=True)

        return newbookings

    def update_demand_helper(self, data):
        bookings = data.copy()
        bookings = self.datetime_booking(bookings)
        KlM_dist_matrix = self.KLM_airport_dist_calculator(bookings)
        bookings = self.split_legs(bookings)
        bookings = self.split_dist(KlM_dist_matrix, bookings)
        bookings = self.split_fare(bookings)
        bookings = self.no_of_leg(bookings)
        bookings = self.replace_nan(bookings)
        bookings = self.leg_pos(bookings)
        bookings = self.class_type(bookings)
        return bookings

    def KLM_airport_dist_calculator(self, bookings):
        url="https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
        airports = pd.read_csv(url, index_col=0,
                               names=['name','city','country','code','code4','lat','long',
                                        'alt','tz','dst','dst_db','type','src'])

        airports = airports[['code','lat','long']]
        airports['cox'] = [(lat, long) for lat, long in zip(airports.lat, airports.long)]

        # all unique airports in dataset
        KLM_dist = set(list(bookings.DominantLegArrivalAirportCode.unique())
                       + list(bookings.DominantLegDepartureAirportCode.unique())
                       + list(bookings.OriginAirportCode.unique())
                       + list(bookings.DestinationAirportCode.unique()))

        KLM_airports = airports[airports.code.isin(list(KLM_dist))].reset_index(drop=True)

        ap_dist = []
        i = len(KLM_airports)
        for i in range(len(KLM_airports)):
            for j in range(len(KLM_airports)):
                ap_dist.append(self.haversine2(KLM_airports.cox[i],KLM_airports.cox[j]))
        ap_dist = np.reshape(ap_dist,(sqrt(len(ap_dist)),sqrt(len(ap_dist))))
        KLM_df = pd.DataFrame(ap_dist, columns=KLM_airports.code, index=KLM_airports.code)
        return KLM_df

    def haversine2(self, loc1,loc2):
        lon1 = loc1[1]
        lat1 = loc1[0]
        lon2 = loc2[1]
        lat2 = loc2[0]
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6367 * c
        return km

def main():
    forecaster = DemandForecast()

if __name__ == '__main__':
    main()

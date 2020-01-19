import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.stats import pearsonr
import scipy.optimize
from textwrap import wrap
from mpl_toolkits.mplot3d import Axes3D
import functools


def mainfn(filename, display=True, Diam_Bar=0.006, ):
    local_g = 9.81256
# https://www.sensorsone.com/local-gravity-calculator/
# https://www.freemaptools.com/elevation-finder.htm

    def inwords(fileprefix):
        list = fileprefix.split('][')
        if len(list) >= 3:
            x = 'Bifilar pendulum with s = {} '
            x += 'and L = {} oscillating in motion \'{}\''
            return x.format(list[0], list[1], list[2].upper())
        else:
            return '{}'.format(fileprefix)

    print(inwords(filename))

# Adds file info into returndict
    def intometres(string, Nofilter=False):
        if '_' in string or Nofilter:
            string = float(string.replace('_', '.')) / 100
        return string

    keynamelist = ['Length s', 'Length L', 'Oscillation type',
                   'T_Adjustment']
    returndict = {}
    for xi, x in enumerate(filename.split('][')):
        returndict[keynamelist[xi]] = intometres(x)

# Opens file and begins analysis
    file = open(filename + '.CSV', 'r+')

    filepath = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
# print(filepath) C:\Users\KK\Desktop\Non Tex Physics project\THW THOUSAND
# Makes a folder each csv with csv's name
    mainlist = []
    for string in file.readlines():
        mainlist.append(string.strip().split(',')[:-1])
# Converts csv into lists where the last column (comment) is removed

    def int_float_str(a):
        try:
            return int(a)
        except:
            try:
                return float(a)
            except:
                return str(a)

# removes column headings in file
    forbiddenrecords = [['s', 'State', 'State'],
                        [],
                        ['s', 's', 'State', 'State']]
    mainlist = [x for x in mainlist if x not in forbiddenrecords]

    A_state = []
    B_state = []
    forbiddenrecords = []
#  splits into unfiltered columns and removes duplicate times
    for counter, value in enumerate(mainlist):
        if value[1] == mainlist[counter - 1][1]:
            forbiddenrecords.append(value[1])
        else:
            A_state.append([int_float_str(i) for i in [value[1], value[2]]])
            B_state.append([int_float_str(i) for i in [value[1], value[3]]])

    mainlist = [x for x in mainlist if x[1] not in forbiddenrecords]

# Removes data points that mean nothing

    def Clean_One_State(ListsToConvert):
        # Removes duplicate times and states
        prev_bool = 2
        returnlist = []
        for counter, value in enumerate(ListsToConvert[1:], 0):
            if value[1] != prev_bool and value[0] != ListsToConvert[counter][0]:
                try:
                    if value[0] != ListsToConvert[counter + 2][0]:
                        returnlist.append(value)
                        prev_bool = value[1]
                except IndexError:
                    returnlist.append(value)
                    prev_bool = value[1]
        return returnlist

    A_state_clean = []  # [['0.00000', '1'], ['0.04757', '0'], ect
    B_state_clean = []  # rows have different values for the second record
    A_state_clean = Clean_One_State(A_state)
    B_state_clean = Clean_One_State(B_state)

# Puts analysis into csv file in case manual data processing is needed

    def NewCSV(NameOfCsvFile, ListsToConvert, folder=''):
        pass
        if not os.path.exists(filepath + folder):
            os.makedirs(filepath + folder)
        filepathtoopen = filepath + folder + '\\' + NameOfCsvFile + '.csv'
        with open(filepathtoopen, 'w', newline='') as temp:
            writer = csv.writer(temp)
            writer.writerows(ListsToConvert)

    NewCSV('A_state_clean', A_state_clean)
    NewCSV('B_state_clean', B_state_clean)

    def Delta_Time_One_State(CleanState):
        Lgt_cut_Time = []
        Lgt_Uncut_Time = []
        for counter, value in enumerate(CleanState[1:], 0):
            DeltaT = round(value[0] - CleanState[counter][0], 7)
            if value[1] == 0:
                Lgt_cut_Time.append([round(value[0] - DeltaT / 2, 7), DeltaT])
            else:
                Lgt_Uncut_Time.append([round(value[0] - DeltaT / 2, 7), DeltaT])

        return Lgt_cut_Time, Lgt_Uncut_Time

# Finds the time spent at one state and records the average time

    A_DeltaT_Cut, A_DeltaT_Uncut = Delta_Time_One_State(A_state_clean)
    B_DeltaT_Cut, B_DeltaT_Uncut = Delta_Time_One_State(B_state_clean)

    NewCSV('A_DeltaT_Cut', A_DeltaT_Cut, folder='\\Cut_Light_Gate')
    NewCSV('A_DeltaT_Uncut', A_DeltaT_Uncut, folder='\\Beyond_Light_Gate')
    NewCSV('B_DeltaT_Cut', B_DeltaT_Cut, folder='\\Cut_Light_Gate')
    NewCSV('B_DeltaT_Uncut', B_DeltaT_Uncut, folder='\\Beyond_Light_Gate')

# Data analysis via numpy, imported as np

    mainarray = np.array([np.array([float(item[1]), int(item[2]),
                int(item[3])]) for item in mainlist[1:]])

    def TrueTime(inputarray, n=2):
        # More or less a rolling average but with None stuck in front of it
        returnarray = np.cumsum(inputarray, dtype=float)
        returnarray[n:] = returnarray[n:] - returnarray[:-n]
        returnarray = np.concatenate((returnarray[n - 1:] / n, [None]),)
        return returnarray.reshape((len(inputarray), 1))

    Realtime = TrueTime(mainarray[:, :1])
    # Gives an average of the time and the next time
    # to give the True equlibrium time
    rptfirst = np.concatenate(([mainarray[0, [1, 2]]], mainarray[:, [1, 2]]), axis=0)
    diff_array = np.absolute(np.diff(rptfirst, axis=0))
    cumulativediff = np.cumsum(diff_array, axis=0)
    # Gives a cumulative sum of the differences of terms, first term is zero
    MetaArray = np.concatenate((mainarray, cumulativediff, Realtime), axis=1)[:-1]
    # [:-1] trims last None value for Realtime
    Equlibriamask = (MetaArray[:, 1] == 1) & (MetaArray[:, 2] == 1)
    Equlibria = MetaArray[Equlibriamask][:, [3, 4, 5]]
    # Keeps only state A = state B = 1
    # Keeps only cumulative difference and Realtime columns
    DeltaEqulibria = np.diff(Equlibria, axis=0)
    notanomalous = (DeltaEqulibria[:, 0] == DeltaEqulibria[:, 1])
    DeltaEqulibria = DeltaEqulibria[notanomalous]

    def reject_outliers(data, m=2):
        y = data[:, 2] / data[:, 1]
        mask = abs(y - np.mean(y)) < m * np.std(y)
        oppositemask = abs(y - np.mean(y)) > m * np.std(y)
        if np.count_nonzero(oppositemask) > 70:
            print('Time Periods Removed: ', data[oppositemask])
        return data[mask]

#    np.set_printoptions(threshold=sys.maxsize) a
# comment out previous to allow full size arrays to be printed
    DeltaEqulibria = reject_outliers(DeltaEqulibria)
#    SumDeltaEqulibria = np.sum(DeltaEqulibria, axis=0)
    timeperiodarray = DeltaEqulibria[:, 2] / DeltaEqulibria[:, 1]
    timeperiodarray = np.repeat(timeperiodarray, DeltaEqulibria[:, 1].astype(int))
# Weights the mean correctly
    timeperiodarray = timeperiodarray * 4
    TParray = timeperiodarray

    np.per = np.percentile
    _ = 'Time Period '
# OPTIONAL
# Set true for statistics about each individual time period
    printstatistics = True
    if printstatistics:
        print(_ + 'Min:      ', np.min(TParray))
        print(_ + 'Max:      ', np.max(TParray))
        print(_ + 'Mean:     ', np.mean(TParray))
        print(_ + 'Median    ', np.median(TParray))
        print(_ + 'Std:      ', np.std(TParray))
        print(_ + 'IQR:      ', np.per(TParray, 75) - np.per(TParray, 25))
        print('Time Periods sampled: ', len(TParray) / 2)

#    returndict['TimePeriod Min'] = np.min(TParray)
#    returndict['TimePeriod Max'] = np.max(TParray)
    returndict['TimePeriod Mean'] = np.mean(TParray)
#    returndict['TimePeriod Median'] = np.median(TParray)
    returndict['TimePeriod St Dev'] = np.std(TParray)
    returndict['TimePeriod IQR:'] = np.per(TParray, 75) - np.per(TParray, 25)
    returndict['TimePeriods sampled:'] = len(TParray) / 2

    def reject_outliers_by_split(data, m=3):
        def subsectionoutliers(data):
            y = data[:, 1]
            xandy = data[abs(y - np.mean(y)) < m * np.std(y)]
            return xandy
# m=2 is too low; occasionally truncates extreme valid values
        splitarray = np.array_split(data, 5)
        xandy = subsectionoutliers(splitarray[0])
        for item in splitarray[1:]:
            xandy = np.concatenate((xandy, subsectionoutliers(item)))
        return xandy[:, 0], xandy[:, 1]

# Plotting with Matplot.lib

# Set true or false to toggle different graphs
# Only activates graphs if required
    DisplayGraph2 = True and display
    DisplayGraph3 = True and display
    DisplayGraph4 = True and display
    moduloshift = 0
    try:
        moduloadjust = returndict['T_Adjustment']
    except:
        moduloadjust = 0.00

    graphfourModulo = (returndict['TimePeriod Mean'] + moduloadjust)
    returndict['True TimePeriod'] = graphfourModulo
    _ = (graphfourModulo / (2 * math.pi)) ** 2 * local_g
#    print('Lin Predicted L: ', _)

# Defining figures and their axes
    plt.style.use('seaborn-whitegrid')
    fig, axs = plt.subplots(2, sharex=True)
    title = inwords(filename)
    fig.suptitle(title, fontsize=13)

    axs[0].set_title('Time spent in light gate compared to the event time')
#    axs[0].set_xlabel('Event Time / s')
    axs[0].set_ylabel('Time spent in light gate / s')
    axs[1].set_xlabel('Event Time / s')
    axs[1].set_title('Time spent beyond the light gate compared to the event time')
    axs[1].set_ylabel('Time spent beyond light gate / s')


    if DisplayGraph2:
        figtwo, axstwo = plt.subplots(1)
        titletwo = 'Log speed at light gate of: ' + inwords(filename)
        titletwo = "\n".join(wrap(titletwo, 60))
        figtwo.suptitle(titletwo, fontsize=13)
        axstwo.set_ylabel('Ln(speed of bar at light gate)')
        axstwo.set_xlabel('Event time / s')
    if DisplayGraph3:
        figthr, axsthr = plt.subplots(1)
        titlethree = 'Average Acceleration compared to event time in '
        titlethree += inwords(filename)
        titlethree = "\n".join(wrap(titlethree, 63))
        figthr.suptitle(titlethree, fontsize=13)
        axsthr.set_xlabel('Event Time / s')
        axsthr.set_ylabel(r'Average Acceleration / $ms^{-2}$')
    if DisplayGraph4:
        figfour, axsfour = plt.subplots(1)
        titlefour = 'Variation in successive time periods in '
        titlefour += inwords(filename)
        titlefour = "\n".join(wrap(titlefour, 60))
        figfour.suptitle(titlefour, fontsize=13)
        axsfour.set_xlabel('Event Time / s')
        axsfour.set_ylabel('Event Time Modulo Time Period / s')

# Analysis of the times at which the light gate cuts the beam

    x, y = reject_outliers_by_split(np.array(A_DeltaT_Cut)[1:])
    axs[0].plot(x, y, '.', color='red', label='Light Gate A')

    x2a, y2a = x, np.log(Diam_Bar / y)
    if DisplayGraph2:
        axstwo.plot(x2a, y2a, '.', color='red', label='Light Gate A')
    if DisplayGraph3:
        y3a = np.copy(y)
        y3a[::2] = - y3a[::2]
        x3a, y3a = x[:-1], np.diff(Diam_Bar / y3a) / np.diff(x)
        axsthr.plot(x3a, y3a, '.', color='red', label='Light Gate A')
    if DisplayGraph4:
        x4a = np.mod(x + moduloshift, np.array(graphfourModulo))
        axsfour.plot(x, x4a, '.', color='red',
                     label='Times at which the bar cuts light gate A')

    x, y = reject_outliers_by_split(np.array(B_DeltaT_Cut)[1:])
    axs[0].plot(x, y, '.', color='blue', label='Light Gate B')

    x2b, y2b = x, np.log(Diam_Bar / y)
    if DisplayGraph2:
        axstwo.plot(x2b, y2b, '.', color='blue', label='Light Gate B')
    if DisplayGraph3:
        y3b = np.copy(y)
        y3b[::2] = - y3b[::2]
        x3b, y3b = x[:-1], np.diff(Diam_Bar / y3b) / np.diff(x)
        axsthr.plot(x3b, y3b, '.', color='blue', label='Light Gate B')
    if DisplayGraph4:
        x4b = np.mod(x + moduloshift, np.array(graphfourModulo))
        axsfour.plot(x, x4b, '.', color='blue',
                     label='Times at which the bar cuts light gate B')

# Analysis of the time the bar is outside of light gate beam

    x, y = reject_outliers_by_split(np.array(A_DeltaT_Uncut)[1:])
    axs[1].plot(x, y, '.', color='red', label='Light Gate A')
    A_Uncutx, A_Uncuty = np.copy(x), np.copy(y)

    if DisplayGraph4:
        x4a = np.mod(x + moduloshift, np.array(graphfourModulo))
        axsfour.plot(x, x4a, '.', color='magenta',
                     label='Times at which the bar is outside Light Gate A')

    x, y = reject_outliers_by_split(np.array(B_DeltaT_Uncut)[1:])
    axs[1].plot(x, y, '.', color='blue', label='Light Gate B')

    if DisplayGraph4:
        x4b = np.mod(x + moduloshift, np.array(graphfourModulo))
        axsfour.plot(x, x4b, '.', color='cyan',
                     label='Times at which the bar is outside Light Gate B')

#

    correlationx = np.hstack((A_Uncutx, x))
    correlationy = np.hstack((A_Uncuty, y))
    ValidOsciallations, _ = pearsonr(correlationx, correlationy)
    returndict['Validity'] = abs(ValidOsciallations)

    axs[0].legend(loc='upper left', frameon=True)
    axs[1].legend(loc='lower left', frameon=True)

    x = np.concatenate((x2b, x2a))
    y = np.concatenate((y2b, y2a))
    polynomial = np.poly1d(np.polyfit(x, y, 1))
#    returndict['Coefficient x'] = polynomial.coefficients[0]
#    returndict['Ln Intercept'] = polynomial.coefficients[1]
    Q = polynomial.coefficients[0]
    Q = (2 * math.pi) / (- math.expm1(Q * np.mean(TParray)))
#    print(Q)
    returndict['Q-factor'] = Q

    if DisplayGraph2:
        axstwo.plot(np.unique(x), polynomial(np.unique(x)),
                    'black', label=str(polynomial))
        axstwo.legend(frameon=True)
#        print(np.poly1d(np.polyfit(x, y, 1)))
    if DisplayGraph4:
        axsfour.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                       fancybox=True, shadow=True)
# Related to time spend outside light gate

    if printstatistics:
        print('Data Validity:        ', str(abs(ValidOsciallations)))
        print('Q-factor:             ', str(Q))
        print('\n')

    if display:
        plt.show()
    else:
        plt.close('all')

    return returndict


# OPTIONAL
# showgraphs allows the graphs of each individual data point to be displayed
showgraphs = False

if len(glob.glob('*.{}'.format('csv'))) < 6:
    for fileprefix in [i[:-4] for i in glob.glob('*.{}'.format('csv'))]:
        mainfn(fileprefix, display=showgraphs)
    exit()

DictsToConvert = []
for fileprefix in [i[:-4] for i in glob.glob('*.{}'.format('csv'))]:
    DictsToConvert.append(mainfn(fileprefix, display=showgraphs))

mainarray = pd.DataFrame(DictsToConvert).values

# OPTIONAL
# Set True to analyse Q across multiple data points
# Set false to analyse T across multiple data points
Qfactoranalysis = False

if Qfactoranalysis:
    threelogaxis = mainarray[:, [0, 1, 10]]
else:
    threelogaxis = np.log(mainarray[:, [0, 1, 8]].astype('float64'))


# Makes a 3d graph
# raw_data is the raw data used for plotting graphs.
# W_points are the individual data weightings
# Used with raw_data for calculating plane of best fit


def Voodoo_Magic_Three_D(W_points, raw_data):
    def error(params, points, weights):
        def plane(x, y, params):
            a = params[0]
            b = params[1]
            c = params[2]
            z = a * x + b * y + c
            return z
        result = 0
        weights = np.expand_dims(weights, axis=0)
        for (x, y, z, w) in np.concatenate((points, weights.T), axis=1):
            plane_z = plane(x, y, params)
            diff = abs(plane_z - z) * w
            result += diff**2
        return result
# Something something multivariate regression something something
    def LinearRegression(data):
        fun = functools.partial(error, points=data, weights=W_points)
        params0 = [0.5, -1, 0.3]
        res = scipy.optimize.minimize(fun, params0)
        a0 = res.x[0]
        b0 = res.x[1]
        c0 = res.x[2]
        print('\n')
        if Qfactoranalysis:
            print('Q = {} * L + {} * s + {}'.format(b0, a0, c0))
        else:
            print('ln(T) = {} * ln(L) + {} * ln(s) + {}'.format(b0, a0, c0))
        print('\n')
        return a0, b0, c0

    a, b, c = LinearRegression(raw_data)

# Coffecients of plane of best fit via scipy.optimize.minimize

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = zip(*raw_data)
    ax.scatter(xs, ys, zs)

    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]]

    point = np.array([0.0, 0.0, c])
    normal = np.array(cross([1, 0, a], [0, 1, b]))
    d = -point.dot(normal)
    _ = 0.1
    xx, yy = np.meshgrid(
        [np.max(raw_data[:, 0]) + _, np.min(raw_data[:, 0]) - _],
        [np.max(raw_data[:, 1]) + _, np.min(raw_data[:, 1]) - _])

    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    ax.plot_surface(xx, yy, z, alpha=0.2, color=[0, 1, 0])

    if Qfactoranalysis:
        ax.set_zlabel('Q-factor')
        ax.set_xlabel('$s$')
        ax.set_ylabel('$L$')
    else:
        ax.set_zlabel('ln($T$)')
        ax.set_xlabel('ln($s$)')
        ax.set_ylabel('ln($L$)')
    if True:
        plt.show()
    plt.close('all')


weightings = 100 * mainarray[:, 7] / (mainarray[:, 9] * mainarray[:, 10]) + 30
Voodoo_Magic_Three_D(weightings.astype('int64'), threelogaxis)

# -*- coding: utf-8 -*-
#Function library for Quantitative Risk and Portfolio Management book
#Copyright (c) 2020 Kenneth Winston

#Generate sample standard deviations over lookback months
def GenSampleSd(LogReturns,lookbacks):
    import numpy as np
    Sqrt12=12.0**0.5
    SampleSd=[[np.std(LogReturns[x:x+lb])*Sqrt12 \
            for x in range(len(LogReturns)-lb)] \
            for lb in lookbacks]
    return(SampleSd)
#Done with GetSampleSd

#Plot a graph of sample standard deviations
def PlotSampleSd(Title,Date,SampleSd,lookbacks,colors):
    import matplotlib.pyplot as plt
    
    for i, lb in enumerate(lookbacks):
        plt.plot(Date[lb:], SampleSd[i], colors[i],\
                label=str(lb)+' month')
    plt.xticks(rotation=45)
    plt.legend(loc='upper right', shadow=False, fontsize='medium')
    plt.grid()

    plt.title(Title)
    plt.ylabel('Sample SDs')
    plt.axis([min(Date),max(Date),0,70])
    plt.show()
    return
#Done with PlotSampleSd

#get Fama French 3 factor data from French's website
def getFamaFrench3(enddate=None):
    #enddate in integer yyyymm format if given    
    import pandas as pd
    
    ffurl='http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'

    #Read just the first line of the FF file into a dataframe
    df_monthly = pd.read_csv(ffurl, header=None, nrows=1)

    #Put the line into a string
    str=df_monthly.iloc[0,0]

    #8th word in the string is the last date in the file in YYYYMM format
    lastyear=int(str.split()[8][:4])
    lastmonth=int(str.split()[8][4:])
    #first date in the file is June 1926 - figure out
    #number of months based on that
    periods=(lastyear-1926)*12+(lastmonth-6)    
    
    #Use specified end date?
    if enddate:
        ed_year = int(enddate/100)
        ed_month = enddate%100
        ed_periods = (ed_year-1926)*12+(ed_month-6)
        if (ed_periods > 0) & (ed_periods < periods):    #Use specified end date
            periods = ed_periods
    
    #Now we know how many periods to read - skip the header and read those monthly periods
    names_monthly = ["yearmon", "mkt_minus_rf", "SMB", "HML", "RF"]
    df_monthly = pd.read_csv(ffurl, skiprows=4, nrows=periods, names=names_monthly)
    
    #Transfer from data frame to output arrays
    Date=df_monthly["yearmon"]
    market_minus_rf=df_monthly["mkt_minus_rf"]
    SMB=df_monthly["SMB"]
    HML=df_monthly["HML"]
    RF=df_monthly["RF"]
    
    return(Date,market_minus_rf,SMB,HML,RF)
#Done with getFamaFrench3

#Change returns in format 5.0=5% to log-returns log(1.05)
#Also add back a risk-free rate
def LogReturnConvert(Ret100,RF):
    import math
    return( [100.0*math.log(1+(r1+rf)/100.) for (r1,rf) in zip(Ret100,RF)] )
#Done with LogReturnConvert

def formula3p3(c,r,t):
    #Formula 3.3 for price of bond
    #with annual coupon c, t years to
    #maturity, discount rate r
    if r<=-100:  #Unreasonable discount rate
        return(100)
    y=1/(1+r/100)
    price=100*(y**t)
    if (y==1):   #no discount rate
        geometric=t
    else:
        geometric=(1-y**t)/(1-y)
    price+=geometric*c*y
    return(price)
#Done with Formula3p3

def formula3p8(c,r,t):
    #Formula 3.8 for Macauley duration of bond
    #with annual coupon c, t years to
    #maturity, discount rate r
    if r<=-100:  #Unreasonable discount rate
        return(0)
    y=1/(1+r/100)
    ytothet=y**t
    duration=100*t*ytothet
    if (y==1):   #no discount rate
        multiplier=t*(t+1)/2
    else:
        multiplier=(1-ytothet-t*(1-y)*ytothet)/(1-y)**2
    duration+=multiplier*c*y
    #formula3p3 is also in qrpm_funcs.py
    price=formula3p3(c,r,t)   #Rescale by price
    duration/=price
    return(duration)
#Done with Formula3p8

def formula3p9(c,r,t):
    #Formula 3.9 for convexity of bond
    #with annual coupon c, t years to
    #maturity, discount rate r
    if r<=-100:  #Unreasonable discount rate
        return(0)
    y=1/(1+r/100)
    ytothet=y**t
    convexity=100*t*(t+1)*ytothet*(y**2)
    if (y==1):   #no discount rate
        ytttterm=0
    else:
        ytttterm=-(t+1)*(t+2)+2*t*(t+2)*y-t*(t+1)*y**2
        ytttterm*=ytothet
        ytttterm+=2
        ytttterm*=c*(y/(1-y))**3
    convexity+=ytttterm
    #formula3p3 is also in qrpm_funcs.py
    price=formula3p3(c,r,t)   #Rescale by price
    convexity/=price
    return(convexity)
#Done with Formula3p9

def LastYearEnd(yearof=None):
    #return YYYY-MM-DD date string
    #that is the last business day
    #of yearof. If no yearof, uses
    #the year before the current date.
    #If the last calendar
    #day is on a weekend, assumes the last
    #Friday is the last business day
    import datetime as dt
    if not yearof:
        yearof=dt.date.today().year-1
    for day in [31,30,29,28]:
        l=dt.date(yearof,12,day)
        if l.weekday()<5:
            return(str(l.year)+'-12-'+str(day))
#Done with LastYearEnd

def TenorsFromNames(seriesnames):
    #Takes a list of FRED series names
    #of the form DGSx or DGSxMO and convers
    #to a list of real numbers giving tenors
    #in years. DGS3MO becomes .25, DGS30 becomes 30.0.
    tenors=[]
    for i in range(len(seriesnames)):
        if seriesnames[i][-2:]=='MO':
            tenors.append(float(seriesnames[i][3:-2])/12)
        else:
            tenors.append(float(seriesnames[i][3:]))
    return(tenors)
#Done with TenorsFromNames

def GetFREDMatrix(seriesnames,progress=False,startdate=None,enddate=None):
    #Get a matrix (rows=dates,columns=series) of data
    #from FRED database
    #See http://mortada.net/python-api-for-fred.html
    #for information on how to get the FRED (Federal
    #Reserve of St. Louis database) API, and how to get
    #an API key. The key below is Ken Winston's.
    #Inputs:
    #    seriesnames - list of strings with desired series
    #    progress - optional Boolean, print progress report if True
    #    startdate, enddate - optional 'YYYY-MM-DD' dates
    #Returns:
    #    cdates - list of yyyy-mm-dd dates
    #    ratematrix - list of time series, one time
    #                 series per exchange rate
    import pandas as pd
    import numpy as np
    import fredapi
    fred = fredapi.Fred(api_key='fd97b1fdb076ff1a86aff9b38d7a0e70')

    #Get each time series and load it into a common dataframe
    initialize=True
    for sn in seriesnames:
        if progress: print('Processing ',sn)
        fs=fred.get_series(sn,observation_start=startdate, \
                           observation_end=enddate)
        fs=fs.rename(sn)   #put the name on the column
        if initialize:
            #Set up the dataframe with the first series
            df=pd.DataFrame(fs)
            initialize=False
        else:
            #concatenate the next series to the dataframe
            df=pd.concat([df,fs],axis=1)
    
    #The dataframe has aligned the dates
    #strip out date series
    cdates=df.index.strftime('%Y-%m-%d').tolist()
    ratematrix=[list(df.iloc[i]) for i in range(len(df))]
    return(cdates,ratematrix)
#Done with GetFREDMatrix

def InterpolateCurve(tenors_in,curve_in):
    #Interpolate curve monthly and return a short
    #rate curve based on the interpolated curve
    #tenors_in has the tenors at the knot points (in years)
    #curve_in has the rate values at the knot point
    #tenors_out has the monthly tenors
    #curve_out has the rates associated with tenors_out
    #shortrates has the bootstrapped short rates
    
    curve_out=[]
    tenors_out=[]
    shortrates=[]
    idxin=0
    mnthin=round(tenors_in[idxin]*12)
    months=round(tenors_in[len(tenors_in)-1]*12)
    #Fill in curve_out every month between the knot
    #points given in curve
    #As curve is filled in, bootstrap a short rate curve
    for month in range(months):
        tenors_out.append(float(month+1)/12)
        if (month+1==mnthin):   #Are we at a knot point?
            #Copy over original curve at this point
            curve_out.append(curve_in[idxin])
            #Move indicator to next knot point
            idxin+=1
            if (idxin!=len(tenors_in)):
                #Set month number of next knot point
                mnthin=round(tenors_in[idxin]*12)
        else:   #Not at a knot point - interpolate
            timespread=tenors_in[idxin]-tenors_in[idxin-1]
            ratespread=curve_in[idxin]-curve_in[idxin-1]
            if (timespread<=0):
                curve_out.append(curve_in[idxin-1])
            else:
                #compute years between previous knot point and now
                time_to_previous_knot=(month+1)/12-tenors_in[idxin-1]
                proportion=(ratespread/timespread)*time_to_previous_knot
                curve_out.append(curve_in[idxin-1]+proportion)
        #Bootstrap a short rate curve
        short=curve_out[month]    
        if (month!=0):
            denom=tenors_out[month]-tenors_out[month-1]
            numer=curve_out[month]-curve_out[month-1]
            if (denom!=0):
                short+=tenors_out[month]*numer/denom
        shortrates.append(short)
        
    return(tenors_out,curve_out,shortrates)
#Done with InterpolateCurve

def levels_to_log_returns(cdates,ratematrix,multipliers):
    import pandas as pd
    import numpy as np
    #Convert levels to log-returns
    #First take logs of the currency levels
    #Currency exchange rates are usually expressed in the direction
    #that will make the rate > 1
    #Swissie and yen are in currency/dollar, but
    #pounds is in dollar/currency. Reverse signs
    #so everything is in dollar/currency

    #Do each currency separately to account for separate missing data patterns
    #dlgs is a list of lists of length 3 corresponding to the 3 currencies
    #The value in dlgs is nan if there is missing data for the present or
    #previous day's observation; otherwise it is the log of today/yesterday
    dlgs=[]
    for i in range(len(multipliers)):
        lgrates=[]
        previous=-1
        for t in range(len(ratematrix)):
            if pd.isna(ratematrix[t][i]) or ratematrix[t][i]<=0:
                lgrates.append(np.nan)    #Append a nan
            else:
                if previous < 0:    #This is the first data point
                    lgrates.append(np.nan)
                else:
                    lgrates.append(np.log(ratematrix[t][i]/previous)*multipliers[i])
                previous=ratematrix[t][i]
        dlgs.append(lgrates)

    #dlgs is the transpose of what we want - flip it
    dlgs=np.transpose(dlgs)

    #Delete any time periods that don't have data
    difflgs=[dl for dl in dlgs if all(pd.notna(dl))]
    lgdates=[cdates[t] for t,dl in enumerate(dlgs) if all(pd.notna(dl))]
    return(lgdates,difflgs)
#Done with levels_to_log_returns

def StatsTable(xret):
    import numpy as np
    from scipy import stats
    #Create statistics table from x vector
    #giving periodic returns or log-returns
    #Returns a vector statnames giving the names
    #of the computed statistics, and metrics giving
    #the actual statistics
    #Also returns a text array table suitable
    #for printing
    statnames=['Count','Min','Max','Mean','Median',
               'Standard Deviation','Skewness',
               'Excess Kurtosis','Jarque-Bera',
               'Chi-Squared p','Serial Correlation',
               '99% VaR','99% cVaR']
    metrics=[]
    #Item count
    metrics.append(len(xret))
    #Extremes
    metrics.append(min(xret))
    metrics.append(max(xret))
    #Mean, median
    metrics.append(np.mean(xret))
    metrics.append(np.median(xret))
    #2, 3, 4 moments
    metrics.append(np.std(xret))
    metrics.append(stats.skew(xret))
    metrics.append(stats.kurtosis(xret))
    #Jarque-Bera
    #Direct computation gives the same thing as
    #the stats.jarque_bera function
    #jb=(metrics[0]/6)*(metrics[6]**2+(metrics[7]**2)/4)
    #metrics.append(jb)
    jb=stats.jarque_bera(xret)
    metrics.append(jb[0])   #The JB statistic
    metrics.append(jb[1])   #Chi-squared test p-value
    #Serial correlation
    metrics.append(stats.pearsonr(xret[:len(xret)-1],xret[1:])[0])
    #99% VaR
    low1=np.percentile(xret,1)
    metrics.append(-low1)
    metrics.append(-np.mean([x for x in xret if x<=low1]))
    
    #Change numbers to text
    table=[]
    for i in range(len(metrics)):
        rowlist=[]
        rowlist.append(statnames[i])
        rowlist.append('%10.7f' % metrics[i])
        table.append(rowlist)
    return(statnames,metrics,table)
#Done with StatsTable

def Garch11Fit(initparams,InputData):
    import scipy.optimize as scpo
    import numpy as np
    #Fit a GARCH(1,1) model to InputData using (8.42)
    #Returns the triplet a,b,c (actually a1, b1, c) from (8.41)
    #Initial guess is the triple in initparams

    array_data=np.array(InputData)

    def GarchMaxLike(params):
        import numpy as np        
        #Implement formula 6.42
        xa,xb,xc=params
        if xa>10: xa=10
        if xb>10: xb=10
        if xc>10: xc=10
        #Use trick to force a and b between 0 and .999;
        #(a+b) less than .999; and c>0
        a=.999*np.exp(xa)/(1+np.exp(xa))
        b=(.999-a)*np.exp(xb)/(1+np.exp(xb))
        c=np.exp(xc)
        t=len(array_data)
        minimal=10**(-20)
        vargarch=np.zeros(t)

        #CHEATS!
        #Seed the variance with the whole-period variance
        #In practice we would have to have a holdout sample
        #at the beginning and roll the estimate forward.
        vargarch[0]=np.var(array_data)

        #Another cheat: take the mean over the whole period
        #and center the series on that. Hopefully the mean
        #is close to zero. Again in practice to avoid lookahead
        #we would have to roll the mean forward, using only
        #past data.
        overallmean=np.mean(array_data)
        #Compute GARCH(1,1) var's from data given parameters
        for i in range(1,t):
            #Note offset - i-1 observation of data
            #is used for i estimate of variance
            vargarch[i]=c+b*vargarch[i-1]+\
            a*(array_data[i-1]-overallmean)**2
            if vargarch[i]<=0:
                vargarch[i]=minimal
                
        #sum logs of variances
        logsum=np.sum(np.log(vargarch))
        #sum yi^2/sigma^2
        othersum=0
        for i in range(t):
            othersum+=((array_data[i]-overallmean)**2)/vargarch[i]
        #Actually -2 times (6.42) since we are minimizing
        return(logsum+othersum)
    #End of GarchMaxLike

    #Transform parameters to the form used in GarchMaxLike
    #This ensures parameters are in bounds 0<a,b<1, 0<c
    aparam=np.log(initparams[0]/(.999-initparams[0]))
    bparam=np.log(initparams[1]/(.999-initparams[0]-initparams[1]))
    cparam=np.log(initparams[2])
    xinit=[aparam,bparam,cparam]
    #Run the minimization. Constraints are built-in.
    results = scpo.minimize(GarchMaxLike,
                            xinit,
                            method='CG')
    aparam,bparam,cparam=results.x
    a=.999*np.exp(aparam)/(1+np.exp(aparam))
    b=(.999-a)*np.exp(bparam)/(1+np.exp(bparam))
    c=np.exp(cparam)

    return([a,b,c])
#Done with Garch11Fit function
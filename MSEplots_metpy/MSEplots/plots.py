import numpy as np
import matplotlib.pyplot as plt

import metpy.calc as mpcalc
from metpy.units import units
from metpy.constants import Cp_d,Lv,Rd,g 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin() # find minimum
    return (idx)

def thermo_plots(pressure,temperature,mixing_ratio):
    """"
    plots for vertical profiles of temperature, dewpoint, mixing ratio and relative humidity.
    
    Parameters
    ----------
    pressure : array-like
            Atmospheric pressure profile (surface to TOA)
    temperature: array-like
            Atmospheric temperature profile (surface to TOA)
    dewpoint: array-like
            Atmospheric dewpoint profile (surface to TOA)    
    Returns
    -------
    """
    q = mixing_ratio
    Td = mpcalc.dewpoint(mpcalc.vapor_pressure(pressure*units.mbar,q*units.kilogram/units.kilogram)).magnitude # dewpoint 
    Tp = mpcalc.parcel_profile(pressure*units.mbar,temperature[0]*units.degC,Td[0]*units.degC).to('degC').magnitude # parcel
    
    plt.figure(figsize=(12,5))
    
    lev=find_nearest(pressure,100);
    plt.subplot(1,3,1)
    plt.plot(temperature[:lev],pressure[:lev],'-ob')
    plt.plot(Td[:lev],pressure[:lev],'-og')
    plt.plot(Tp[:lev],pressure[:lev],'-or')
    plt.xlabel('Temperature [C]',fontsize=12)
    plt.ylabel('Pressure [hpa]',fontsize=12)
    plt.gca().invert_yaxis()
    plt.legend(['Temp','Temp_Dew','Temp_Parcel'],loc=1)
    plt.grid()
    
    qs = mpcalc.mixing_ratio(mpcalc.saturation_vapor_pressure(temperature*units.degC),pressure*units.mbar)
    # Relative humidity
    RH=q/qs*100; # Relative humidity

    plt.subplot(1,3,2)
    plt.plot(q[:lev],pressure[:lev],'-og')
    plt.xlabel('Mixing ratio [kg/kg]',fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid()

    plt.subplot(1,3,3)
    plt.plot(RH[:lev],pressure[:lev],'-og')
    plt.xlabel('Relative humiduty [%]',fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid()
    
    plt.tight_layout()
    return (plt)

def theta_plots(pressure,temperature,mixing_ratio):
    """
    plots for vertical profiles of potential temperature, equivalent potential temperature, 
    and saturated equivalent potential temperature

    """
    lev = find_nearest(pressure,100)
    Td = mpcalc.dewpoint(mpcalc.vapor_pressure(pressure*units.mbar,mixing_ratio*units.kilogram/units.kilogram)).magnitude # dewpoint 
    theta = mpcalc.potential_temperature(pressure*units.mbar,temperature*units.degC)
    theta_e = mpcalc.equivalent_potential_temperature(pressure*units.mbar,temperature*units.degC,Td*units.degC)
    theta_es = mpcalc.equivalent_potential_temperature(pressure*units.mbar,temperature*units.degC,temperature*units.degC)
       
    plt.figure(figsize=(7,7))
    plt.plot(theta[:lev],pressure[:lev],'-ok')
    plt.plot(theta_e[:lev],pressure[:lev],'-ob')
    plt.plot(theta_es[:lev],pressure[:lev],'-or')
    plt.xlabel('Temperature [K]',fontsize=12)
    plt.ylabel('Pressure [hpa]',fontsize=12)
    plt.gca().invert_yaxis()
    plt.legend(['$\\theta$','$\\theta_e$','$\\theta_{es}$'],loc=1)
    plt.grid()
    return (plt)

def msed_plots(pressure,temperature,mixing_ratio,altitude=None):
    """
    plotting the summarized static energy diagram with annotations and thermodynamic parameters
    """
    q  = mixing_ratio
    qs = mpcalc.mixing_ratio(mpcalc.saturation_vapor_pressure(temperature*units.degC),pressure*units.mbar)
    Td = mpcalc.dewpoint(mpcalc.vapor_pressure(pressure*units.mbar,q*units.kilogram/units.kilogram)).magnitude # dewpoint 
    Tp = mpcalc.parcel_profile(pressure*units.mbar, temperature[0]*units.degC, Td[0]*units.degC).to('degC').magnitude; # parcel

    # Altitude 
    if (altitude is None):
        altitude = np.zeros((np.size(temperature))) # surface is 0 meter
        for i in range(1,np.size(temperature)):
            altitude[i] = mpcalc.thickness_hydrostatic(pressure[:i+1]*units.mbar,temperature[:i+1]*units.degC).magnitude; # Hypsometric Eq. for height

        else:
            altitide = altitude
  
    # Static energy calculations   
    mse = mpcalc.moist_static_energy(altitude*units.meter,temperature*units.degC,q*units.kilogram/units.kilogram).magnitude
    mse_s = mpcalc.moist_static_energy(altitude*units.meter,temperature*units.degC,qs*units.kilogram/units.kilogram).magnitude
    dse = mpcalc.dry_static_energy(altitude*units.meter,temperature*units.degC)

    # Water vapor calculations
    p_PWtop = max(200*units.mbar, min(pressure*units.mbar) +1*units.mbar) # integrating until 200mb 
    cwv = mpcalc.precipitable_water(Td*units.degC,pressure*units.mbar, top=p_PWtop) # column water vapor [mm]
    cwvs = mpcalc.precipitable_water(temperature*units.degC,pressure*units.mbar, top=p_PWtop) # saturated column water vapor [mm]
    crh = (cwv/cwvs).magnitude *100. # column relative humidity [%]

    #================================================
    # plotting MSE vertical profiles
    h=plt.figure(figsize=[10,8])
    plt.plot(dse[:],pressure[:],'-k',linewidth=2)
    plt.plot(mse[:],pressure[:],'-b',linewidth=2)
    plt.plot(mse_s[:],pressure[:],'-r',linewidth=2)
    
    # mse based on different percentages of relative humidity
    qr=np.zeros((9,np.size(qs.magnitude))); mse_r=qr# container
    for i in range(9):
        qr[i,:]=qs*0.1*(i+1);
        mse_r[i,:]=(Cp_d.magnitude*(temperature+273.15)+g.magnitude*altitude+Lv.magnitude*qr[i,:])/1000;

    for i in range(9):
        plt.plot(mse_r[i,:],pressure[:],'-',color='grey',linewidth=0.7)
        plt.text(mse_r[i,3]-1,pressure[3],str((i+1)*10))
                
    # drawing LCL and LFC levels
    [lcl_pressure, lcl_temperature] = mpcalc.lcl(pressure[0]*units.mbar, temperature[0]*units.degC, Td[0]*units.degC)
    lcl_idx = np.argmin(np.abs(pressure - lcl_pressure.magnitude))
    line_lcl=np.squeeze(np.ones((1,300))*lcl_pressure);
    plt.plot(np.linspace(280,400,300),line_lcl,'--',color='orange')
    
    [lfc_pressure,lfc_idx]=mpcalc.lfc(pressure*units.mbar,temperature*units.degC,Td*units.degC)
    line_lfc=np.squeeze(np.ones((1,300))*lfc_pressure);
    plt.plot(np.linspace(280,400,300),line_lfc,'--',color='magenta')
    
    # conserved mse of air parcel arising from 1000 hpa    
    mse_p=np.squeeze(np.ones((1,np.size(temperature)))*mse[0]);
    
    # illustration of CAPE
    el_pressure,el_temperature = mpcalc.el(pressure*units.mbar,temperature*units.degC,Td*units.degC) # equilibrium level
    el_idx = np.argmin(np.abs(pressure - el_pressure.magnitude))
    [CAPE,CIN]=mpcalc.cape_cin(pressure[:el_idx]*100*units.pascal,temperature[:el_idx]*units.degC,Td[:el_idx]*units.degC,Tp[:el_idx]*units.degC)
    
    plt.plot(mse_p[:],pressure[:],color='green',linewidth=2)
    plt.fill_betweenx(pressure[lcl_idx:el_idx+1],mse_p[lcl_idx:el_idx+1],mse_s[lcl_idx:el_idx+1],interpolate=True
                    ,color='green',alpha='0.3');

    plt.fill_betweenx(pressure[:],dse[:],mse[:],color='deepskyblue',alpha='0.5')
    plt.xlim([280,380])
    plt.xlabel('Specific static energy, hs, h, S [KJ/kg]',fontsize=14)
    plt.ylabel('Pressure [hpa]',fontsize=14)
    plt.xticks(fontsize=12);plt.yticks(fontsize=12)
    plt.ylim(1030,150)
    
    # Depict Entraining parcels
    # Parcel mass solves dM/dz = eps*M, solution is M = exp(eps*Z)
    # M=1 at ground without loss of generality
    entrainment_distance = 10000., 5000., 2000. 

    for ED in entrainment_distance: 
        eps = 1.0 / (ED)
        M = np.exp(eps * (altitude-altitude[0]));

        # dM is the mass contribution at each level, with 1 at the origin level. 
        M[0] = 0
        dM = np.gradient(M)

        # parcel mass is a  sum of all the dM's at each level
        # conserved linearly-mixed variables like h are weighted averages 
        hent = np.cumsum(dM*mse) / np.cumsum(dM)

        plt.plot( hent[0:el_idx+3], pressure[0:el_idx+3], linewidth=0.5, color='g')

    # Text parts
    plt.text(290,pressure[3],'RH (%)',fontsize=11,color='k')
    plt.text(285,250,'CAPE = '+str(np.around(CAPE.magnitude,decimals=2))+' [J/kg]',fontsize=14,color='green');
    plt.text(285,300,'CIN = '+str(np.around(CIN.magnitude,decimals=2))+' [J/kg]',fontsize=14,color='green')
    plt.text(285,350,'LCL = '+str(np.around(lcl_pressure.magnitude,decimals=2))+' [hpa]',fontsize=14,color='orange');
    plt.text(285,400,'LFC = '+str(np.around(lfc_pressure.magnitude,decimals=2))+' [hpa]',fontsize=14,color='magenta');
    plt.text(285,450,'CWV = '+str(np.around(cwv.magnitude,decimals=2))+' [mm]',fontsize=14,color='deepskyblue');
    plt.text(285,500,'CRH = '+str(np.around(crh,decimals=2))+' [%]',fontsize=14,color='blue');
    plt.text(330,220,'entrain: \n 10,5,2 km',fontsize=12,color='green');
    plt.text(mse[0]-3,400,'Parcel h',fontsize=12,color='green')
    plt.legend(['dry air','moist air','saturated air'],fontsize=12,loc=1);
    return (plt)


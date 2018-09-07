import numpy as np
import matplotlib.pyplot as plt
from MSEplots.calc.thermo_calc import *

def theta_plots(pressure,altitude,temperature,dewpoint):
    """
    plots for vertical profiles of potential temperature, equivalent potential temperature, 
    and saturated equivalent potential temperature

    """
    lev=find_nearest(pressure,100)
    [theta,theta_e,theta_es]=theta_calc(pressure,altitude,temperature,dewpoint)
    
    plt.figure(figsize=(7,7))
    plt.plot(theta[:lev],pressure[:lev],'-ok')
    plt.plot(theta_e[:lev],pressure[:lev],'-ob')
    plt.plot(theta_es[:lev],pressure[:lev],'-or')
    plt.xlabel('Temperature [K]',fontsize=12)
    plt.ylabel('Pressure [hpa]',fontsize=12)
    plt.gca().invert_yaxis()
    plt.legend(['$\\theta$','$\\theta_e$','$\\theta_{es}$'],loc=1)
    plt.grid()

def thermo_plots(pressure,altitude,temperature,dewpoint):
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
    
    Temp=temperature;Temp_dew=dewpoint;
    Tp=Tp_calc(pressure,altitude,temperature,dewpoint)
    
    lev=find_nearest(pressure,100)
    
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,3,1)
    plt.plot(Temp[:lev],pressure[:lev],'-ob')
    plt.plot(Temp_dew[:lev],pressure[:lev],'-og')
    plt.plot(Tp[:lev],pressure[:lev],'-or')
    plt.xlabel('Temperature [C]',fontsize=12)
    plt.ylabel('Pressure [hpa]',fontsize=12)
    plt.gca().invert_yaxis()
    plt.legend(['Temp','Temp_Dew','Temp_Parcel'],loc=1)
    plt.grid()

    [q,qs]=q_calc(pressure,Temp,Temp_dew)
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
    plt.show()

def msed_plots(pressure,altitude,temperature,dewpoint):
    """
    plotting the summarized static energy diagram with annotations and thermodynamic parameters
    """
    Cp=1004;
    Lv=(2500.8-2.36*temperature+0.0016*temperature**2-0.00006*temperature**3)*1000;
    
    [dse,mse,mse_s]=mse_calc(pressure,altitude,temperature,dewpoint)
    # MSE vertical profiles
    h=plt.figure(figsize=[10,8])
    plt.plot(dse[:],pressure[:],'-k',linewidth=2)
    plt.plot(mse[:],pressure[:],'-b',linewidth=2)
    plt.plot(mse_s[:],pressure[:],'-r',linewidth=2)

    # different Relative humidity 
    [q,qs]=q_calc(pressure,temperature,dewpoint)
    qr=np.zeros((9,np.size(qs))); mse_r=qr# container
    for i in range(9):
        qr[i,:]=qs*0.1*(i+1);
        mse_r[i,:]=(Cp*(temperature+273.15)+9.8*altitude+Lv*qr[i,:])/1000;

    for i in range(9):
        plt.plot(mse_r[i,:],pressure[:],'-',color='grey',linewidth=0.7)
        plt.text(mse_r[i,3]-1,pressure[3],str((i+1)*10))

    # drawing LCL
    [LCL,lcl_idx]=lcl_calc(pressure,altitude,temperature,dewpoint)
    line_lcl=np.squeeze(np.ones((1,300))*LCL);
    plt.plot(np.linspace(280,400,300),line_lcl,'--',color='orange')
    
    [LFC,lfc_idx]=lfc_calc(pressure,altitude,temperature,dewpoint)
    line_lfc=np.squeeze(np.ones((1,300))*LFC);
    plt.plot(np.linspace(280,400,300),line_lfc,'--',color='magenta')
    
    # conserved mse of air parcel arising from 1000 hpa    
    mse_p=np.squeeze(np.ones((1,np.size(temperature)))*mse[0]);
    
    # illustration of CAPE
    [EL,EL_idx]=el_calc(pressure,altitude,temperature,dewpoint)
    [CAPE,CIN]=cape_cin_calc(pressure,altitude,temperature,dewpoint)
    cwv=cwv_calc(pressure,altitude,temperature,dewpoint)
    
    plt.plot(mse_p[:],pressure[:],color='green',linewidth=2)
    plt.fill_betweenx(pressure[lcl_idx:EL_idx+1],mse_p[lcl_idx:EL_idx+1],mse_s[lcl_idx:EL_idx+1],interpolate=True
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

        plt.plot( hent[0:EL_idx+3], pressure[0:EL_idx+3], linewidth=0.5, color='g')

    # Text parts
    plt.text(290,pressure[3],'RH (%)',fontsize=11,color='k')
    plt.text(285,250,'CAPE = '+str(np.around(CAPE,decimals=2))+' [J/kg]',fontsize=14,color='green');
    plt.text(285,300,'CIN = '+str(np.around(CIN,decimals=2))+' [J/kg]',fontsize=14,color='green')
    plt.text(285,350,'LCL = '+str(np.around(LCL,decimals=2))+' [hpa]',fontsize=14,color='orange');
    plt.text(285,400,'LFC = '+str(np.around(LFC,decimals=2))+' [hpa]',fontsize=14,color='magenta');
    plt.text(285,450,'CWV = '+str(np.around(cwv,decimals=2))+' [mm]',fontsize=14,color='deepskyblue');
    plt.text(330,220,'entrain: \n 10,5,2 km',fontsize=12,color='green');
    plt.text(mse[0]-3,400,'Parcel h',fontsize=12,color='green')
    plt.legend(['dry air','moist air','saturated air'],fontsize=12,loc=1);

    plt.show()


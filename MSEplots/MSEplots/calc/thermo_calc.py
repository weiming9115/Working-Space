import numpy as np
import metpy.calc as mpcalc
from metpy.units import units

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin() # find minimum  
    return (idx)

def Td_calc(pressure,temperature,mixing_ratio):
    """
    calculating dewpoint [C] from mixing ratio [kg/kg]
    """
    Temp=temperature;q=mixing_ratio;
    Td=np.zeros((np.size(Temp)));
    for z in np.linspace(0,np.size(Temp)-1,np.size(Temp),dtype=int):
        tmp=np.linspace(Temp[z]-40,Temp[z],801)
    # CC-equation for specific humidity retrival
        for x in tmp:
            e=6.1094*np.exp(17.625*x/(x+243.04))
            qs=0.622*e/(pressure[z]-e)
            if np.abs((qs-q[z])/qs) < 0.05:
                Tdz=x;
                break
        Td[z]=Tdz
    return (Td)    

def q_calc(pressure,temperature,dewpoint):
    """"
    calculating mixing ratio (q) from dewpoint
    """
    Temp=temperature;Temp_dew=dewpoint;
    q=np.zeros((np.size(Temp)));qs=np.zeros((np.size(Temp)));
    for z in np.linspace(0,np.size(Temp)-1,np.size(Temp),dtype=int):
    # CC-equation for specific humidity retrival
        e=6.1094*np.exp(17.625*Temp_dew[z]/(Temp_dew[z]+243.04))
        qz=0.622*e/(pressure[z]-e)
        es=6.1094*np.exp(17.625*Temp[z]/(Temp[z]+243.04))
        qsz=0.622*es/(pressure[z]-es)
        if qz < 0 or qz < 1e-7 or qz > 3e-2:
            q[z]=0;qs[z]=0 
        else:
            q[z]=qz;qs[z]=qsz
    return (q,qs)

def mse_calc(pressure,altitude,temperature,dewpoint):
    """
    [dse,mse,mse_s] = mse_calc(pressure, altitude, temperature, dewpoint)
        dse   --> dry static energy
        mse   --> moist static energy
        mse_s --> saturatued moist static energy
    
    calculatiing dry static energy, moist static energy and saturated moist static energy
    """
    pressure=pressure;Temp=temperature;Temp_dew=dewpoint
    # Parameters
    Cp=1004; # J/(kg K)
    Lv=(2500.8-2.36*Temp+0.0016*Temp**2-0.00006*Temp**3)*1000; # latent heat J/kg
    Rd=287.15 # J/kg
    
    # (a). Calculate equivalent potential temperature (theta_e) and saturated theta_e
    [q,qs]=q_calc(pressure,Temp,Temp_dew)
    theta_e=(Temp+273.15+Lv*q/Cp)*(1000/pressure)**(Rd/Cp);
    es=6.1094*np.exp(17.625*Temp/(Temp+243.04));
    theta_es=(Temp+273.15+Lv*qs/Cp)*(1000/pressure)**(Rd/Cp);
    # dry air: theta_e-->theta
    theta=(Temp+273.15)*(1000/pressure)**(Rd/Cp);

    # (b). static energy 
    dse=(Cp*(Temp+273.15)+9.8*altitude)/1000 # dry static energy [KJ/kg]
    mse=(Cp*(Temp+273.15)+9.8*altitude+Lv*q)/1000; # moist static energy [KJ/kg]
    mse_s=(Cp*(Temp+273.15)+9.8*altitude+Lv*qs)/1000; # moist static energy [KJ/kg]
    
    return(dse,mse,mse_s)

def theta_calc(pressure,altitude,temperature,dewpoint):
    """
    """
    pressure=pressure;Temp=temperature;Temp_dew=dewpoint
    # Parameters
    Cp=1004; # J/(kg K)
    Lv=(2500.8-2.36*Temp+0.0016*Temp**2-0.00006*Temp**3)*1000; # latent heat J/kg
    Rd=287.15 # J/kg
    
    # (a). Calculate equivalent potential temperature (theta_e) and saturated theta_e
    [q,qs]=q_calc(pressure,Temp,Temp_dew)
    theta_e=(Temp+273.15+Lv*q/Cp)*(1000/pressure)**(Rd/Cp);
    es=6.1094*np.exp(17.625*Temp/(Temp+243.04));
    theta_es=(Temp+273.15+Lv*qs/Cp)*(1000/pressure)**(Rd/Cp);
    # dry air: theta_e-->theta
    theta=(Temp+273.15)*(1000/pressure)**(Rd/Cp);
    
    return(theta,theta_e,theta_es)

def cwv_calc(pressure,altitude,temperature,dewpoint):
    """
    calculatiing vertically integrated water vapor [mm]
    """
    q=q_calc(pressure,temperature,dewpoint)[0]
    
    cwv=np.trapz(q,altitude); # column water vapor [mm]
    
    return cwv

def lcl_calc(pressure,altitude,temperature,dewpoint):
    """
    [LCL,idx]=lcl_calc(pressure,altitude,temperature,dewpoint)
    calculating interpolated lifting condesation level (LCL) and the corresponding index
    """    
    Temp=temperature;q=q_calc(pressure,temperature,dewpoint)[0];
    Cp=1004;
    
    # (d). Lifting condenstation level (LCL)

    qs=q_calc(pressure,temperature,dewpoint)[1]
    for x in np.linspace(1,20,20,dtype=int):
        Tpd=Temp[0]-9.8/Cp*(altitude[x]-altitude[0])
        esp=6.1094*np.exp(17.625*Tpd/(Tpd+243.04));
        qsp=0.622*esp/(pressure[x]-esp)   
        if q[0]-qsp > 0: # when q is larger than qs for the air parcel -> condensate
            LCL=np.interp(q[0],[qsp,qs[0]],[pressure[x],pressure[0]]);
            lcl_idx=x
            break
        tmp=qsp
        
    return (LCL,lcl_idx)

def Tp_calc(pressure,altitude,temperature,dewpoint):
    """"
    calculating the temperature of air parcel lifting from the bottom level of sounding following
    the pesudo moist adiabatic process.
    """
    Temp=temperature;
    lcl_idx=lcl_calc(pressure,altitude,temperature,dewpoint)[1];
    Cp=1004;Rd=287.15
    
    # parcel temperature below LCL
    Tp=np.zeros([np.shape(Temp)[0]]);
    if lcl_idx == 0:
        Tp[0]=Temp[0]
    else:
        Tp[:lcl_idx]=Temp[:lcl_idx]
        for x in range(lcl_idx):
            Tp[x+1]=Tp[x]-9.8/Cp*(altitude[x+1]-altitude[x])

    # parcel temperature above LCL
    a=lcl_idx; #index for LCL

    for x in np.linspace(a,np.shape(Tp)[0]-2,np.shape(Tp)[0]-a-1,dtype=int):    
        esp=6.1094*np.exp(17.625*Tp[x]/(Tp[x]+243.04))
        qsp=0.622*esp/(pressure[x]-esp)   
        Lvp=(2500.8-2.36*Tp[x]+0.0016*Tp[x]**2-0.00006*Tp[x]**3)*1000; # latent heat J/kg
        lm=9.8*(1+Lvp*qsp/(Rd*(Tp[x]+273.15)))/(Cp+Lvp**2*qsp*0.622/(Rd*(Tp[x]+273.15)**2)) # moist adiabatic lapse rate
        if qsp ==0:
            lev_top=x;break
        Tp[x+1]=Tp[x]-lm*(altitude[x+1]-altitude[x])
    return Tp

def lfc_calc(pressure,altitude,temperature,dewpoint):
    """
    calculating the level of free convection (LFC)
    """
    Tp=Tp_calc(pressure,altitude,temperature,dewpoint)
    Temp=temperature
    # finding EL
    #k=0
    lcl_idx=lcl_calc(pressure,altitude,temperature,dewpoint)[1]
    for index, item in enumerate(Tp-Temp):
         if index > lcl_idx and item > 0:
            LFC=pressure[index];lfc_idx=index;break
    
    return (LFC,lfc_idx)

def el_calc(pressure,altitude,temperature,dewpoint):
    """
    [EL,idx]=el_calc(pressure,altitude,temperature)
    calculating the equilibrium level of lifting air parcel and the corresponding index
    """
    Tp=Tp_calc(pressure,altitude,temperature,dewpoint)
    Temp=temperature
    # finding EL
    #k=0
    lcl_idx=lcl_calc(pressure,altitude,temperature,dewpoint)[1]
    for index, item in enumerate(Tp-Temp):
         if index > lcl_idx and item > 0:
            LFC=pressure[index];lfc_idx=index;break
    
    for index, item in enumerate(Tp-Temp):
         if index > lfc_idx and item < 0:
            EL=pressure[index];EL_idx=index;break
        
    return (EL,EL_idx)

def cape_cin_calc(pressure,altitude,temperature,dewpoint):
    """
    calculating CAPE and CIN based on Metpy (metpy.calc.cape_cin)
    """
    import metpy.calc as mpcalc
    from metpy.units import units
    
    Temp=temperature;Temp_dew=dewpoint
    Tp=Tp_calc(pressure,altitude,temperature,dewpoint)
    lev_top=el_calc(pressure,altitude,temperature,dewpoint)[1]
    # calculating CAPE and CIN based on MetPy 
    [CAPE,CIN]=mpcalc.cape_cin(pressure[:lev_top]*1000*units.pascal,Temp[:lev_top]*units.degC,Temp_dew[:lev_top]*units.degC,Tp[:lev_top]*units.degC)
    return (CAPE.magnitude,CIN.magnitude)



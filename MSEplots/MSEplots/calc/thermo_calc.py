import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from metpy.constants import Cp_d,Lv,Rd,g 

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
            if np.abs((qs-q[z])/qs) < 0.01:
                Tdz=x;
                break
        Td[z]=Tdz
    return (Td)    

def q_calc(pressure,temperature,dewpoint):
    """"
    calculating mixing ratio (q) and saturated mixing ratio (qs) from dewpoint
    """
    Temp=temperature;Temp_dew=dewpoint;
    q = mpcalc.mixing_ratio(mpcalc.saturation_vapor_pressure(Temp_dew*units.degC),pressure*100*units.pascal)
    qs = mpcalc.mixing_ratio(mpcalc.saturation_vapor_pressure(Temp*units.degC),pressure*100*units.pascal)
    return (q.magnitude,qs.magnitude)

def mse_calc(pressure,temperature,mixing_ratio,altitude=None):
    """
    [dse,mse,mse_s] = mse_calc(pressure, temperature, mixing_ratio, altitude [optional])
        dse   --> dry static energy
        mse   --> moist static energy
        mse_s --> saturatued moist static energy
    
    calculatiing dry static energy, moist static energy and saturated moist static energy
    using the hypsometric equation to derive the altitude at each pressure level (assume H_sfc=0 m)
    """
    pressure=pressure;Temp=temperature;q=mixing_ratio
    qs = mpcalc.mixing_ratio(mpcalc.saturation_vapor_pressure(Temp*units.degC),pressure*100*units.pascal).magnitude
    
    if (altitude is None):
        altitude = np.zeros((np.size(Temp))) # surface is 0 meter
        for i in range(1,np.size(Temp)):
            altitude[i]=mpcalc.thickness_hydrostatic(pressure[:i+1]*units.mbar,Temp[:i+1]*units.degC).magnitude; # Hypsometric Eq. for height
    else:
        altitude=altitude; # meter
    
    # static energy 
    dse=(Cp_d.magnitude*(Temp+273.15)+g.magnitude*altitude)/1000 # dry static energy [KJ/kg]
    mse=(Cp_d.magnitude*(Temp+273.15)+g.magnitude*altitude+Lv.magnitude*q)/1000; # moist static energy [KJ/kg]
    mse_s=(Cp_d.magnitude*(Temp+273.15)+g.magnitude*altitude+Lv.magnitude*qs)/1000; # moist static energy [KJ/kg]
    
    return(dse,mse,mse_s)

def theta_calc(pressure,temperature,mixing_ratio):
    """
    [theta,theta_e,theta_es] = theta_calc(pressure, temperature, mixing_ratio)
    dse   --> dry static energy
    mse   --> moist static energy
    mse_s --> saturatued moist static energy
    """    
    pressure=pressure;Temp=temperature;q=mixing_ratio
    
    # Calculate equivalent potential temperature (theta_e) and saturated theta_e
    qs=mpcalc.mixing_ratio(mpcalc.saturation_vapor_pressure(Temp*units.degC),pressure*100*units.pascal)
    theta_e=(Temp+273.15+Lv.magnitude*q/Cp_d.magnitude)*(1000/pressure)**(Rd.magnitude*1000/Cp_d.magnitude);
    es=6.1094*np.exp(17.625*Temp/(Temp+243.04));
    theta_es=(Temp+273.15+Lv.magnitude*qs/Cp_d.magnitude)*(1000/pressure)**(Rd.magnitude*1000/Cp_d.magnitude);
    # dry air: theta_e-->theta
    theta=(Temp+273.15)*(1000/pressure)**(Rd.magnitude*1000/Cp_d.magnitude);
    
    return(theta,theta_e,theta_es)

def cwv_calc(pressure,temperature,mixing_ratio):
    """
    [cwv,cwvs,crh] = cwv_calc(pressure,temperature,mixing_ratio)
    
    calculatiing column water vapor, saturated column water vapor and column relative humidity
    """
    Temp=temperature;
    Td=Td_calc(pressure,temperature,mixing_ratio)
    p_PWtop = max(200*units.mbar, min(pressure*units.mbar) +1*units.mbar) # integrating until 200mb 
    cwv = mpcalc.precipitable_water(Td*units.degC,pressure*units.mbar, top=p_PWtop)
    cwvs = mpcalc.precipitable_water(Temp*units.degC,pressure*units.mbar, top=p_PWtop)
    crh = (cwv/cwvs).magnitude *100. 
    
    return (cwv,cwvs,crh)

def lcl_calc(pressure,temperature,mixing_ratio):
    """
    [LCL,idx] = lcl_calc(pressure,altitude,temperature,dewpoint)
    
    calculating interpolated lifting condesation level (LCL) and the corresponding index
    """    
    Temp=temperature;
    Td=Td_calc(pressure,temperature,mixing_ratio);

    [lcl_pressure, lcl_temperature] = mpcalc.lcl(pressure[0]*units.mbar, Temp[0]*units.degC, Td[0]*units.degC)
    lcl_index = np.argmin(np.abs(pressure - lcl_pressure.magnitude))

    return(lcl_pressure, lcl_index)

def Tp_calc(pressure,temperature,mixing_ratio):
    """"
    calculating the temperature of air parcel lifting from the bottom level of sounding following
    the pesudo moist adiabatic process.
    """
    Temp=temperature;
    Td = Td_calc(pressure,temperature,mixing_ratio);
    Tp = mpcalc.parcel_profile(pressure*units.mbar, Temp[0]*units.degC, Td[0]*units.degC).to('degC');
    
    return Tp.magnitude

def lfc_calc(pressure,temperature,mixing_ratio):
    """
    calculating the level of free convection (LFC) and the corresponding index
    """
    Temp=temperature
    Td=Td_calc(pressure,temperature,mixing_ratio);
    [lfc_pressure,lfc_temperature] = mpcalc.lfc(pressure*units.mbar,Temp*units.degC,Td*units.degC)
    lfc_idx = np.argmin(np.abs(pressure - lfc_pressure.magnitude))
    
    return (lfc_pressure,lfc_idx)

def el_calc(pressure,temperature,mixing_ratio):
    """
    [EL,idx] = el_calc(pressure,altitude,temperature)
    calculating the equilibrium level of lifting air parcel and the corresponding index
    """
    Temp=temperature
    Td = Td_calc(pressure,temperature,mixing_ratio);
    el_pressure,el_temperature = mpcalc.el(pressure*units.mbar,Temp*units.degC,Td*units.degC)
    el_index = np.argmin(np.abs(pressure - el_pressure.magnitude))
    
    return(el_pressure, el_index)

def cape_cin_calc(pressure,temperature,mixing_ratio):
    """
    [cape,cin] = cape_cin_calc(pressure,temperature,mixing_ratio)
    calculating CAPE and CIN 
    """
    Temp=temperature
    Td=Td_calc(pressure,temperature,mixing_ratio)
    Tp=Tp_calc(pressure,temperature,mixing_ratio)
    lev_top=el_calc(pressure,Temp,mixing_ratio)[1]
    # calculating CAPE and CIN based on MetPy 
    [CAPE,CIN]=mpcalc.cape_cin(pressure[:lev_top]*100*units.pascal,Temp[:lev_top]*units.degC,Td[:lev_top]*units.degC,Tp[:lev_top]*units.degC)
    
    return (CAPE,CIN)


### MSEplots project
#### pip install MSEplots-pkg
------
A package built for the MSE analysis of sounding data/ model output which provides required vertical profiles of thermodynamic parameters.

```python
from MSEplots import plots as mpt
:
mpt.msedplot(T,P,q)
```
<img src="https://github.com/weiming9115/Working-Space/blob/master/MSEplots_metpy/demo.png" width="600" height="400">

1. Required paramters: Air temperature, Mixing ratio, Pressure, Altitude [optional]. NOT specifically for sounding data!
2. Functions are provided for deriving thermodynamic variables eg. potential tmeperature and static energy. All calculations included depend on the metpy.calc.thermo module.
(https://unidata.github.io/MetPy/latest/_modules/metpy/calc/thermo.html)
3. Plotting options: thermo.plots, theta.plots, and mesd_plots

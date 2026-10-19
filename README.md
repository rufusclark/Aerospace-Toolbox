# Aerospace Toolbox

> Under active development. Please report any bugs and suggest any features/bug fixes. Changes may alter/break APIs

A collection of tools and scripts for the design and analysis of aerospace vehicles.

## Core

* Implements a wrapper around [xFoil](https://web.mit.edu/drela/Public/web/xfoil/) (2D Airfoil Aerodynamic Simulation Program) to predict the performance of airfoils programmatically and compare them efficiently.
* Implements the United States Standard Atmosphere 1976 based on the [ussa1976](https://pypi.org/project/ussa1976/) library with additional extensions and plotting to determine atmospheric properties up to 1000 km.

## Rocketry

* Supports 6-degree-of-freedom flight simulations using [rocketpy](https://docs.rocketpy.org/en/latest/index.html)
* Implements an extensible motor object model and supporting functions to support the creation of custom rocket motors and amending existing motors.
* Supports downloading rocket motor data from [thrustcurve.org](https://www.thrustcurve.org/) and importing into motor object model with efficient caching.
* Implements methods to select the optimal rocket motor for a rocket and determine the optimal fixed or variable thrust curve to maximise altitude within a specific impulse limit.
* Supports scripting [RASAero II](https://www.rasaero.com/) (Aerodynamic simulation software for rockets) with support for automatically estimating aerodynamic properties directly from a defined rocket object model, including efficient caching to improve running speed
* Provides additional plotting and information from rocketpy simulation and automatic error handling

## Under active development

* Support for stochastic monte-carlo simulations using predefined and user-defined input distributions
* Support for simulating different failure modes with simulation flags

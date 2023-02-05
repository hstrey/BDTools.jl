# Instructions

1. Install Pluto notebooks package
    - `] add Pluto`
2. Configure dependencies
    - Activate notebook environment
        - `Pluto.activate_notebook_environment("Fitting_Time_Series.jl")`
    - Configure `BDTools` dependency
        - `] develop --local ..`
    - Instantiate project
        - `] instantiate`
3. Run Pluto
    - `Pluto.run()`
4. Provide path to Phantom data
    - Change constant `DATA_DIR` in the notebook to point to the location with phantom data.
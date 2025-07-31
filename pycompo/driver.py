# 1) Read in data

# 2) BG trend removal and filtering
    # a) Substract climatology
    # b) Scale separation using a Gaussian filter

# 3) Restrict data to spatial region that should be analyzed

# 4) Detect SST anomaly features

# 5) Filter features that should be used for composites:
    # a) Size-based sampling
    # b) Geographic sampling (only within a monthly moving window of the 48mm-contour of PRW)
    # c) Analysis-time sampling
    # d) BG-wind-based sampling

# 6) Create feature-centric Cartesian coordinate
    # a) Coordinate transformation to Cartesian coordinate
    # b) Rotation to align major axis with abcissa (in geophys. coord. space)
    # c) Normalize with major axis length (in geophys. coord. space)

# 7) Create wind-aligned normalized coordinate system

# 8) Remap to the same Cartesian coordinate system to be able to align data and create composites

# 9) Construction of feature-based composites
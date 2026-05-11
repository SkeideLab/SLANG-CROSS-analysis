from   pathlib import Path
from   statsmodels.stats.multitest import multipletests
from   statsmodels.stats.anova import anova_lm
from   scipy.stats import pearsonr
from   itertools import combinations
from   nilearn import plotting, image, datasets, surface
from   matplotlib.colors import ListedColormap, to_rgba
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib import cm
from scipy.stats import rankdata
import ptitprince as pt
from scipy import stats
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import math
from collections import Counter
from nilearn.surface import load_surf_mesh
import matplotlib.patches as mpatches
import pingouin as pg
from matplotlib.patches import Patch
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
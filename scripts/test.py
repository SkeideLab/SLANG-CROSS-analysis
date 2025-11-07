import matplotlib.pyplot as plt
import numpy as np

# %% Example Figure 1, 2, 3
groups    = ['1', '2', '4']  # x-axis
visual_w  = [300, 350, 320]  # y-values in mm
spoken_w  = [500, 630, 1050]

# Plot lines
plt.plot(groups, visual_w, marker='o', label='Visual word')
plt.plot(groups, spoken_w, marker='s', label='Auditory word')

# Add labels and title
plt.xlabel('Grades')
plt.ylabel(r'Total volume (mm$^3$)')
plt.title('Left hemisphere (MTG)')
plt.ylim(0,2000)
plt.legend()
plt.grid(False)
plt.savefig('left.png', dpi=300, bbox_inches='tight')  # PNG format, high resolution
plt.show()

# %% Example Figure 4
# %% Example Figure 1, 2, 3
groups    = ['1', '2', '4']  # x-axis
corr      = [0.2, 0.25, 0.3] # y-values in r

# Plot lines
plt.plot(groups, corr, marker='o', color='purple')

# Add labels and title
plt.xlabel('Grades')
plt.ylabel('Correlation (r)')
plt.title('Right hemisphere (pSTG)')
plt.ylim(0,1)
plt.legend()
plt.grid(False)
plt.savefig('right_corr.png', dpi=300, bbox_inches='tight')  # PNG format, high resolution
plt.show()
# %%

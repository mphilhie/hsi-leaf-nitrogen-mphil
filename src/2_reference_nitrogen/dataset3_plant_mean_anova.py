import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv("Dataset3_plant_means.csv")

model = ols('N_mean ~ C(Treatment)', data=df).fit()
anova = sm.stats.anova_lm(model, typ=2)
print(anova)

tukey = pairwise_tukeyhsd(df['N_mean'], df['Treatment'])
print(tukey)

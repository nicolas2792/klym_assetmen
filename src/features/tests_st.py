import pandas as pd
import scipy.stats as stats 


def anova_test(df, col_1, col_2, alpha = 0.05):
    df = df[[col_1,col_2]]
    data = [['Between Groups', '', '', '', '', '', ''], ['Within Groups', '', '', '', '', '', ''], ['Total', '', '', '', '', '', '']] 
    anova_table = pd.DataFrame(data, columns = ['Source of Variation', 'SS', 'df', 'MS', 'F', 'P-value', 'F crit']) 
    anova_table.set_index('Source of Variation', inplace = True)

    # calculate SSTR and update anova table
    x_bar = df[col_1].mean()
    SSTR = df.groupby(col_2).count() * (df.groupby(col_2).mean() - x_bar)**2
    anova_table['SS']['Between Groups'] = SSTR[col_1].sum()

    # calculate SSE and update anova table
    SSE = (df.groupby(col_2).count() - 1) * df.groupby(col_2).std()**2
    anova_table['SS']['Within Groups'] = SSE[col_1].sum()

    # calculate SSTR and update anova table
    SSTR = SSTR[col_1].sum() + SSE[col_1].sum()
    anova_table['SS']['Total'] = SSTR

    # update degree of freedom
    anova_table['df']['Between Groups'] = df[col_2].nunique() - 1
    anova_table['df']['Within Groups'] = df.shape[0] - df[col_2].nunique()
    anova_table['df']['Total'] = df.shape[0] - 1

    # calculate MS
    anova_table['MS'] = anova_table['SS'] / anova_table['df']

    # calculate F 
    F = anova_table['MS']['Between Groups'] / anova_table['MS']['Within Groups']
    anova_table['F']['Between Groups'] = F

    # p-value
    anova_table['P-value']['Between Groups'] = 1 - (stats.f.cdf(F, anova_table['df']['Between Groups'], anova_table['df']['Within Groups']))*2

    # F critical 
    
   
    if anova_table['P-value']['Between Groups'] < alpha:
        return 'rejected, there are difference mean'
    else:
        return 'no differences'
    
    
def chi_squared_test(df, col_1, col_2, alpha = 0.05):
    
        data_crosstab = pd.crosstab(df[col_1],
                                df[col_2],
                            margins=True, margins_name="Total")

        chi_square = 0
        rows = df[col_1].unique()
        columns = df[col_2].unique()
        for i in columns:
            for j in rows:
                O = data_crosstab[i][j]
                E = data_crosstab[i]['Total'] * data_crosstab['Total'][j] / data_crosstab['Total']['Total']
                chi_square += (O-E)**2/E

        p_value = 1 -( stats.chi2.cdf(chi_square, (len(rows)-1)*(len(columns)-1)))
        
        if p_value < alpha:
            return f'variables ar not simimilars'
        else:
            return f'potencially high correlation'
    
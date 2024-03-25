import pandas as pd
import scipy.stats as stats 
#from tests_st import anova_test, chi_squared_test
from sklearn.ensemble import RandomForestRegressor



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
    

class feature_selection:
    def __init__(self, file):
        self.file= file
        self.list_cat = None
        self.list_num = None
        self.cat_final = None
        self.business_vars=None
        
    def potencial_corr(self, categorical_list):
        dic_chi_pvalue = {'cols':[], 'pvalue':[]}
        support_list = [i for i in categorical_list]
        for cols in categorical_list:
            for cols_other in support_list:
                if cols == cols_other or  cols =='model_brand' or cols_other == 'model_brand':
                    pass
                else :
                    dic_chi_pvalue['cols'].append(cols)
                    dic_chi_pvalue['pvalue'].append(chi_squared_test(self.file, cols, cols_other, alpha = 0.05))
            support_list.pop(support_list.index(cols)) 
        
        df = pd.DataFrame(dic_chi_pvalue)
        df = df[df['pvalue']=='variables ar not simimilars']
        list_vars = list(df['cols'].unique())
        self.list_cat =list_vars
        return self.list_cat
    
    def numercial_selection(self, list_numerical):
        df= self.file.dropna()
        dic_chi_pvalue = {'cols':[], 'pvalue':[]}
        support_list = [i for i in list_numerical]
        for col in list_numerical:
            for col2 in support_list:
                spearman_correlation, _ = stats.spearmanr(df[col], df[col2])
                dic_chi_pvalue['cols'].append(col)
                dic_chi_pvalue['pvalue'].append(abs(spearman_correlation))
            support_list.pop(support_list.index(col))
        
        df_fin = pd.DataFrame(dic_chi_pvalue)
        df_fin = df_fin[df_fin['pvalue']<0.5]
        list_vars = df_fin['cols'].unique().tolist()
        self.list_num =list_vars
        return self.list_num 
    
    def cat_selection_var(self):
        dic_anova_pvalue = {'cols':[], 'pvalue':[]}
        for cols in self.list_cat:
            dic_anova_pvalue['cols'].append(cols)
            dic_anova_pvalue['pvalue'].append(anova_test(self.file, 'price', cols))

        df = pd.DataFrame(dic_anova_pvalue)
        df = df[df['pvalue']!='no differences']
        self.cat_final = df['cols'].unique().tolist()
        return self.cat_final
    
    def business_variables(self):
        file_path=r'C:\Users\super\OneDrive\Escritorio\klym_project\data\external\bussines_vars.txt'
        with open(file_path, "r") as file:
        
            file_contents = file.read()
        
            lines = file_contents.splitlines()
            
            self.business_vars= lines
        return lines
    
    def final_selection(self):
        df_support = self.file
        print(self.cat_final+ self.list_num+self.business_vars)
        df_support = df_support[self.cat_final+ self.list_num]
        df_support = pd.get_dummies(df_support,columns=self.cat_final)
        df_support['dummi_variable'] =1
        x = df_support.drop(columns='price')
        y = df_support[['price']]
        model_selection = RandomForestRegressor()
        model_selection.fit(x,y)

        feature_importances = model_selection.feature_importances_

        importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances})

        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        dummi = importance_df.loc[importance_df['Feature']=='dummi_variable', 'Importance'].iloc[0]
        
        importance_df = importance_df[importance_df['Importance']>dummi]
        
        return importance_df['Feature'].unique().tolist()
    




        
                
                

        
    

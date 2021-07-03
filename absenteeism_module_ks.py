# import all libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


class absenteeism_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
        
        
         # take a data file (*.csv) and preprocess it in the same way as in the lectures
        def load_and_clean_data(self, data_file):
             # import the data
            df = pd.read_csv(data_file,delimiter=',')
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
             # drop the 'ID' column
            df=df.drop('ID', axis=1)
            # to preserve the code we've created in the previous section, we will add a column with 'NaN' strings
            df['Absenteeism Time in Hours'] = 'NaN'
            df['Date'] = pd.to_datetime(df['Date'])
            
            # create a list with month values retrieved from the 'Date' column
            list_months =[]
            for i in range(df.shape[0]):
                list_months.append(df['Date'][i].month)
            # insert the values in a new column in df, called 'Month Value'   
            df['Month Value']=list_months
            
            # create a new feature called 'Day of the Week'
            df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())
            
            # drop the 'Date' column from df
            df=df.drop(['Date'], axis=1)
            
            # re-order the columns in df
            columns_new = ['Reason for Absence','Day of the Week','Month Value', 'Transportation Expense', 'Distance to Work',
                            'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education','Children', 'Pets', 
                           'Absenteeism Time in Hours']
            
            df=df[columns_new]
            
            # map 'Education' variables; the result is a dummy
            df['Education']=df['Education'].map({1:0,2:1,3:1,4:1})
            
           
            # drop the variables we decide we don't need
            df = df.drop(['Absenteeism Time in Hours','Distance to Work','Daily Work Load Average'],axis=1)
            
            #the inputs that need to be scaled
            unscaled_inputs=df.drop(['Education'],axis=1)
            
            #stores all the names of columns of unscaled inputs
            column_val = unscaled_inputs.columns.values
            
            #our scaled inputs
            scaled_inputs= self.scaler.transform(unscaled_inputs)
            
            #new dataframe
            df1= pd.DataFrame(columns=column_val,data=scaled_inputs)
            
            
            df1['Education'] = df['Education']
            
            #create dummies for reasons of absence column
            reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
            
            reason_type1 = reason_columns.loc[:,1:14].max(axis=1)
            reason_type2 = reason_columns.loc[:,15:17].max(axis=1)
            reason_type3 = reason_columns.loc[:,18:21].max(axis=1)
            reason_type4 = reason_columns.loc[:,22:].max(axis=1)
            
            #add the dummy columns to the dataframe
            df2= pd.concat([df1,reason_type1,reason_type2,reason_type3,reason_type4], axis=1)
            
            #rename and reorder the columns
            column_values = ['Reason for Absence', 'Day of the Week','Month Value', 'Transportation Expense',
                            'Age', 'Body Mass Index', 'Children', 'Pets', 'Education',
                             'Reason_1','Reason_2','Reason_3',
                            'Reason_4']
            df2.columns=column_values
            
            #drop the reason for absence column
            df2 =df2.drop(['Reason for Absence'], axis=1)
            
            
            columns_arranged = ['Reason_1','Reason_2','Reason_3',
                                'Reason_4','Day of the Week','Month Value', 'Transportation Expense', 'Age', 'Body Mass Index',
                                'Education', 'Children', 'Pets']
            
            df2=df2[columns_arranged]
            
            # replace the NaN values if any
            df2 = df2.fillna(value=0)
            
            
            
            
            ## a seperate dataframe for pre processed data
            df3=df.drop(['Education'],axis=1)
            df3['Education'] = df['Education']
            df3= pd.concat([df3,reason_type1,reason_type2,reason_type3,reason_type4], axis=1)
            
            column_values = ['Reason for Absence','Day of the Week', 'Month Value', 'Transportation Expense',
                            'Age', 'Body Mass Index', 'Children', 'Pets', 'Education',
                             'Reason_1','Reason_2','Reason_3',
                            'Reason_4']
            
            df3.columns=column_values
            df3 =df3.drop(['Reason for Absence'], axis=1)
            columns_arranged = ['Reason_1','Reason_2','Reason_3',
                                'Reason_4','Day of the Week','Month Value', 'Transportation Expense', 'Age', 'Body Mass Index',
                                'Education', 'Children', 'Pets']
            
            df3=df3[columns_arranged]
            # replace the NaN values
            df3 = df3.fillna(value=0)
            
            
            
            
            
            
            
            
            self.preprocessed_data = df3.copy()
            
            self.data = df2
            
            
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            
            if (self.data is not None):  
                
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
        # a function which outputs 0 or 1 based on our model
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs
            
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
                return self.preprocessed_data
                


# In[ ]:





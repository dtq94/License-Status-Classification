import numpy as np
import os
from ML_pipeline import preprocessing, model_selection, utils 


df = utils.read_data("License_data.csv")

# change the columns name
new_col_name = preprocessing.replace_char(df," ","_")
df.columns = new_col_name

# Drop columns which are not relevent for the prediction / too many missing values
drop_col_list = ["id","license_id","ssa","location","application_created_date","account_number","address"]
df=preprocessing.drop_col(df, drop_col_list)


# convert string object into date
df.application_requirements_complete = preprocessing.convert_to_date(df.application_requirements_complete)
df.payment_date = preprocessing.convert_to_date(df.payment_date)
df.license_term_start_date = preprocessing.convert_to_date(df.license_term_start_date)
df.license_term_expiration_date = preprocessing.convert_to_date(df.license_term_expiration_date)
df.license_approved_for_issuance = preprocessing.convert_to_date(df.license_approved_for_issuance)
df.date_issued = preprocessing.convert_to_date(df.date_issued)

# Find no.of days btw different application status date
df["completion_to_start"] = preprocessing.date_diff(df.license_term_start_date, df.application_requirements_complete)
df["start_to_expiry"] =  preprocessing.date_diff(df.license_term_expiration_date , df.license_term_start_date)
df["approval_to_issuance"] = preprocessing.date_diff(df.date_issued , df.license_approved_for_issuance)
df["completion_to_payment"] = preprocessing.date_diff(df.payment_date , df.application_requirements_complete)

df["presence_of_enquiry_details"] = np.where(df.ward.isnull() | df.ward_precinct.isnull() | df.police_district | df.precinct , 0 ,1 )

df["target"] = preprocessing.convert_numeric(df[['license_status']])

df = preprocessing.target_encoding(df,col_to_transform=["license_description","state","city"])


# Resample the data to make it balanced
target_list = np.sort(df.target.unique()).tolist()
target_prop = [0.3,0.3,200,200,2]
sampled_df = preprocessing.random_sampling(df,target_list,target_prop)

X = sampled_df[['latitude', 'longitude',
       'completion_to_start',
       'start_to_expiry', 'approval_to_issuance', 'completion_to_payment',
       'presence_of_enquiry_details',  'license_description_target_1',
       'state_target_1', 'city_target_1', 'license_description_target_2',
       'state_target_2', 'city_target_2', 'license_description_target_3',
       'state_target_3', 'city_target_3', 'license_description_target_4',
       'state_target_4', 'city_target_4', 'license_description_target_5',
       'state_target_5', 'city_target_5']]

y = sampled_df['target']
X=preprocessing.impute(X,value=X.mean())

X_train, X_test, y_train, y_test = utils.data_split(X,y, size=0.1)

report=model_selection.run_models(X_train,y_train,X_test,y_test)






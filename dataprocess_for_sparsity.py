'''
author:Wentao Hu
'''
import numpy as np
import os
from sklearn.model_selection import KFold
from dataprocess_for_default import *
import pandas as pd
np.random.seed(0)


def main():
    dataset="yelp"
    ratingList = load_rating_file_as_list(f"Data/{dataset}/{dataset}-20filter.dat")    

    #generate different sparsity datasets
    for fraction in [0.2,0.4,0.6,0.8,1]:
        df=pd.DataFrame(ratingList)
        df=df.rename(columns={0:"user",1:"item",2:"rating"})
        dir=f"Data/{dataset}-{fraction}"
        if not os.path.exists(dir):
            os.makedirs(dir)

        df=df.groupby(df["user"]).apply(lambda x:x.sample(frac=fraction))  #sampling by user

        kf=KFold(n_splits=5,shuffle=True,random_state=1)
        file_index=1
        for train_index,test_index in kf.split(df):
            train,test=df.iloc[train_index],df.iloc[test_index]
            train.to_csv(f"{dir}/u{file_index}.base",sep='\t', index=False, header=False)
            test.to_csv(f"{dir}/u{file_index}.test",sep='\t', index=False, header=False)
            file_index+=1

        #after sampling with fraction, using leave-1-out strategy to generate train and test dataset
        df=df.rename(columns={"user":"user2"}) 
        df=df.reset_index() 
        df=df[["user2","item","rating"]]
        test=df.groupby("user2").sample(n=1,random_state=1)
        train=df.drop(test.index)
        test.to_csv(f"{dir}/u.test",sep='\t', index=False, header=False)
        train.to_csv(f"{dir}/u.base",sep='\t', index=False, header=False)


if __name__=="__main__":
    main()
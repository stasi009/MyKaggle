
# coding: utf-8

# In[103]:

get_ipython().magic(u'pylab inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Index
# * [fillna in Age attribute](#fillna_age)
# * [transform Sex into numeric type](#transform_sex)
# * [encode Embarked attribute](#encode_embark)
# * [Remove useless features](#remove_useless)

# In[104]:

titanic = pd.read_csv("train.csv")
titanic.describe()


# <a id="fillna_age"></a>
# ### fillna in Age attribute
# notice that in "Age" column, there are only 714 items, fewer than other attributes (which is 891). it indicates that there are a lot missing values in "Age"

# In[105]:

print "there are %d NaN in Age attribute"%(np.count_nonzero( np.isnan(titanic.Age) ))


# In[106]:

print "mean of Age is: %3.2f"%(titanic.Age.mean())
print "median of Age is: %3.2f"%(titanic.Age.median())


# In[107]:

# decide to use median to fill the missing values
titanic.Age = titanic.Age.fillna(titanic.Age.median())
titanic.Age.describe()


# <a id="transform_sex"></a>
# ### transform Sex attributes to numeric type

# In[108]:

sexgrps = titanic.Sex.groupby(titanic.Sex).size()
sexgrps / (sexgrps.sum())


# In[109]:

titanic["IsMale"] = (titanic.Sex == "male").astype(int)
titanic.IsMale.mean()


# In[110]:

del titanic["Sex"]# remove redundant attributes


# <a id="encode_embark"></a>
# ### Encode the Embarked attribute

# In[111]:

titanic.Embarked.unique()


# In[112]:

embark_counts = titanic.Embarked.groupby(titanic.Embarked).size()
embark_counts


# In[113]:

print "maximum embark position is '%s' with count=%d"%(embark_counts.argmax(),embark_counts.max())


# In[114]:

titanic.Embarked = titanic.Embarked.fillna(embark_counts.argmax())
titanic.Embarked.groupby(titanic.Embarked).size()


# In[115]:

def encode_category(category_values,nameprefix):
    unique_values = np.unique(category_values)
    
    datas = []
    names = []
    for onevalue in unique_values:
        datas.append( (category_values == onevalue).astype(int) )
        names.append(nameprefix+onevalue)
        
    return pd.concat(datas,axis=1,keys=names)

embarks = encode_category(titanic.Embarked,"Embark")
embarks.sum()


# In[116]:

# concatenate the Embarks DataFrame back to the original feature DataFrame
titanic = pd.concat([ titanic,embarks],axis=1)
del titanic["Embarked"]# remove redundant attribute
titanic.columns


# <a id="remove_useless"></a>
# ### Remove useless features

# In[117]:

nancount = np.count_nonzero( pd.isnull(  titanic.Cabin ) )
print "Cabin attribute has %d NaN, which is %3.2f%%"%(nancount,nancount*100.0/titanic.shape[0])


# In[118]:

# there are more than 50% missing values in Cabin attribute, so delete that column
del titanic["Cabin"]
titanic.columns


# In[119]:

del titanic["PassengerId"]# useless for classification
del titanic["Ticket"]# cannot understand this feature, think it is useless
titanic.columns


# In[120]:

titanic.columns


# In[ ]:




# In[ ]:




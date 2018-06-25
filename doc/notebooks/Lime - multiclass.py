
# coding: utf-8

# In[1]:


from __future__ import print_function
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
import sys


# ## Fetching data, training a classifier

# In the [previous tutorial](http://marcotcr.github.io/lime-ml/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.html), we looked at lime in the two class case. In this tutorial, we will use the [20 newsgroups dataset](http://scikit-learn.org/stable/datasets/#the-20-newsgroups-text-dataset) again, but this time using all of the classes.

# In[2]:


from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
# making class names shorter
class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:]) for x in newsgroups_train.target_names]
class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'


# In[3]:


print(','.join(class_names))

# Again, let's use the tfidf vectorizer, commonly used for text.

# In[4]:


vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)


# This time we will use Multinomial Naive Bayes for classification, so that we can make reference to [this document](http://scikit-learn.org/stable/datasets/#filtering-text-for-more-realistic-training).

# In[5]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=.01)
nb.fit(train_vectors, newsgroups_train.target)


# In[6]:


pred = nb.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='weighted')


# We see that this classifier achieves a very high F score. [The sklearn guide to 20 newsgroups](http://scikit-learn.org/stable/datasets/#filtering-text-for-more-realistic-training) indicates that Multinomial Naive Bayes overfits this dataset by learning irrelevant stuff, such as headers, by looking at the features with highest coefficients for the model in general. We now use lime to explain individual predictions instead.

# ## Explaining predictions using lime

# In[7]:


from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, nb)


# In[8]:


print(c.predict_proba([newsgroups_test.data[0]]).round(3))


# In[9]:


from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)


# Previously, we used the default parameter for label when generating explanation, which works well in the binary case.  
# For the multiclass case, we have to determine for which labels we will get explanations, via the 'labels' parameter.  
# Below, we generate explanations for labels 0 and 17.

# In[10]:


idx = 1340
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6, labels=[0, 17])
print('Document id: %d' % idx)
print('Predicted class =', class_names[nb.predict(test_vectors[idx]).reshape(1,-1)[0,0]])
print('True class: %s' % class_names[newsgroups_test.target[idx]])


# Now, we can see the explanations for different labels. Notice that the positive and negative signs are with respect to a particular label - so that words that are negative towards class 0 may be positive towards class 15, and vice versa.

# In[27]:


print ('Explanation for class %s' % class_names[0])
print ('\n'.join(map(str, exp.as_list(label=0))))
print ()
print ('Explanation for class %s' % class_names[17])
print ('\n'.join(map(str, exp.as_list(label=17))))


# Another alternative is to ask LIME to generate labels for the top K classes. This is shown below with K=2.  
# To see which labels have explanations, use the available_labels function.

# In[28]:


exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6, top_labels=2)
print(exp.available_labels())


# Now let's see some the visualization of the explanations. Notice that for each class, the words on the right side on the line are positive, and the words on the left side are negative. Thus, 'Caused' is positive for atheism, but negative for christian.

# In[29]:


exp.show_in_notebook(text=False)


# We notice that the classifier is using reasonable words (such as 'Genocide', 'Luther', 'Semitic', etc), as well as unreasonable ones ('Rice', 'owlnet'). Let's zoom in and just look at the explanations for class 'Atheism'.

# In[30]:


exp.show_in_notebook(text=newsgroups_test.data[idx], labels=(0,))


# Looking at this example demonstrates that there can be useful signal in the header or quotes that would generalize - i.e., the Subject line. There is also signal that would not generalize (e.g. email addresses and institution names).

# ## Explaining predictions without headers, quotes and footers

# Finally, we follow the [suggestion of removing headers, footers and quotes](http://scikit-learn.org/stable/datasets/#filtering-text-for-more-realistic-training), and explain the same example with the new data.

# In[31]:


newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)
nb = MultinomialNB(alpha=.01)
nb.fit(train_vectors, newsgroups_train.target)
c = make_pipeline(vectorizer, nb)
explainer = LimeTextExplainer(class_names=class_names)


# In[32]:


exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6, top_labels=2)
print(exp.available_labels())


# Notice how different the explanations are for the classifier without headers, footers and quotes. The prediction changes, but so do the reasons.

# In[33]:


exp.show_in_notebook(text=False)


# Let's see the explanation with the text for the top class (christian):

# In[34]:


exp.show_in_notebook(text=newsgroups_test.data[idx], labels=(15,))


# Notice how short the text became after removing all of that information. One begins to wonder if this version of the dataset is still useful, or if it is better to find another dataset altogether. Could a reasonable classifier detect that this document belongs to the class atheism?
# 
# Anyway, I hope this illustrated how to use LIME to explain arbitrary classifiers in the multiclass case!

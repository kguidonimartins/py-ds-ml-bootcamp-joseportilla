# 19-recommender-systems_01-recommender_systems_with_python

NOTE: Jannach et al. 2010 - Recommender Systems

NOTE: https://developers.google.com/machine-learning/recommendation

NOTE: https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada

NOTE: https://realpython.com/build-recommendation-engine-collaborative-filtering/

NOTE: https://towardsdatascience.com/how-to-build-a-recommendation-engine-quick-and-simple-aec8c71a823e

NOTE: Grus 2015 - Data Science from Scratch

NOTE: Gorakala and Usuelli 2015 - Building a recomendation system with R

NOTE: https://www.udemy.com/course/building-recommender-systems-with-machine-learning-and-ai/

NOTE: https://b-ok.org/s/?q=recommendation+system

NOTE: https://towardsdatascience.com/recommendation-system-series-part-1-an-executive-guide-to-building-recommendation-system-608f83e2630a

NOTE: https://medium.com/data-hackers/deep-learning-para-sistemas-de-recomenda%C3%A7%C3%A3o-parte-3-recomenda%C3%A7%C3%A3o-por-similaridade-d788c126d808

NOTE: https://medium.com/datadriveninvestor/how-to-built-a-recommender-system-rs-616c988d64b2

The two most common types of recommender systems are:

<!--
TODO: Conferir meu chute sobre a natureza dos dados usados em sistemas de recomendação
BODY: Content based se baseia numa matriz de atributos dos itens e o collaborative filtering se baseia numa matriz de itens adquiridos por indivíduos.
 -->

- Content based: focus on the attributes of the items and give you recommendations based on the similarity between them.

- Collaborative filtering: produces recommendations based on the knowledge of users' attitude to items, that is it uses the "wisdom of the crowd" to recommend items. Based on other people's shopping experiences, Amazon will suggest items that it believes you will enjoy. Most commonnly used! It has the ability to do feature learning on its own. Divided into:
    - Memory-based Collaborative Filtering: cosine similarity
    - Model-based Collaborative Filtering: single value decomposition

# Load libraries

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
sns.set_style('white')

```

# Load data

```python

columns_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('u.data', sep = '\t', names = columns_names)

df.head()

movie_titles = pd.read_csv('Movie_Id_Titles')

movie_titles.head()

df = pd.merge(df, movie_titles, on='item_id')

df.head()

```

# Explore data

```python

df.groupby('title')['rating'].mean().sort_values(ascending=False)

df.groupby('title')['rating'].count().sort_values(ascending=False)

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()

ratings['num_of_ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()

```

## Plotting

```python

ratings['num_of_ratings'].hist(bins=100)
plt.show()

ratings['rating'].hist(bins=100)
plt.show()

sns.jointplot(x='rating', y='num_of_ratings', data=ratings, alpha=0.5)
plt.show()

```

# Recommender system

```python

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

moviemat.shape

moviemat.head()

ratings.sort_values('num_of_ratings', ascending=False).head(10)

```

## Exploring Star Wars

```python
starwars_user_ratings = moviemat['Star Wars (1977)']
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()

corr_starwars.sort_values('Correlation', ascending=False).head()

corr_starwars = corr_starwars.join(ratings['num_of_ratings'])
corr_starwars.head()

corr_starwars[corr_starwars['num_of_ratings'] > 100].sort_values('Correlation', ascending=False)

```

## Explore Liar Liar

```python

liar_user_ratings = moviemat['Liar Liar (1997)']
similar_to_liarliar = moviemat.corrwith(liar_user_ratings)

corr_liarliar = (
    pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
      .dropna()
      .join(ratings['num_of_ratings'])
      .query('num_of_ratings > 100')
    )

corr_liarliar.sort_values

```

Basicamente, é uma correlação de um filme rankeado pelo usuário.

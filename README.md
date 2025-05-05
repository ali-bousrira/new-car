Projet New Car : Prédiction du prix de vente d'une voiture

Contexte

L'achat d'une voiture est une décision importante, et comme tout bon analyste de données, nous avons souhaité prendre cette décision sur la base d'analyses fiables. Ce projet part d'une situation simple : étant passionné par les données et ayant économisé pour acheter une voiture, l'idée est de déterminer de manière précise et objective le prix d'un véhicule donné, grâce à un modèle de régression linéaire.

Les données proviennent du site CarDekho et contiennent des informations sur plusieurs caractéristiques de voitures telles que l'année, le kilométrage, le type de carburant, la transmission, etc.

Analyse de données

Une première phase exploratoire a permis de prendre connaissance du jeu de données. Nous avons utilisé la bibliothèque Pandas pour charger les données, puis généré des statistiques descriptives (moyenne, quartiles, valeurs extrêmes). Des histogrammes ont été tracés pour visualiser la distribution des variables numériques telles que le prix de vente, l'année ou les kilomètres parcourus.

Des graphiques catégoriques (catplot de Seaborn) ont permis de comparer le prix moyen selon le type de carburant ou la transmission. Enfin, une analyse de corrélation a mis en évidence une relation négative forte entre l'âge du véhicule et son prix, confirmant la pertinence d'un modèle linéaire.

Algorithme utilisé

Nous avons mis en place deux modèles de régression linéaire :

Une régression linéaire univariée, en prenant uniquement l'âge du véhicule comme variable explicative.

Une régression linéaire multiple, intégrant l'âge, le kilométrage (Kms_Driven) et la transmission comme variables d'entrée.

Les deux modèles ont été évalués grâce à la MSE (erreur quadratique moyenne) et au coefficient de détermination R².

Conclusion

Les résultats obtenus montrent que l'âge du véhicule est le facteur principal influant sur le prix de vente. Le modèle multiple offre de meilleures performances prédictives en tenant compte de plusieurs critères pertinents. Ce projet permet ainsi de mieux comprendre les composantes du prix d'un véhicule et d'estimer de manière fiable la valeur marché d'une voiture.

En réponse à une requête concrète (le cas de "Martin"), nous avons utilisé le modèle pour estimer le prix moyen d'une voiture correspondant à ses critères (moins de 7 ans, moins de 100 000 km, transmission manuelle).

Ce projet illustre l'application concrète de la science des données dans un contexte du quotidien, tout en renforçant la maîtrise de l'analyse exploratoire, de la visualisation et de la modélisation prédictive.


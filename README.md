# Decision Support for Pricing of Second-Hand items

<br/>

## Source de données

<br/>

[*Vestiaire Collective*](https://fr.vestiairecollective.com/) est une plateforme de vente d'articles de mode de seconde main. Son objectif est de mettre en relation des vendeurs et des acheteurs tout en certifiant l'authenticité des produits proposés par les vendeurs. Cette plateforme permet ainsi de lever le problème d'asymmétrie d'information lié à la vente de produits de luxe.

<br/>

## Objectif 

Ce projet a pour objectif d'aider les vendeurs à fixer le bon prix lorsqu'ils souhaitent mettre en vente un article sur Vestiaire Collective.

Etapes de la mise en vente
1. Le vendeur renseigne les caractéristiques du produit.
2. Le vendeur fixe un prix pour son produit.
3. L'algorithme de *machine learning* prédit le prix en fonction des caractéristiques du produit. Cette prédiction sert de *benchmark*.
4. Le prix *benchmark* est renvoyé à l'utilisateur pour l'aider à choisir le bon prix. 
5. Extension : on renvoie aussi des annonces similaires au produit que le vendeur souhaite mettre en vente en guise de comparaison.

<br/>

## Intérêts du projet

<br/>

### Pour le vendeur 

La prédiction de l'algorithme lui permet de ne pas perdre de temps dans la fixation du prix. Cette aide à la décision maximise ses gains :
- en lui évitant de vendre à un prix trop bas, 
- en lui évitant de ne pas trouver d'acheteurs car son prix est trop élevé.

<br/>

### Pour les potentiels acheteurs

La prédiction de l'algorithme sur le prix du marché est aussi une approximation de la *disposition à payer des acheteurs pour un type d'article donné. En aidant les vendeurs à fixer le "bon" prix, l'algorithme permet aux acheteurs de trouver des articles vendus à un prix proche de leur disposition à payer.

<br/>

### Pour Vestiaire Collective 

**Maximisation des profits** : l'aide dans la fixation des prix assure le volume et la rapidité des transactions. Plus il y a de transactions sur la plateforme, plus Vestiaire Collective reçoit de commissions.

**Satisfaction des utilisateurs** : les acheteurs trouvent des articles dont le prix leur permet de générer un surplus et les vendeurs gagnent du temps dans la vente de leurs articles.

<br/>

## Données 

<br/>

| Variable   |      Type      | Description    |
|:---------- |:------------- |:------------- |
| `id`         |  int        | identifiant de l'article               | 
| `url`        |   string    | lien de l'annonce               | 
| `num_likes`   | int  | nombre de *likes* reçu par l'article               | 
| `price` | float | prix de l'article |
| `we_love_tag` | bool | indique si l'article est un coup de coeur de Vestiaire Collective |
| `online_date` | datetime | date de mise en ligne | 
| `gender` | bool | genre du produit | 
| `category` | category | categorie du produit (chaussures, sacs, vêtements, etc.) | 
|  `sub_category` | category | sous-catégorie (bottes, jeans, chemise, etc.) | 
| `designer` | category | marque de l'article | 
| `condition` | category | état de l'article |
| `material` | category | matière de l'article | 
| `color` | category | couleur de l'article | 
| `size` | category | taille de l'article (M, L, 40, etc.) | 
| `location` | category ou string | localisation du vendeur | 

<br/>

## Stratégie de prédiction 

<br/> 

### Prédicition de `we_love_tag`

Lors de la mise en ligne d'un nouvel article, on n'a pas d'information sur la variable `we_love_tag`. Un premier modèle de classification consiterait à prédire si l'article sera un coup de coeur de Vestiaire Collective.

<br/>

### Prédiction du prix

Variables explicatives :

- Caractéristiques du produit à l'exception de `id`, `url`, `online_date` et `location` (?) 
- Prédiction de `we_love_tag` 

Cible : `price` 

Optimisation de plusieurs modèles de régression pour déterminer la relation la plus adaptée entre la cible et les variables explicatives

Choix du meilleur modèle, entrainement sur l'ensemble des données et sauvegarde

<br/>

## Création d'une application pour l'utilisateur 


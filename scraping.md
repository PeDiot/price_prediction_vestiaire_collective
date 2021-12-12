# Organisation Projet Machine Learning

<br/>

## Etape 1 : *Scraping* 

<br/>

### Cible

[*Vestiaire Collective*](https://fr.vestiairecollective.com/)

<br/>

### Collecte des marques

- Marques à récolter dans "Designers" puis "Notre sélection" 
- Pour chaque marque : 
    - Cliquer sur le nom de la marque
    - Récolter les $n$ premières pages triées par pertinence (~ 48 $n$ produits)
    - Stocker chaque page dans une dataclass `Marque` avec pour argument "nom_marque", "numero_page" et "code_page"
    - Sauvegarder les $n$ pages au format `html` ou `json` dans un dossier spécifique pour chaque marque

**Remarque** : l'enjeu va être de déterminer $n$

<br/>

### Collecte des articles

Pour chaque marque (dossier) : 
- Pour chaque page (fichier de *backup*) : 
    - Identifier les articles présents dans le code de la page
    - Pour chaque article : 
        - Récupérer le lien et lancer une navigation 
        - Récupérer les infos liés au produit
        - Stocker les infos du produit dans une dataclass `Produit` avec pour arguments : "prix", "nombre_likes", "mise_en_ligne", "genre", etc.
    - Stocker l'ensemble des infos pour tous les produits de la page dans une dataclass `ProduitsMarque` héritant de `Marque` avec un argument supplémentaires représentant une liste d'objets de type `Produit`
    - Sauvegarder au format `json` dans le dossier lié à la marque

<br/>

### Packages 

- `Selenium` pour la navigation : 
    - Se rendre sur une page,
    - Changements de page, 
    - Accepter les cookies.

- `beautifulSoup` pour la manipulation du code html



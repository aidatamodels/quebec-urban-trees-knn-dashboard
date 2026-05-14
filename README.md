# Québec Urban Trees K-NN Dashboard

## Présentation du projet

Québec Urban Trees K-NN Dashboard est un projet de science des données appliqué aux arbres répertoriés de la Ville de Québec.

Le projet utilise l’API CKAN de Données Québec pour charger les données ouvertes, prépare les variables avec Python, applique un modèle K-NN et génère un fichier `dashboard_data.json` utilisé par un dashboard web interactif.

L’objectif est de créer une application pédagogique et visuelle permettant de comprendre le fonctionnement de K-NN à partir d’un exemple concret : la recherche d’arbres similaires et la classification du type d’arbre.

## Objectif

Le projet répond à deux questions principales :

> Quels arbres sont les plus similaires à un arbre sélectionné ?

> Peut-on classifier le type d’un arbre, feuillu ou conifère, à partir de ses caractéristiques spatiales et descriptives ?

Le dashboard permet notamment de :

- explorer les arbres répertoriés sur une carte interactive ;
- filtrer les arbres par type, lieu, propriété, espèce et diamètre ;
- sélectionner un arbre de référence ;
- afficher les K arbres les plus similaires ;
- visualiser la répartition des espèces ;
- consulter les résultats d’un modèle K-NN de classification.

## Source des données

Les données proviennent du portail Données Québec.

Jeu de données utilisé : `Arbres répertoriés`  
Organisation : Ville de Québec  
Ressource CKAN : `13a51853-a5b5-4add-8791-02ccba5c1be7`  
Licence des données : Attribution Creative Commons 4.0 International (CC BY 4.0)

## Variables principales

| Variable | Description |
|---|---|
| `ID` | Identifiant unique de l’arbre |
| `TYPE_LIEU` | Type de lieu où est planté l’arbre |
| `NOM_LATIN` | Nom latin de l’arbre |
| `NOM_FRANCAIS` | Nom français de l’arbre |
| `TYPE_ARBRE` | Type d’arbre : feuillu ou conifère |
| `DIAMETRE` | Diamètre du tronc en centimètres |
| `POSITION_MESURE` | Position de la mesure du diamètre |
| `MULTI_TRONC` | Indique si l’arbre possède plusieurs troncs |
| `DATE_PLANTE` | Date de plantation lorsqu’elle est connue |
| `TYPE_PROP` | Type de propriété : terrain public ou privé |
| `LATITUDE` | Latitude de la position de l’arbre |
| `LONGITUDE` | Longitude de la position de l’arbre |

La colonne `NOM_TOPOGRAPHIE`, qui contient une adresse complète, n’est pas exposée directement dans le dashboard public.

## Approche méthodologique

### 1. Chargement des données par API

Le script `build_dashboard_data.py` interroge l’API CKAN de Données Québec avec pagination.

### 2. Nettoyage des données

Le script nettoie les coordonnées, le diamètre, les dates de plantation, les catégories et les valeurs manquantes.

### 3. Modélisation K-NN

Le projet utilise deux approches :

- `NearestNeighbors` pour trouver les arbres les plus similaires ;
- `KNeighborsClassifier` pour classifier `TYPE_ARBRE`, soit `Feuillu` ou `Conifère`.

### 4. Génération du JSON

Le script Python produit `dashboard_data.json`, qui alimente le dashboard web.

### 5. Dashboard web

Le fichier `index.html` charge `dashboard_data.json` et affiche une carte Leaflet avec filtres, voisins similaires, statistiques et résultats du modèle.

## Technologies utilisées

- Python ;
- Pandas ;
- NumPy ;
- Requests ;
- Scikit-learn ;
- HTML5 ;
- CSS3 ;
- JavaScript ;
- Leaflet.js ;
- GitHub Pages.

## Structure du projet

```text
quebec-urban-trees-knn-dashboard/
│
├── build_dashboard_data.py
├── dashboard_data.json
├── index.html
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

Le fichier `dashboard_data.json` inclus sert de démonstration. Pour générer le JSON complet à partir de l’API, exécuter le script Python.

## Installation locale

```bash
pip install -r requirements.txt
python build_dashboard_data.py
python -m http.server 8000
```

Sur Windows :

```bash
py -m pip install -r requirements.txt
py build_dashboard_data.py
py -m http.server 8000
```

Ouvrir ensuite :

```text
http://localhost:8000
```

## Publication avec GitHub Pages

Après avoir généré `dashboard_data.json`, publier les fichiers du projet dans GitHub et activer GitHub Pages :

```text
Settings → Pages → Deploy from a branch → main → /root
```

## Limites du projet

Ce projet est conçu à des fins pédagogiques et démonstratives. Les résultats ne constituent pas une analyse officielle de l’inventaire arboricole.

## Licence

Le code source du projet est distribué sous licence MIT.

Les données utilisées restent soumises aux conditions de leur source d’origine, notamment la licence Attribution Creative Commons 4.0 International (CC BY 4.0).

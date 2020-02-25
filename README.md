# P-androide

Ce repositoire contient tous les fichiers pour le projet ANDROIDE - Diagnostic and Value Of Information, dans le cadre du master M1 ANDROIDE à Sorbonne Uninversité pour l'année 2019/2020. 

Enseigneurs résponsables :
  Pierre-Henri Wuillemin, 
  Paolo Viappiani

Étudiants :
  Ariana Carnielli, 
  Ivan Kachaikin
  
Derniers modifications :

Correction du premier algorithme pour le TSP. Le calcul de l'esperance dans cet algorithme a été corrigé, le code utilise des appels à chgEvidence() maintenant et est commenté, la fonction a une mode debug aussi. 

Des fonctions auxiliares pour le reset du BN ont été créées (start_bay_lp() and reset_bay_lp(dict_inf)). Une première fonction pour le calcul d'une séquence en utilisant des obsérvations à été codé d'après les articles de Heckerman mais n'est pas complete, elle ne retourne pas encore l'esperance de coût de la séquence calculé (ce qui est necéssaire pour la séquence du travail). 


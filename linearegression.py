import numpy as np

class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, l2_penalty=0):
        """
        Initialise le modèle de régression linéaire.
        
        learning_rate: pas d'apprentissage pour la descente de gradient
        n_iterations: nombre d'itérations pour l'algorithme de descente de gradient
        l2_penalty: coefficient de régularisation L2 (Ridge)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def _add_intercept(self, X):
        """Ajoute une colonne de 1 à X pour gérer le terme d'intercept"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
        
    def fit(self, X, target):
        """
        Entraîne le modèle de régression linéaire en utilisant la descente de gradient.
    
        X: matrice des caractéristiques (features)
        y: vecteur cible (target)
        """

        # Initialisation
        m, n = X.shape  # m = nombre d'exemples, n = nombre de caractéristiques
        self.weights = np.zeros(n)
        self.bias = 0
        
        # Entraînement par descente de gradient
        for i in range(self.n_iterations):
            # Calcul des prédictions
            target_pred = self.predict(X)
            
            # Calcul des gradients
            dw = (1/m) * np.dot(X.T, (target_pred - target))
            db = (1/m) * np.sum(target_pred - target)
            
            # Ajout de la régularisation L2 (Ridge)
            if self.l2_penalty > 0:
                dw += (self.l2_penalty / m) * self.weights
            
            # Mise à jour des paramètres
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calcul du coût pour le suivi
            cost = self._compute_cost(X, target, target_pred)
            self.cost_history.append(cost)
            
    def _compute_cost(self, X, y, y_pred):
        """
        Calcule la fonction de coût.
        
        X: matrice des caractéristiques
        y: vecteur cible
        y_pred: vecteur des prédictions
        """
        m = X.shape[0]
        # Erreur quadratique moyenne
        cost = (1/(2*m)) * np.sum((y_pred - y) ** 2)
        
        # Ajouter le terme de régularisation L2
        if self.l2_penalty > 0:
            cost += (self.l2_penalty/(2*m)) * np.sum(self.weights**2)
            
        return cost
            
    def predict(self, caracteristiques):
        return np.dot(caracteristiques, self.weights) + self.bias
    
    def score(self, X, y):
        """
        Calcule le coefficient de détermination.
        
        X: matrice des caractéristiques
        y: vecteur cible
        
        Retourne:
        R²: score entre 0 et 1
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        
        return 1 - (ss_residual / ss_total)
    
    def mse(self, X, y):
        """
        Calcule l'erreur quadratique moyenne.

        X: matrice des caractéristiques
        y: vecteur cible
        
        Retourne:
        MSE: erreur quadratique moyenne
        """
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)
    
    def mae(self, X, y):
        """
        Calcule l'erreur absolue moyenne.

        X: matrice des caractéristiques
        y: vecteur cible
        
        Retourne:
        MAE: erreur absolue moyenne
        """
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))
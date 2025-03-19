import random
import numpy as np
from blackjack import play_blackjack_round, play_with_display, get_hand_value, get_card_value

class GeneticBlackjackAI:
    def __init__(self, population_size=100, mutation_rate=0.01):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.best_strategy = None
        
        # Initialiser la population
        for _ in range(population_size):
            # Stratégie: tableau 2D [valeur_main][valeur_visible_croupier] = action (0=rester, 1=tirer)
            # Valeur main: 4-21 (18 valeurs possibles)
            # Carte visible croupier: 2-11 (10 valeurs possibles)
            strategy = np.random.randint(0, 2, size=(18, 10))
            
            # Stratégies forcées: toujours tirer si < 11, toujours rester si 21
            strategy[0:7, :] = 1  # Toujours tirer si main <= 11
            strategy[17, :] = 0   # Toujours rester si main = 21
            
            self.population.append(strategy)
    
    def decision_function(self, player_hand, dealer_card):
        """Utilise la meilleure stratégie pour prendre une décision."""
        if self.best_strategy is None:
            return "tirer" if random.random() < 0.5 else "rester"
        
        hand_value = get_hand_value(player_hand)
        dealer_value = get_card_value(dealer_card)
        
        # Ajuster les indices pour correspondre au tableau
        hand_index = min(max(0, hand_value - 4), 17)  # 4->0, 21->17
        dealer_index = dealer_value - 2  # 2->0, 11->9
        
        action = self.best_strategy[hand_index][dealer_index]
        return "tirer" if action == 1 else "rester"
    
    def evaluate_fitness(self, strategy, games=100):
        """Évalue la performance d'une stratégie en jouant plusieurs parties."""
        wins = 0
        draws = 0
        
        def strategy_decision_function(player_hand, dealer_card):
            hand_value = get_hand_value(player_hand)
            dealer_value = get_card_value(dealer_card)
            
            # Stratégies forcées
            if hand_value >= 21:
                return "rester"
            
            # Ajuster les indices pour correspondre au tableau
            hand_index = min(max(0, hand_value - 4), 17)  # 4->0, 21->17
            dealer_index = dealer_value - 2  # 2->0, 11->9
            
            action = strategy[hand_index][dealer_index]
            return "tirer" if action == 1 else "rester"
        
        for _ in range(games):
            result = play_blackjack_round(strategy_decision_function)
            if result == "ia":
                wins += 1
            elif result == "égalité":
                draws += 0.5
        
        # Score de fitness: pourcentage de victoires + demi-points pour les égalités
        return (wins + draws) / games
    
    def select_parents(self, fitnesses):
        """Sélectionne des parents avec une probabilité proportionnelle à leur fitness."""
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            # Éviter la division par zéro
            selection_probs = [1/len(fitnesses)] * len(fitnesses)
        else:
            selection_probs = [f/total_fitness for f in fitnesses]
            
        parent_indices = np.random.choice(
            range(len(self.population)), 
            size=len(self.population), 
            p=selection_probs
        )
        return [self.population[i] for i in parent_indices]
    
    def crossover(self, parent1, parent2):
        """Combine deux stratégies parentales pour créer un enfant."""
        child = np.zeros_like(parent1)
        
        # Crossover à point unique pour chaque ligne du tableau de stratégie
        for row in range(len(parent1)):
            crossover_point = random.randint(0, len(parent1[0]))
            child[row, :crossover_point] = parent1[row, :crossover_point]
            child[row, crossover_point:] = parent2[row, crossover_point:]
            
        return child
    
    def mutate(self, strategy):
        """Applique des mutations aléatoires à la stratégie."""
        for i in range(len(strategy)):
            for j in range(len(strategy[0])):
                if random.random() < self.mutation_rate:
                    # On ne mute pas les stratégies forcées
                    if not ((i < 7 and j < 10) or (i == 17 and j < 10)):
                        strategy[i][j] = 1 - strategy[i][j]  # Inverser 0->1 ou 1->0
        
        # S'assurer que les stratégies forcées sont maintenues
        strategy[0:7, :] = 1  # Toujours tirer si main <= 11
        strategy[17, :] = 0   # Toujours rester si main = 21
        
        return strategy
    
    def evolve(self, generations=50, games_per_eval=100):
        """Fait évoluer la population sur plusieurs générations."""
        for gen in range(generations):
            # Évaluer la fitness de chaque individu
            fitnesses = [self.evaluate_fitness(strategy, games_per_eval) for strategy in self.population]
            
            # Trouver la meilleure stratégie
            best_idx = np.argmax(fitnesses)
            self.best_strategy = self.population[best_idx].copy()
            best_fitness = fitnesses[best_idx]
            
            print(f"Génération {gen+1}/{generations} - Meilleure fitness: {best_fitness:.4f}")
            
            # Sélectionner les parents pour la nouvelle génération
            parents = self.select_parents(fitnesses)
            
            # Créer une nouvelle population
            new_population = []
            for i in range(0, len(parents), 2):
                if i+1 < len(parents):
                    # Crossover
                    child1 = self.crossover(parents[i], parents[i+1])
                    child2 = self.crossover(parents[i+1], parents[i])
                    
                    # Mutation
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    new_population.extend([child1, child2])
            
            # Élitisme: garder la meilleure stratégie dans la nouvelle génération
            if len(new_population) > 0:
                new_population[0] = self.best_strategy.copy()
                
                # Compléter la population si nécessaire
                while len(new_population) < self.population_size:
                    new_population.append(random.choice(new_population))
                
                self.population = new_population[:self.population_size]
    
    def display_best_strategy(self):
        """Affiche la meilleure stratégie sous forme de tableau."""
        if self.best_strategy is None:
            print("Aucune stratégie n'a encore été trouvée.")
            return
            
        print("\nMeilleure stratégie (1=Tirer, 0=Rester):")
        print("  | 2  3  4  5  6  7  8  9  10 A")
        print("--+-------------------------------")
        
        for i in range(18):
            hand_value = i + 4
            row = self.best_strategy[i]
            print(f"{hand_value:2d}| {' '.join(str(int(x)) for x in row)}")


def main():
    print("IA génétique pour le Blackjack")
    print("------------------------------")
    
    # Créer et entraîner l'IA génétique
    population_size = int(input("Taille de la population (par défaut 100): ") or "100")
    generations = int(input("Nombre de générations (par défaut 50): ") or "50")
    games_per_eval = int(input("Parties par évaluation (par défaut 100): ") or "100")
    
    ai = GeneticBlackjackAI(population_size=population_size)
    ai.evolve(generations=generations, games_per_eval=games_per_eval)
    
    ai.display_best_strategy()
    
    # Jouer avec la meilleure stratégie trouvée
    print("\nVoulez-vous jouer avec cette stratégie? (o/n)")
    if input().lower() == 'o':
        scores = {"ia": 0, "croupier": 0, "égalité": 0}
        parties = 0
        
        while True:
            gagnant = play_with_display(ai.decision_function)
            scores[gagnant] += 1
            parties += 1
            
            print(f"\nScore après {parties} parties:")
            print(f"IA: {scores['ia']}, Croupier: {scores['croupier']}, Égalités: {scores['égalité']}")
            
            rejouer = input("\nJouer une autre partie? (o/n): ")
            if rejouer.lower() != 'o':
                break
        
        print("Merci d'avoir joué!")


if __name__ == "__main__":
    main()

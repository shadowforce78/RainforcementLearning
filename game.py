import pygame
import random
import numpy as np
import os

# Initialisation de Pygame
pygame.init()

# Paramètres du jeu
WIDTH, HEIGHT = 400, 600
PLAYER_SIZE = 50
OBSTACLE_SIZE = 50
FPS = 60
MAX_EPISODES = 500  # Augmenté pour plus d'entraînement
Q_TABLE_FILE = "q_table_enhanced.npy"  # Nouveau fichier pour la Q-table améliorée
TRAINING_SPEED = 10  # Multiplicateur de vitesse

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
TRANSPARENT = (0, 0, 0, 128)

# Création de la fenêtre
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dodge Game")


# Classe du joueur
class Player:
    def __init__(self):
        self.x = WIDTH // 2 - PLAYER_SIZE // 2
        self.y = HEIGHT - 100
        self.speed = 10  # augmenté pour que l'IA bouge plus

    def move(self, direction):
        if direction == 0 and self.x > 0:
            self.x -= self.speed
        elif direction == 1 and self.x < WIDTH - PLAYER_SIZE:
            self.x += self.speed

    def draw(self):
        pygame.draw.rect(screen, WHITE, (self.x, self.y, PLAYER_SIZE, PLAYER_SIZE))


# Classe des obstacles
class Obstacle:
    def __init__(self):
        self.x = random.randint(0, WIDTH - OBSTACLE_SIZE)
        self.y = -OBSTACLE_SIZE
        self.speed = random.randint(3, 7)  # Vitesse variable des obstacles

    def move(self):
        self.y += self.speed

    def draw(self):
        pygame.draw.rect(screen, RED, (self.x, self.y, OBSTACLE_SIZE, OBSTACLE_SIZE))


# IA simple avec Q-Learning
class AI:
    def __init__(self):
        # Q-table avec dimensions augmentées:
        # [position_x, obstacle_proche, vitesse_obstacle, action]
        self.q_table = np.zeros((WIDTH // PLAYER_SIZE, 2, 3, 2))
        print("Nouvelle Q-table améliorée créée")

        self.epsilon = 0.7  # Augmenté pour favoriser encore plus l'exploration au début
        self.learning_rate = 0.2
        self.discount_factor = 0.95
        self.last_actions = []

    def save_q_table(self):
        np.save(Q_TABLE_FILE, self.q_table)
        print("Q-table sauvegardée")

    def get_state(self, player, obstacles):
        # Par défaut, pas d'obstacle proche et vitesse moyenne
        obstacle_close = 0
        obstacle_speed = 1  # 0: lent, 1: moyen, 2: rapide
        
        # Chercher l'obstacle le plus menaçant
        closest_obstacle = None
        min_distance = float('inf')
        
        for obstacle in obstacles:
            # Est-ce que l'obstacle est dans la trajectoire du joueur?
            if obstacle.x < player.x + PLAYER_SIZE and obstacle.x + OBSTACLE_SIZE > player.x and obstacle.y > 0:
                # Calculer la distance verticale
                distance = player.y - (obstacle.y + OBSTACLE_SIZE)
                
                # Garder l'obstacle le plus proche
                if distance > 0 and distance < min_distance:
                    min_distance = distance
                    closest_obstacle = obstacle
        
        # Si on a trouvé un obstacle menaçant
        if closest_obstacle:
            if min_distance < 200:
                obstacle_close = 1
                
            # Catégoriser la vitesse
            if closest_obstacle.speed <= 4:
                obstacle_speed = 0  # Lent
            elif closest_obstacle.speed <= 6:
                obstacle_speed = 1  # Moyen
            else:
                obstacle_speed = 2  # Rapide
                
        return (player.x // PLAYER_SIZE, obstacle_close, obstacle_speed)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 1)  # Exploration
        else:
            return np.argmax(self.q_table[state])  # Exploitation

    def update_q_table(self, state, action, reward, next_state):
        best_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]

        # Formule Q-learning mise à jour
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
            reward + self.discount_factor * best_future_q
        )

        self.q_table[state][action] = new_q


# Fonction pour afficher la visualisation de l'IA
def draw_ai_visualization(screen, ai, player, obstacles, current_state, action):
    # Créer une surface transparente
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))  # Fond semi-transparent

    # Dessiner la grille d'états (divisée par PLAYER_SIZE)
    cell_width = PLAYER_SIZE
    num_cells = WIDTH // cell_width

    # Extraire la position x de l'état (qui est maintenant un tuple)
    current_cell_x = current_state[0] * cell_width
    pygame.draw.rect(overlay, (0, 255, 0, 100), (current_cell_x, 0, cell_width, HEIGHT))

    # Dessiner les valeurs de Q pour chaque état
    for x in range(num_cells):
        # Pour chaque état d'obstacle (0: pas d'obstacle proche, 1: obstacle proche)
        obstacle_state = current_state[1]  # Utiliser l'état d'obstacle actuel
        obstacle_speed = current_state[2]  # Utiliser l'état de vitesse actuel
        state = (x, obstacle_state, obstacle_speed)

        # Valeurs Q pour cet état (Gauche et Droite)
        q_left = ai.q_table[state][0]
        q_right = ai.q_table[state][1]

        # Normaliser pour l'affichage
        max_q = max(abs(q_left), abs(q_right), 1)  # Éviter division par zéro
        left_height = min(100, abs(q_left) / max_q * 100)
        right_height = min(100, abs(q_right) / max_q * 100)

        # Dessiner les barres de valeur Q
        left_color = RED if q_left < 0 else GREEN
        right_color = RED if q_right < 0 else GREEN

        # Position en bas de l'écran
        pygame.draw.rect(
            overlay,
            left_color,
            (x * cell_width + 5, HEIGHT - left_height - 110, 15, left_height),
        )
        pygame.draw.rect(
            overlay,
            right_color,
            (x * cell_width + 30, HEIGHT - right_height - 110, 15, right_height),
        )

        # Indiquer l'action optimale pour cet état
        best_action = np.argmax(ai.q_table[state])
        if best_action == 0:  # Gauche
            pygame.draw.polygon(
                overlay,
                YELLOW,
                [
                    (x * cell_width + 12, HEIGHT - 120),
                    (x * cell_width + 2, HEIGHT - 130),
                    (x * cell_width + 22, HEIGHT - 130),
                ],
            )
        else:  # Droite
            pygame.draw.polygon(
                overlay,
                YELLOW,
                [
                    (x * cell_width + 38, HEIGHT - 120),
                    (x * cell_width + 28, HEIGHT - 130),
                    (x * cell_width + 48, HEIGHT - 130),
                ],
            )

    # Dessiner la trajectoire prédite basée sur l'action actuelle
    predicted_x = player.x
    if action == 0:  # Gauche
        predicted_x = max(0, player.x - player.speed * 5)
    elif action == 1:  # Droite
        predicted_x = min(WIDTH - PLAYER_SIZE, player.x + player.speed * 5)
    pygame.draw.rect(
        overlay, CYAN, (predicted_x, player.y, PLAYER_SIZE, PLAYER_SIZE), 2
    )

    # Dessiner les zones de danger pour les obstacles
    for obstacle in obstacles:
        # Créer une zone de danger qui s'étend vers le bas
        danger_height = HEIGHT - obstacle.y
        pygame.draw.rect(
            overlay,
            (255, 0, 0, 30),
            (obstacle.x, obstacle.y, OBSTACLE_SIZE, danger_height),
        )

    # Dessiner une légende
    font = pygame.font.SysFont(None, 24)

    # Explication des éléments visuels
    overlay.blit(font.render("Vue de l'IA:", True, WHITE), (WIDTH - 150, 10))
    overlay.blit(font.render("État actuel", True, GREEN), (WIDTH - 150, 40))

    pygame.draw.rect(overlay, GREEN, (WIDTH - 160, 38, 8, 8))
    pygame.draw.rect(overlay, RED, (WIDTH - 160, 58, 8, 8))
    overlay.blit(font.render("Q-val. négative", True, WHITE), (WIDTH - 150, 55))

    pygame.draw.rect(overlay, GREEN, (WIDTH - 160, 78, 8, 8))
    overlay.blit(font.render("Q-val. positive", True, WHITE), (WIDTH - 150, 75))

    pygame.draw.polygon(
        overlay, YELLOW, [(WIDTH - 156, 98), (WIDTH - 160, 108), (WIDTH - 152, 108)]
    )
    overlay.blit(font.render("Action optimale", True, WHITE), (WIDTH - 150, 95))

    # Afficher sur l'écran
    screen.blit(overlay, (0, 0))


# Une partie du jeu (un épisode d'entraînement)
def game_episode(ai, episode_num, training_speed):
    clock = pygame.time.Clock()
    player = Player()
    obstacles = []
    running = True
    score = 0
    frame_count = 0
    show_ai_view = True  # Affichage de la visualisation de l'IA

    # Ajouter ces variables pour suivre les mouvements constants
    oscillation_count = 0
    last_action = None

    while running:
        frame_count += 1
        screen.fill(BLACK)

        reward = 1  # Récompense de départ par frame

        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return -1  # Signal pour quitter complètement
            # Permettre de modifier la vitesse d'entraînement pendant le jeu
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return {"action": "speed_up"}
                elif event.key == pygame.K_DOWN:
                    return {"action": "speed_down"}
                elif event.key == pygame.K_v:
                    show_ai_view = not show_ai_view  # Toggle l'affichage de l'IA

        # Obtenir l'état actuel
        state = ai.get_state(player, obstacles)
        prev_x = player.x  # Stocker la position avant mouvement
        # Choisir une action avec l'IA
        action = ai.choose_action(state)

        # Détection d'oscillation (gauche-droite-gauche...)
        if last_action is not None and action != last_action:
            oscillation_count += 1
        else:
            oscillation_count = max(0, oscillation_count - 1)  # Réduire progressivement
        last_action = action

        player.move(action)

        # Récompenses
        reward = 1  # Récompense de base pour survivre

        # Détection d'oscillation excessive et pénalité
        if oscillation_count > 5:
            reward -= 2  # Pénaliser l'oscillation excessive
            oscillation_count -= 1  # Réduire le compteur

        # Bonus réduit pour le mouvement - pour éviter qu'il ne bouge juste pour les points
        if player.x != prev_x:
            reward += 1  # Réduit de 3 à 1

        # Initialisation de la variable collision avant la boucle des obstacles
        collision = False

        # Génération des obstacles
        if random.randint(1, 60) == 1:
            obstacles.append(Obstacle())

        # Déplacement et affichage des obstacles
        for obstacle in obstacles[:]:
            obstacle.move()
            if obstacle.y > HEIGHT:
                obstacles.remove(obstacle)
                score += 1
            obstacle.draw()

            # Vérification de la collision
            if (
                obstacle.x < player.x + PLAYER_SIZE
                and obstacle.x + OBSTACLE_SIZE > player.x
                and obstacle.y + OBSTACLE_SIZE > player.y
                and obstacle.y < player.y + PLAYER_SIZE
            ):
                reward = -200  # Pénalité augmentée pour collision
                collision = True
                running = False
                break

        # Bonus pour éviter activement un obstacle
        if not collision:
            for obstacle in obstacles:
                if obstacle.y > 0 and obstacle.y < HEIGHT - 100:
                    # Distance horizontale entre le joueur et l'obstacle
                    horizontal_distance = abs(player.x + PLAYER_SIZE/2 - (obstacle.x + OBSTACLE_SIZE/2))
                    # Distance verticale
                    vertical_distance = abs(player.y - (obstacle.y + OBSTACLE_SIZE))
                    
                    # Évitement propre: obstacle proche mais pas de collision
                    if horizontal_distance < PLAYER_SIZE*1.5 and vertical_distance < 150:
                        # Plus l'obstacle est proche et rapide, plus la récompense est grande
                        proximity_factor = max(0, (150 - vertical_distance) / 50)  # 0-3 selon la proximité
                        speed_factor = obstacle.speed / 5  # 0.6-1.4 selon la vitesse
                        
                        # Récompense dynamique basée sur la difficulté d'évitement
                        evasion_reward = 10 * proximity_factor * speed_factor
                        reward += evasion_reward
                        
                    # Bonus supplémentaire pour s'éloigner d'un obstacle aligné
                    elif (obstacle.x < player.x + PLAYER_SIZE and obstacle.x + OBSTACLE_SIZE > player.x):
                        # L'obstacle est dans la même colonne que le joueur
                        if (action == 0 and obstacle.x > 0) or (action == 1 and obstacle.x < WIDTH - OBSTACLE_SIZE):
                            reward += 8  # Augmenté pour encourager les mouvements d'évitement

        # Obtenir le nouvel état et mettre à jour la Q-table
        next_state = ai.get_state(player, obstacles)
        ai.update_q_table(state, action, reward, next_state)

        # Affichage du joueur
        player.draw()

        # Afficher la visualisation de l'IA si activée
        if show_ai_view:
            draw_ai_visualization(screen, ai, player, obstacles, state, action)

        # Afficher les informations d'entraînement
        font = pygame.font.SysFont(None, 24)
        episode_text = font.render(
            f"Episode: {episode_num}/{MAX_EPISODES}", True, WHITE
        )
        score_text = font.render(f"Score: {score}", True, WHITE)
        epsilon_text = font.render(f"Epsilon: {ai.epsilon:.4f}", True, WHITE)
        speed_text = font.render(f"Vitesse: {training_speed}x (↑/↓)", True, GREEN)
        ai_view_text = font.render(
            f"Vue IA: {'ON' if show_ai_view else 'OFF'} (V)",
            True,
            PURPLE if show_ai_view else WHITE,
        )

        screen.blit(episode_text, (10, 10))
        screen.blit(score_text, (10, 40))
        screen.blit(epsilon_text, (10, 70))
        screen.blit(speed_text, (10, 100))
        screen.blit(ai_view_text, (10, 130))

        pygame.display.flip()
        # Ajuster la vitesse du jeu en fonction du multiplicateur de vitesse
        clock.tick(FPS * training_speed)

    return score


# Boucle principale du programme
def main():
    ai = AI()
    episode = 0
    training_speed = TRAINING_SPEED

    # Tableau pour suivre les performances
    scores = []

    while episode < MAX_EPISODES:
        episode += 1

        # Réduire epsilon progressivement mais plus lentement
        if episode % 50 == 0:  # Changé de 20 à 50 épisodes
            ai.epsilon = max(0.1, ai.epsilon * 0.99)  # Taux de décroissance ralenti à 0.99 (était 0.98)

        # Jouer un épisode
        result = game_episode(ai, episode, training_speed)

        # Gérer les résultats ou commandes renvoyés par l'épisode
        if result == -1:  # Signal pour quitter
            break
        elif isinstance(result, dict) and "action" in result:
            if result["action"] == "speed_up":
                training_speed = min(10, training_speed + 1)  # Maximum 10x
                continue  # Rejouer le même épisode
            elif result["action"] == "speed_down":
                training_speed = max(1, training_speed - 1)  # Minimum 1x
                continue  # Rejouer le même épisode
        else:
            score = result
            scores.append(score)

        # Sauvegarder la Q-table périodiquement
        if episode % 10 == 0:
            ai.save_q_table()

            # Afficher les statistiques
            if len(scores) >= 10:
                avg_score = sum(scores[-10:]) / 10
                print(
                    f"Episode {episode}, Score moyen sur les 10 derniers: {avg_score:.2f}, Vitesse: {training_speed}x"
                )

    # Compte rendu complet sur l'entraînement
    if scores:
        avg_score = sum(scores) / len(scores)
        best_score = max(scores)
        worst_score = min(scores)
        print("\n=== Compte Rendu Complet sur l'Entrainement ===")
        print(f"Nombre d'épisodes: {episode}")
        print(f"Score moyen: {avg_score:.2f}")
        print(f"Meilleur score: {best_score}")
        print(f"Pire score: {worst_score}")
        print(f"Epsilon final: {ai.epsilon:.4f}")
        print("Q-table sauvegardée")

    ai.save_q_table()
    pygame.quit()


if __name__ == "__main__":
    main()

import pygame
import random

# Initialisation de Pygame
pygame.init()

# Constantes du jeu
WIDTH, HEIGHT = 800, 600
BALL_SPEED = 5
PADDLE_SPEED = 6
BALL_SIZE = 20
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Création de la fenêtre
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong AI")

# Classes des objets du jeu
class Paddle:
    def __init__(self, x):
        self.rect = pygame.Rect(x, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
    
    def move(self, up=True):
        if up and self.rect.top > 0:
            self.rect.y -= PADDLE_SPEED
        if not up and self.rect.bottom < HEIGHT:
            self.rect.y += PADDLE_SPEED
    
    def draw(self):
        pygame.draw.rect(screen, WHITE, self.rect)

class Ball:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH // 2, HEIGHT // 2, BALL_SIZE, BALL_SIZE)
        self.vx = BALL_SPEED * random.choice((1, -1))
        self.vy = BALL_SPEED * random.choice((1, -1))
    
    def move(self):
        self.rect.x += self.vx
        self.rect.y += self.vy
        
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.vy *= -1  # Rebond sur les bords
            
        # Respawn de la balle au centre quand elle sort des bordures latérales
        if self.rect.left <= 0 or self.rect.right >= WIDTH:
            self.rect.x = WIDTH // 2 - BALL_SIZE // 2
            self.rect.y = HEIGHT // 2 - BALL_SIZE // 2
            self.vx = BALL_SPEED * random.choice((1, -1))
            self.vy = BALL_SPEED * random.choice((1, -1))

    def draw(self):
        pygame.draw.ellipse(screen, WHITE, self.rect)

# Initialisation des objets
player = Paddle(WIDTH - 30)
pc = Paddle(20)
ball = Ball()

running = True
while running:
    pygame.time.delay(15)
    screen.fill(BLACK)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Contrôles du joueur
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        player.move(up=True)
    if keys[pygame.K_DOWN]:
        player.move(up=False)
    
    # Déplacement de la balle
    ball.move()
    
    # Détection de collision avec les paddles avec effet
    if ball.rect.colliderect(player.rect):
        ball.vx *= -1
        # Calcul de l'effet en fonction de la position d'impact
        relative_impact = (ball.rect.centery - player.rect.top) / player.rect.height
        ball.vy = (relative_impact - 0.5) * 2 * BALL_SPEED  # -BALL_SPEED à +BALL_SPEED
        
    elif ball.rect.colliderect(pc.rect):
        ball.vx *= -1
        # Même effet pour le paddle de l'IA
        relative_impact = (ball.rect.centery - pc.rect.top) / pc.rect.height
        ball.vy = (relative_impact - 0.5) * 2 * BALL_SPEED
    
    # Déplacement de l'IA (suivi basique de la balle)
    if pc.rect.centery < ball.rect.centery:
        pc.move(up=False)
    elif pc.rect.centery > ball.rect.centery:
        pc.move(up=True)
    
    # Dessin des éléments
    player.draw()
    pc.draw()
    ball.draw()
    
    # Mise à jour de l'affichage
    pygame.display.flip()

pygame.quit()
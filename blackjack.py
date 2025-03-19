# Blackjack game
import random

# Définir un paquet complet comme une constante
FULL_DECK = [
    "Carreau2",
    "Carreau3",
    "Carreau4",
    "Carreau5",
    "Carreau6",
    "Carreau7",
    "Carreau8",
    "Carreau9",
    "Carreau10",
    "CarreauJ",
    "CarreauQ",
    "CarreauK",
    "CarreauA",
    "Coeur2",
    "Coeur3",
    "Coeur4",
    "Coeur5",
    "Coeur6",
    "Coeur7",
    "Coeur8",
    "Coeur9",
    "Coeur10",
    "CoeurJ",
    "CoeurQ",
    "CoeurK",
    "CoeurA",
    "Pique2",
    "Pique3",
    "Pique4",
    "Pique5",
    "Pique6",
    "Pique7",
    "Pique8",
    "Pique9",
    "Pique10",
    "PiqueJ",
    "PiqueQ",
    "PiqueK",
    "PiqueA",
    "Trefle2",
    "Trefle3",
    "Trefle4",
    "Trefle5",
    "Trefle6",
    "Trefle7",
    "Trefle8",
    "Trefle9",
    "Trefle10",
    "TrefleJ",
    "TrefleQ",
    "TrefleK",
    "TrefleA",
]

# Liste de travail pour le jeu actuel
cards = FULL_DECK.copy()

def shuffle_deck():
    random.shuffle(cards)
    
def draw_card():
    return cards.pop()

def get_hand_value(hand):
    value = 0
    aces = 0
    for card in hand:
        if card.endswith("10"):
            value += 10
        elif card[-1] in "JQK":
            value += 10
        elif card[-1] == "A":
            aces += 1
            value += 11
        else:
            value += int(card[-1])
    
    # Adjust for aces if needed (they can be worth 1 instead of 11)
    while value > 21 and aces:
        value -= 10
        aces -= 1
    
    return value

def dealer_play(dealer_hand):
    """Croupier tire des cartes jusqu'à atteindre au moins 17 points."""
    while get_hand_value(dealer_hand) < 17:
        dealer_hand.append(draw_card())
    return dealer_hand

def get_card_value(card):
    """Retourne la valeur numérique d'une carte."""
    if card.endswith("10") or card[-1] in "JQK":
        return 10
    elif card[-1] == "A":
        return 11
    else:
        return int(card[-1])

def play_blackjack_round(ai_decision_function):
    """
    Joue une partie de blackjack avec l'IA fournie.
    
    :param ai_decision_function: Une fonction qui prend (player_hand, dealer_card) et retourne "tirer" ou "rester"
    :return: "ia", "croupier", ou "égalité" selon le gagnant
    """
    global cards
    # Réinitialiser le paquet avec toutes les cartes
    cards = FULL_DECK.copy()
    shuffle_deck()
    
    player_hand = []
    dealer_hand = []
    
    # Distribution initiale: 2 cartes chacun
    player_hand.append(draw_card())
    dealer_hand.append(draw_card())
    player_hand.append(draw_card())
    dealer_hand.append(draw_card())
    
    # Tour du joueur (IA)
    while get_hand_value(player_hand) < 21:
        decision = ai_decision_function(player_hand, dealer_hand[0])
        if decision == "tirer":
            player_hand.append(draw_card())
            if get_hand_value(player_hand) > 21:
                return "croupier"
        else:
            break
    
    # Tour du croupier
    dealer_hand = dealer_play(dealer_hand)
    
    # Détermination du gagnant
    player_value = get_hand_value(player_hand)
    dealer_value = get_hand_value(dealer_hand)
    
    if dealer_value > 21:
        return "ia"
    elif player_value > dealer_value:
        return "ia"
    elif dealer_value > player_value:
        return "croupier"
    else:
        return "égalité"

def play_with_display(ai_decision_function):
    """Version avec affichage pour jouer manuellement."""
    global cards
    # Réinitialiser le paquet avec toutes les cartes
    cards = FULL_DECK.copy()
    shuffle_deck()
    
    player_hand = []
    dealer_hand = []
    
    # Distribution initiale: 2 cartes chacun
    player_hand.append(draw_card())
    dealer_hand.append(draw_card())
    player_hand.append(draw_card())
    dealer_hand.append(draw_card())
    
    print(f"Carte visible du croupier: {dealer_hand[0]}")
    print(f"Main de l'IA: {player_hand} (Valeur: {get_hand_value(player_hand)})")
    
    # Tour du joueur (IA)
    while True:
        decision = ai_decision_function(player_hand, dealer_hand[0])
        if decision == "tirer":
            player_hand.append(draw_card())
            print(f"L'IA tire et reçoit {player_hand[-1]}")
            print(f"Main de l'IA: {player_hand} (Valeur: {get_hand_value(player_hand)})")
            if get_hand_value(player_hand) > 21:
                print("L'IA dépasse 21! Le croupier gagne.")
                return "croupier"
        else:
            print("L'IA reste.")
            break
    
    # Tour du croupier
    print(f"Main du croupier: {dealer_hand} (Valeur: {get_hand_value(dealer_hand)})")
    dealer_hand = dealer_play(dealer_hand)
    print(f"Main finale du croupier: {dealer_hand} (Valeur: {get_hand_value(dealer_hand)})")
    
    # Détermination du gagnant
    player_value = get_hand_value(player_hand)
    dealer_value = get_hand_value(dealer_hand)
    
    if dealer_value > 21:
        print("Le croupier dépasse 21! L'IA gagne.")
        return "ia"
    elif player_value > dealer_value:
        print("L'IA gagne!")
        return "ia"
    elif dealer_value > player_value:
        print("Le croupier gagne!")
        return "croupier"
    else:
        print("Égalité!")
        return "égalité"


from strategy_engine import best_gambler_strategy_recommendation, super_computer_strategy_recommendation, \
    evaluate_pre_round_stats, clear_strategy_cache, get_stand_probs
import random


class GameLogic:
    """
    Contains the pure mathematical logic, rules of Blackjack,
    and bankroll management for the game.
    Optimized to track cards purely by their mathematical value (1-10).
    """

    def __init__(self):
        # Default is 2 decks (will be updated immediately at game start from the UI)
        self.num_decks = 2
        self.card_inventory = self._initialize_decks()
        # Generate the first random cut card threshold
        self.shuffle_threshold = self._generate_shuffle_threshold()

    def set_game_config(self, num_decks: int):
        """
        Sets the number of decks (based on user selection) and resets the inventory.
        """
        self.num_decks = num_decks
        self.card_inventory = self._initialize_decks()
        self.shuffle_threshold = self._generate_shuffle_threshold()

    def _generate_shuffle_threshold(self) -> int:
        """
        Generates a randomized Cut Card position based on a Normal Distribution.
        Calculates how many cards should be left in the shoe when a shuffle is triggered.
        """
        total_cards_in_shoe = 52 * self.num_decks

        # Target 75% penetration, meaning a shuffle happens when ~25% of cards are left
        mean_cards_left = total_cards_in_shoe * 0.25
        std_dev = 10  # Human variance: dealer might insert the card +/- 10 cards off target

        # Draw from the normal distribution
        threshold = int(random.gauss(mean_cards_left, std_dev))

        # Clamp the values to ensure realistic casino logic
        # (Must leave at least 10 cards, but cannot shuffle earlier than 50% of the shoe)
        return max(10, min(threshold, int(total_cards_in_shoe * 0.5)))
        # return 50

    def _initialize_decks(self):
        """
        Initializes the card dictionary dynamically based on the number of decks (self.num_decks).
        Each standard deck has 4 cards of each type (1-9), and 16 tens/face cards.
        """
        inventory = {i: 4 * self.num_decks for i in range(1, 10)}
        inventory[10] = 16 * self.num_decks
        return inventory

    def remove_card_from_inventory(self, card_value):
        """
        Subtracts a detected card from the inventory based on its numeric value (1-10).
        """
        if card_value in self.card_inventory and self.card_inventory[card_value] > 0:
            self.card_inventory[card_value] -= 1

    def calculate_hand_value(self, cards_value_list: list) -> tuple:
        """
        Calculates the effective hand total and determines if the hand is soft.

        Args:
            cards_value_list (list): A list of integer card values (e.g., [1, 10]).

        Returns:
            tuple: (effective_sum: int, has_active_ace: bool)
        """
        base_sum = sum(cards_value_list)

        # The logic checks internally if an Ace is present in the detected cards
        has_ace = 1 in cards_value_list

        # Soft hand logic: if there's an Ace and adding 10 doesn't bust, add 10
        if has_ace and (base_sum + 10) <= 21:
            return base_sum + 10, True

        return base_sum, False

    def bank_roll_update(self, current_bankroll: float, current_bet: float, match_result: str) -> float:
        """
        Updates the player's total bankroll based on the round's outcome.
        Assumes the current_bet was ALREADY DEDUCTED from the bankroll in State 2.

        Args:
            current_bankroll (float): The player's current total money.
            current_bet (float): The amount the player bet this round.
            match_result (str): The outcome ("WIN", "BLACKJACK", "PUSH", "LOSE").

        Returns:
            float: The updated bankroll.
        """
        if match_result == "WIN":
            # Return the original bet plus a 1:1 profit
            return current_bankroll + (current_bet * 2)

        elif match_result == "BLACKJACK":
            # Return the original bet plus a 3:2 profit
            return current_bankroll + int(current_bet * 2.5)

        elif match_result == "SURRENDERED":
            return current_bankroll + (current_bet // 2)

        elif match_result == "PUSH":
            # Tie: Return the original bet to the player
            return current_bankroll + current_bet

        elif match_result == "LOSE":
            # Player lost. The bet was already deducted in State 2, so do nothing.
            return current_bankroll

        return current_bankroll

    def get_strategy_recommendation(self, player_cards: list, dealer_upcard_value: int, game_mode: str) -> dict:
        """
        The Router/Adapter. Calls the appropriate engine based on game_mode.
        Passes the exact list of player cards to match the strategy engine's requirements.
        Returns ALL calculated statistics so the UI can build an advanced dashboard.
        """
        if game_mode == "Super Computer":
            result = super_computer_strategy_recommendation(
                player_cards, dealer_upcard_value, self.card_inventory.copy()
            )
            if isinstance(result, str) and result.startswith("Error"):
                print(f"[LOGIC ERROR] Engine returned: {result}")
                return {"action": "ERROR", "stats": None}

            # Since the Super Computer returns a very detailed dictionary (including EV and stats
            # for all possible actions), we just pass the entire dictionary to the UI.
            # We explicitly pull out 'action' to maintain consistency with the other mode.
            return {
                "action": result["action"],
                "stats": result  # Pass the entire dictionary of raw stats (EV, hit_stats, double_stats, etc.)
            }

        elif game_mode == "The Perfect Gambler":
            action = best_gambler_strategy_recommendation(player_cards, dealer_upcard_value, self.num_decks)
            return {
                "action": action,
                "stats": None
            }

        return {
            "action": None,
            "stats": None
        }

    def get_terminal_stats(self, player_effective_sum: int, dealer_upcard_value: int, player_status: str) -> dict:
        """
        Calculates exact final probabilities at the end of a player's turn.
        Used for STAND, BLACKJACK, and BUSTED statuses.
        """
        # 1. Busted - Mathematically guaranteed 100% loss
        if player_status == "BUSTED":
            return {"win": 0.0, "tie": 0.0, "loss": 1.0}

        # 2. Stand or Blackjack (Effective sum 21) - Run simulation against Dealer
        try:
            stats = get_stand_probs(player_effective_sum, dealer_upcard_value, self.card_inventory.copy())
            return stats
        except Exception as e:
            print(f"[LOGIC ERROR] Failed to calculate terminal stats: {e}")
            return {"win": 0.0, "tie": 0.0, "loss": 0.0}

    def get_dealer_forecast(self, hand_list: list) -> dict:
        """
        Calculates the probability distribution for the dealer based on the current hand.
        """
        from strategy_engine import get_dealer_probs
        hard_sum = sum(hand_list)
        has_ace = 1 in hand_list
        return get_dealer_probs(hard_sum, has_ace, self.card_inventory.copy())

    def get_pre_round_evaluation(self) -> dict:
        """
        Passes the current deck state to the strategy engine to calculate
        the player's EV and bet recommendation before the round starts.
        """
        return evaluate_pre_round_stats(self.card_inventory.copy(), self.num_decks)

    def needs_reshuffle(self) -> bool:
        """
        Checks if the shoe has reached the dynamic Cut Card threshold.
        """
        total_cards_left = sum(self.card_inventory.values())
        return total_cards_left <= self.shuffle_threshold

    def reshuffle_decks(self) -> None:
        """
        Resets the card inventory to a fresh state based on the dynamic num_decks.
        Calculates the exact total number of cards for accurate logging.
        """
        self.card_inventory = self._initialize_decks()

        # Crucial: Generate a NEW cut card position for the new shoe
        self.shuffle_threshold = self._generate_shuffle_threshold()

        clear_strategy_cache()
        total_cards = sum(self.card_inventory.values())
        print(
            f"[GAME LOGIC] Decks reshuffled. New Cut Card at {self.shuffle_threshold} cards left. "
            f"Inventory reset to {total_cards} cards.")
